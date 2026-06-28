"""
cresci-2017 baseline pipeline — Step 2: Graph construction + structural features
=================================================================================
Builds a directed retweet graph:
  - Nodes  = all 14,368 users (with or without tweets)
  - Edges  = (user_id → original_author_id) for every retweet
             edge weight = number of times A retweeted B

Then computes per-node structural features and merges with node features from
Step 1, producing the full feature matrix for the classifier.

Design decisions:
  - ts2/3/4 users (no tweets) become isolated nodes. Isolation itself is
    informative — we encode it as a binary feature.
  - We use the DIRECTED graph for in/out-degree (asymmetry is signal),
    then an UNDIRECTED projection for clustering coefficient (nx is slow on
    directed clustering, and the undirected version is the standard baseline).
  - PageRank on directed graph (standard).
  - For users appearing in tweets but NOT in users_clean (scraped edges to
    external accounts), we drop those nodes — we only score known accounts.
"""

import pandas as pd
import networkx as nx
import os
import time
import warnings
warnings.filterwarnings("ignore")

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ── Load Step 1 outputs ───────────────────────────────────────────────────────
ts("Loading Step 1 outputs...")
users  = pd.read_parquet(os.path.join(DATA_ROOT, "users_clean.parquet"))
tweets = pd.read_parquet(os.path.join(DATA_ROOT, "tweets_clean.parquet"))

# Canonical set of known user IDs
known_ids = set(users["id"].dropna().astype(str))
ts(f"  Known users: {len(known_ids):,}")

# ── Build edge list from retweets ─────────────────────────────────────────────
ts("Building edge list from retweets...")

# We need: who retweeted whom?
# tweets.csv gives us: user_id (the retweeter) + retweeted_status_id (tweet id)
# BUT we don't have a tweet_id → author_id lookup directly.
# We DO have user_id on every tweet, so we can build it:
#   tweet_id → user_id   from the full tweet table (id col = tweet id)
ts("  Building tweet_id → author_id map (this takes ~30s for 6.6M rows)...")

# 'id' in tweets is the tweet id; 'user_id' is who posted it
tweet_author = (
    tweets[["id", "user_id"]]
    .dropna(subset=["id", "user_id"])
    .drop_duplicates(subset=["id"])
    .set_index("id")["user_id"]
)
ts(f"  Tweet→author map: {len(tweet_author):,} entries")

# Retweets: rows where retweeted_status_id is set
rt = tweets[tweets["retweeted_status_id"].notna()][["user_id", "retweeted_status_id"]].copy()
rt.columns = ["retweeter_id", "original_tweet_id"]
rt = rt.dropna()

# Map original_tweet_id → original_author_id
rt["original_author_id"] = rt["original_tweet_id"].map(tweet_author)
rt = rt.dropna(subset=["original_author_id"])

# Keep only edges where BOTH endpoints are known users
rt["retweeter_id"]      = rt["retweeter_id"].astype(str)
rt["original_author_id"] = rt["original_author_id"].astype(str)
rt = rt[
    rt["retweeter_id"].isin(known_ids) &
    rt["original_author_id"].isin(known_ids)
]

# Self-loops add noise — remove
rt = rt[rt["retweeter_id"] != rt["original_author_id"]]

# Weighted edge list: count(A retweeted B)
edges = (
    rt.groupby(["retweeter_id", "original_author_id"])
    .size()
    .reset_index(name="weight")
)
ts(f"  Edges (weighted): {len(edges):,}")
ts(f"  Unique retweeters: {rt['retweeter_id'].nunique():,}")
ts(f"  Unique retweeted:  {rt['original_author_id'].nunique():,}")

# ── Build NetworkX graphs ─────────────────────────────────────────────────────
ts("Building directed graph...")

G_dir = nx.DiGraph()
G_dir.add_nodes_from(known_ids)                         # ensure ALL users present
G_dir.add_weighted_edges_from(
    zip(edges["retweeter_id"], edges["original_author_id"], edges["weight"])
)
ts(f"  Directed  — nodes: {G_dir.number_of_nodes():,}  edges: {G_dir.number_of_edges():,}")

# Undirected version for clustering + community detection
G_und = nx.Graph()
G_und.add_nodes_from(known_ids)
G_und.add_weighted_edges_from(
    zip(edges["retweeter_id"], edges["original_author_id"], edges["weight"])
)
ts(f"  Undirected — nodes: {G_und.number_of_nodes():,}  edges: {G_und.number_of_edges():,}")

# ── Structural feature computation ────────────────────────────────────────────
ts("Computing in-degree and out-degree...")
in_deg  = dict(G_dir.in_degree(weight="weight"))
out_deg = dict(G_dir.out_degree(weight="weight"))

# Raw degree (unweighted) — different signal from weighted
in_deg_raw  = dict(G_dir.in_degree())
out_deg_raw = dict(G_dir.out_degree())

ts("Computing PageRank (directed)...")
pagerank = nx.pagerank(G_dir, weight="weight", max_iter=200, tol=1e-5)

ts("Computing clustering coefficients (undirected, may take ~1 min)...")
clustering = nx.clustering(G_und, weight="weight")

ts("Computing HITS (authority + hub scores)...")
# HITS can fail on disconnected graphs — wrap safely
try:
    hubs, authorities = nx.hits(G_dir, max_iter=300, tol=1e-5, normalized=True)
except (nx.PowerIterationFailedConvergence, Exception) as e:
    ts(f"  HITS failed ({type(e).__name__}: {e}) — using zeros")
    hubs       = {n: 0.0 for n in G_dir.nodes()}
    authorities = {n: 0.0 for n in G_dir.nodes()}

ts("Computing connected component sizes...")
# For directed: weakly connected components
wcc = {n: 0 for n in G_dir.nodes()}
for comp in nx.weakly_connected_components(G_dir):
    sz = len(comp)
    for n in comp:
        wcc[n] = sz

# ── Ego-network density ───────────────────────────────────────────────────────
# Full ego-network density is O(k²) per node — skip for high-degree nodes
# Instead: use local clustering (already captures this) + degree as proxy
# We'll compute ego density only for nodes with degree ≤ 50 (fast enough)
ts("Computing ego-network density (degree ≤ 50 nodes)...")
ego_density = {}
for n in G_und.nodes():
    d = G_und.degree(n)
    if d == 0:
        ego_density[n] = 0.0
    elif d <= 50:
        ego = nx.ego_graph(G_und, n)
        ego_density[n] = nx.density(ego)
    else:
        # Approximate: use clustering coefficient as proxy for high-degree nodes
        ego_density[n] = clustering.get(n, 0.0)

# ── Assemble graph feature dataframe ─────────────────────────────────────────
ts("Assembling graph feature dataframe...")

node_ids = list(known_ids)

graph_features = pd.DataFrame({
    "id":                node_ids,
    "in_degree_w":       [in_deg.get(n, 0) for n in node_ids],
    "out_degree_w":      [out_deg.get(n, 0) for n in node_ids],
    "in_degree":         [in_deg_raw.get(n, 0) for n in node_ids],
    "out_degree":        [out_deg_raw.get(n, 0) for n in node_ids],
    "pagerank":          [pagerank.get(n, 0.0) for n in node_ids],
    "clustering_coef":   [clustering.get(n, 0.0) for n in node_ids],
    "hub_score":         [hubs.get(n, 0.0) for n in node_ids],
    "authority_score":   [authorities.get(n, 0.0) for n in node_ids],
    "wcc_size":          [wcc.get(n, 1) for n in node_ids],
    "ego_density":       [ego_density.get(n, 0.0) for n in node_ids],
})

# Derived graph ratios
EPS = 1e-6
graph_features["degree_ratio"]   = (
    graph_features["in_degree_w"] / (graph_features["out_degree_w"] + EPS)
)
graph_features["is_isolated"]    = (
    (graph_features["in_degree"] == 0) & (graph_features["out_degree"] == 0)
).astype(int)
graph_features["total_degree"]   = graph_features["in_degree"] + graph_features["out_degree"]

# ── Merge with node features ──────────────────────────────────────────────────
ts("Merging with node features from Step 1...")

users["id"] = users["id"].astype(str)
graph_features["id"] = graph_features["id"].astype(str)

full = users.merge(graph_features, on="id", how="left")

# Fill graph features for users not in tweet graph (isolated, no tweets found)
graph_cols = [c for c in graph_features.columns if c != "id"]
full[graph_cols] = full[graph_cols].fillna(0)
full["is_isolated"] = full["is_isolated"].fillna(1).astype(int)

# ── Impute missing account_age_days (traditional_spambots_1) ─────────────────
ts("Imputing missing account_age_days...")
median_age_by_class = full.groupby("label")["account_age_days"].median()
for label_val, median_val in median_age_by_class.items():
    mask = full["label"].eq(label_val) & full["account_age_days"].isna()
    full.loc[mask, "account_age_days"] = median_val
    count = mask.sum()
    if count > 0:
        ts(f"  Imputed {count} rows (label={label_val}) with median={median_val:.0f} days")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("GRAPH FEATURES — SUMMARY BY CLASS")
print("=" * 60)
graph_summary_cols = ["in_degree_w", "out_degree_w", "pagerank",
                      "clustering_coef", "hub_score", "authority_score",
                      "wcc_size", "is_isolated", "degree_ratio"]
summary = full.groupby("label")[graph_summary_cols].mean().T
summary.columns = ["genuine", "bot"]
summary["ratio"] = (summary["bot"] / (summary["genuine"] + EPS)).round(3)
print(summary.round(5).to_string())
print()

print("=" * 60)
print("ISOLATION RATE BY SUBSET")
print("=" * 60)
iso = full.groupby("subset")["is_isolated"].mean().sort_values(ascending=False)
print(iso.round(3).to_string())
print()

print("=" * 60)
print("FULL FEATURE MATRIX SHAPE")
print("=" * 60)
feature_cols = [
    # Node features
    "statuses_count", "followers_count", "friends_count",
    "favourites_count", "listed_count", "ff_ratio", "engagement",
    "listed_per_fol", "account_age_days", "profile_complete",
    "default_profile_image", "geo_enabled",
    "has_description", "has_url", "has_location",
    # Graph features
    "in_degree_w", "out_degree_w", "in_degree", "out_degree",
    "pagerank", "clustering_coef", "hub_score", "authority_score",
    "wcc_size", "ego_density", "degree_ratio", "is_isolated", "total_degree",
]
print(f"  Rows:     {len(full):,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Missing:  {full[feature_cols].isnull().sum().sum()}")
print(f"  Cols:     {feature_cols}")
print()

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_ROOT, "full_features.parquet")
save_cols = ["id", "label", "bot_type", "subset"] + feature_cols
full[save_cols].to_parquet(out_path, index=False)
ts(f"Saved → {out_path}")

# Also save the graph itself for Step 4 (GNN needs edge index)
import pickle
graph_out = os.path.join(DATA_ROOT, "retweet_graph.pkl")
with open(graph_out, "wb") as f:
    pickle.dump({"G_dir": G_dir, "G_und": G_und, "edges": edges}, f)
ts(f"Saved → {graph_out}")
ts("Step 2 complete.")
