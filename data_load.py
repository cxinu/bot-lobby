"""
cresci-2017 baseline pipeline — Step 1: Load, normalize, EDA
=============================================================
Handles the schema quirks found during inspection:
  - fake_followers/users.csv: missing 'created_at' column name (positionally present)
  - genuine_accounts/users.csv: extra metadata columns (timestamp, crawled_at, test_set_*)
  - traditional_spambots 2/3/4: no tweets.csv (handled gracefully downstream)
"""

import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))


# Core numeric user features available across all subsets
NUMERIC_COLS = [
    "statuses_count", "followers_count", "friends_count",
    "favourites_count", "listed_count"
]

# Boolean/flag features (stored as 0/1 or True/False in the CSVs)
BOOL_COLS = [
    "default_profile", "default_profile_image", "geo_enabled",
    "verified", "protected"
]

PROFILE_COLS = ["description", "url", "location"]  # presence = 1, absent = 0

# ── Dataset registry ─────────────────────────────────────────────────────────
DATASETS = {
    "genuine_accounts":     {"label": 0, "bot_type": "genuine"},
    "fake_followers":       {"label": 1, "bot_type": "fake_followers"},
    "social_spambots_1":    {"label": 1, "bot_type": "social_spambot"},
    "social_spambots_2":    {"label": 1, "bot_type": "social_spambot"},
    "social_spambots_3":    {"label": 1, "bot_type": "social_spambot"},
    "traditional_spambots_1": {"label": 1, "bot_type": "trad_spambot"},
    "traditional_spambots_2": {"label": 1, "bot_type": "trad_spambot"},
    "traditional_spambots_3": {"label": 1, "bot_type": "trad_spambot"},
    "traditional_spambots_4": {"label": 1, "bot_type": "trad_spambot"},
}

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_users(name, meta):
    path = os.path.join(DATA_ROOT, name, "users.csv")
    df = pd.read_csv(path, low_memory=False, encoding="latin-1", dtype={"id": str})

    # fake_followers: 'created_at' missing from header but data is there
    # It appears at column position 8 (0-indexed), between listed_count and url
    if name == "fake_followers" and "created_at" not in df.columns:
        cols = list(df.columns)
        # find where the unnamed column landed — pandas names it something like 'Unnamed: N'
        unnamed = [c for c in cols if str(c).startswith("Unnamed")]
        if unnamed:
            df = df.rename(columns={unnamed[0]: "created_at"})
        else:
            # header shift: insert at pos 8
            df.insert(8, "created_at", df.iloc[:, 8])

    df["label"]    = meta["label"]
    df["bot_type"] = meta["bot_type"]
    df["subset"]   = name
    return df


def load_tweets(name):
    path = os.path.join(DATA_ROOT, name, "tweets.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, low_memory=False, encoding="latin-1",
                       dtype={
                           "id": str,
                           "user_id": str,
                           "retweeted_status_id": str,
                           "in_reply_to_user_id": str
                       },
                       usecols=lambda c: c in [
                           "id", "user_id", "retweeted_status_id",
                           "in_reply_to_user_id", "created_at",
                           "num_hashtags", "num_urls", "num_mentions"
                       ])


# ── Main load ─────────────────────────────────────────────────────────────────
print("Loading users...")
user_frames = []
for name, meta in DATASETS.items():
    df = load_users(name, meta)
    user_frames.append(df)
    print(f"  {name:30s} → {len(df):5d} users")

users = pd.concat(user_frames, ignore_index=True)
print(f"\n  Total: {len(users)} users\n")

print("Loading tweets...")
tweet_frames = []
for name in DATASETS:
    df = load_tweets(name)
    if df is not None:
        df["subset"] = name
        tweet_frames.append(df)
        print(f"  {name:30s} → {len(df):7d} tweets")
    else:
        print(f"  {name:30s} → (no tweets.csv)")

tweets = pd.concat(tweet_frames, ignore_index=True)
print(f"\n  Total: {len(tweets)} tweets\n")


# ── Feature engineering ──────────────────────────────────────────────────────
print("Engineering node features...")

# Numeric: coerce, clip negatives (a few rows have -1 sentinels)
for col in NUMERIC_COLS:
    users[col] = pd.to_numeric(users[col], errors="coerce").clip(lower=0)

# Boolean flags → 0/1
for col in BOOL_COLS:
    users[col] = pd.to_numeric(users[col], errors="coerce").fillna(0).astype(int)

# Profile completeness signals
users["has_description"] = users["description"].notna().astype(int)
users["has_url"]         = users["url"].notna().astype(int)
users["has_location"]    = users["location"].notna().astype(int)
users["profile_complete"] = (
    users["has_description"] + users["has_url"] + users["has_location"]
)

# Account age (days since creation, relative to dataset crawl ~2016-03)
CRAWL_DATE = pd.Timestamp("2016-03-15")
users["created_at_parsed"] = pd.to_datetime(
    users["created_at"], errors="coerce", utc=True
).dt.tz_localize(None)
users["account_age_days"] = (CRAWL_DATE - users["created_at_parsed"]).dt.days.clip(lower=0)

# Derived ratios (add small epsilon to avoid div/0)
EPS = 1e-6
users["ff_ratio"]      = users["followers_count"] / (users["friends_count"] + EPS)
users["engagement"]    = users["favourites_count"] / (users["statuses_count"] + EPS)
users["listed_per_fol"] = users["listed_count"] / (users["followers_count"] + EPS)

print("  Done.\n")


# ── EDA summary ──────────────────────────────────────────────────────────────
print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
dist = users.groupby(["subset", "label"]).size().rename("count")
print(dist.to_string())
print(f"\n  Genuine (0): {(users.label==0).sum():5d}")
print(f"  Bot     (1): {(users.label==1).sum():5d}")
print(f"  Imbalance ratio: 1:{(users.label==1).sum()/(users.label==0).sum():.2f}\n")

print("=" * 60)
print("FEATURE STATS BY CLASS (mean)")
print("=" * 60)
FEAT_COLS = NUMERIC_COLS + ["ff_ratio", "engagement", "account_age_days",
                             "profile_complete", "default_profile_image", "geo_enabled"]
summary = users.groupby("label")[FEAT_COLS].mean().T
summary.columns = ["genuine", "bot"]
summary["ratio_bot_genuine"] = (summary["bot"] / (summary["genuine"] + EPS)).round(2)
print(summary.round(2).to_string())
print()

print("=" * 60)
print("MISSING VALUES IN NODE FEATURES")
print("=" * 60)
missing = users[FEAT_COLS + ["label", "bot_type"]].isnull().sum()
missing = missing[missing > 0]
print(missing.to_string() if len(missing) else "  None — all clean.")
print()

print("=" * 60)
print("TWEET GRAPH EDGE AVAILABILITY")
print("=" * 60)
rt = tweets[tweets["retweeted_status_id"].notna()]
print(f"  Total tweets:         {len(tweets):>8,}")
print(f"  Retweet edges:        {len(rt):>8,} ({100*len(rt)/len(tweets):.1f}%)")
print(f"  Reply edges:          {tweets['in_reply_to_user_id'].notna().sum():>8,}")
rt_users = rt["user_id"].nunique()
print(f"  Unique users in RTs:  {rt_users:>8,}")
print(f"  Users with no tweets: {users['id'].astype(str).isin(tweets['user_id'].astype(str)).value_counts().get(False, 0):>8,}")
print()

print("=" * 60)
print("BOT TYPE BREAKDOWN")
print("=" * 60)
print(users.groupby("bot_type")["id"].count().rename("users").to_string())
print()

# Clean ID and numeric columns to prevent dtype issues and precision loss
print("Cleaning and normalizing column types for saving...")
users["id"] = users["id"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
users["id"] = users["id"].where(users["id"].str.match(r'^\d+$', na=False), None).astype("Int64")

for col in ["id", "user_id", "retweeted_status_id", "in_reply_to_user_id"]:
    if col in tweets.columns:
        tweets[col] = tweets[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        tweets[col] = tweets[col].where(tweets[col].str.match(r'^\d+$', na=False), None).astype("Int64")

for col in ["num_hashtags", "num_urls", "num_mentions"]:
    if col in tweets.columns:
        tweets[col] = pd.to_numeric(tweets[col], errors="coerce").fillna(0).astype(int)

if "created_at" in tweets.columns:
    tweets["created_at"] = tweets["created_at"].astype(str)
    tweets["created_at"] = tweets["created_at"].where(~tweets["created_at"].isin(["nan", "None", "<NA>"]), None)

# Save cleaned user table for next steps
out_path = os.path.join(DATA_ROOT, "users_clean.parquet")
save_cols = ["id", "label", "bot_type", "subset"] + NUMERIC_COLS + \
            ["profile_complete", "default_profile_image", "geo_enabled",
             "has_description", "has_url", "has_location", "account_age_days",
             "ff_ratio", "engagement", "listed_per_fol", "created_at_parsed"]
users[save_cols].to_parquet(out_path, index=False)
print(f"Saved cleaned user table → {out_path}")

out_tweets = os.path.join(DATA_ROOT, "tweets_clean.parquet")
tweets.to_parquet(out_tweets, index=False)
print(f"Saved tweets table      → {out_tweets}")
