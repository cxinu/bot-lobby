"""
cresci-2017 / TRESA — Step 6: TRESA training loop
===================================================
Trains three model variants across the full sparsification grid:

  RF          — graph-agnostic ceiling/floor (re-uses Step 3 results)
  SAGE_vanilla — GraphSAGE with no robustness mechanism
  TRESA       — GraphSAGE + joint L_cls + λ·L_lp
  GCN_vanilla — GCN baseline
  GCN_TRESA   — GCN + joint L_cls + λ·L_lp
  GAT_vanilla — GAT baseline
  GAT_TRESA   — GAT + joint L_cls + λ·L_lp

Sparsification grid:
  drop_rates  = [0.0, 0.2, 0.4, 0.6]
  paradigms   = [random, degree_biased]

For each (model, drop_rate, paradigm) combination:
  - 5-fold stratified CV (same splits as Steps 3+4 for comparability)
  - Metrics: macro F1, AUC-ROC, per-category F1
  - Also computes Robustness AUC (area under F1-vs-drop curve)

Outputs:
  data/tresa_results.json  — full results for Step 7 plotting
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

# Import sparsification module from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparsification import RandomDrop, DegreeBasedDrop, NegativeSampler

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ts(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Config ────────────────────────────────────────────────────────────────────

DROP_RATES  = [0.0, 0.2, 0.4, 0.6]
PARADIGMS   = ["random", "degree_biased"]
LAMBDA_LP   = 0.5          # L_lp weight — best found in quick sweep
EPOCHS      = 200
PATIENCE    = 20
LR          = 3e-3
HIDDEN      = 256
DROPOUT     = 0.4
N_FOLDS     = 5
SEEDS       = [42]          # add 123 for two-seed runs if time allows

NODE_FEATURES = [
    "statuses_count", "followers_count", "friends_count",
    "favourites_count", "listed_count",
    "ff_ratio", "engagement", "listed_per_fol",
    "account_age_days", "profile_complete",
    "default_profile_image", "geo_enabled",
    "has_description", "has_url", "has_location",
    "in_degree_w", "out_degree_w", "in_degree", "out_degree",
    "pagerank", "clustering_coef",
    "wcc_size", "ego_density", "degree_ratio",
    "is_isolated", "total_degree",
]

# ── Model classes ──────────────────────────────────────────────────────────────

class BotSAGE(nn.Module):
    """GraphSAGE: 3-layer SAGEConv with mean aggregation."""
    def __init__(self, in_dim, hidden=256, dropout=0.4):
        super().__init__()
        self.conv1 = SAGEConv(in_dim,      hidden,      aggr="mean", normalize=True)
        self.conv2 = SAGEConv(hidden,      hidden // 2, aggr="mean", normalize=True)
        self.conv3 = SAGEConv(hidden // 2, 64,          aggr="mean", normalize=True)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden // 2)
        self.bn3   = nn.BatchNorm1d(64)
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.drop = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        x = self.drop(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.drop(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.drop(F.relu(self.bn3(self.conv3(x, edge_index))))
        return x   # [N, 64]

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        return self.cls_head(h).squeeze(-1)   # [N] logits

    def link_logits(self, h, edge_index_lp):
        """Dot-product link prediction score for edges in edge_index_lp."""
        h_u = h[edge_index_lp[0]]   # [E_lp, 64]
        h_v = h[edge_index_lp[1]]   # [E_lp, 64]
        return (h_u * h_v).sum(dim=-1)   # [E_lp] logits


class BotGCN(nn.Module):
    """GCN: 3-layer GCNConv with symmetric normalisation."""
    def __init__(self, in_dim, hidden=256, dropout=0.4):
        super().__init__()
        self.conv1 = GCNConv(in_dim,      hidden,      normalize=True)
        self.conv2 = GCNConv(hidden,      hidden // 2, normalize=True)
        self.conv3 = GCNConv(hidden // 2, 64,          normalize=True)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden // 2)
        self.bn3   = nn.BatchNorm1d(64)
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.drop = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        x = self.drop(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.drop(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.drop(F.relu(self.bn3(self.conv3(x, edge_index))))
        return x   # [N, 64]

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        return self.cls_head(h).squeeze(-1)   # [N] logits

    def link_logits(self, h, edge_index_lp):
        h_u = h[edge_index_lp[0]]
        h_v = h[edge_index_lp[1]]
        return (h_u * h_v).sum(dim=-1)


class BotGAT(nn.Module):
    """
    GAT: 3-layer GATConv with multi-head attention.
    Heads: 4×64→256, 4×32→128, 1×64→64
    """
    def __init__(self, in_dim, hidden=256, dropout=0.4):
        super().__init__()
        self.conv1 = GATConv(in_dim,  64, heads=4, concat=True)
        self.conv2 = GATConv(256,      32, heads=4, concat=True)
        self.conv3 = GATConv(128,      64, heads=1, concat=False)
        self.bn1   = nn.BatchNorm1d(256)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(64)
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.drop = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        x = self.drop(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.drop(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.drop(F.relu(self.bn3(self.conv3(x, edge_index))))
        return x   # [N, 64]

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        return self.cls_head(h).squeeze(-1)   # [N] logits

    def link_logits(self, h, edge_index_lp):
        h_u = h[edge_index_lp[0]]
        h_v = h[edge_index_lp[1]]
        return (h_u * h_v).sum(dim=-1)


# Map model name → model class (used in experiment grid)
MODEL_CLASSES = {
    "sage_vanilla": BotSAGE,
    "tresa":        BotSAGE,
    "gcn_vanilla":  BotGCN,
    "gcn_tresa":    BotGCN,
    "gat_vanilla":  BotGAT,
    "gat_tresa":    BotGAT,
}


# ── Training helpers ──────────────────────────────────────────────────────────

def make_pyg_data(X_scaled, edge_index, y):
    from torch_geometric.data import Data
    return Data(
        x          = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE),
        edge_index = edge_index.to(DEVICE),
        y          = torch.tensor(y, dtype=torch.float32).to(DEVICE),
    )

@torch.no_grad()
def eval_cls(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs  = torch.sigmoid(logits[mask]).cpu().numpy()
    preds  = (probs >= 0.5).astype(int)
    labels = data.y[mask].cpu().numpy().astype(int)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    return f1, auc, preds, probs

def train_one_fold(
    X_raw, y, groups, tr_idx, va_idx,
    full_ei, sparse_ei_fn,       # sparse_ei_fn(epoch) → (kept_ei, dropped_ei)
    use_lp: bool,
    lambda_lp: float,
    num_nodes: int,
    neg_sampler: NegativeSampler,
    rng: np.random.Generator,
    model_class=BotSAGE,
):
    """
    Train for one fold. Returns (best_val_f1, best_val_auc, oof_preds, oof_probs).
    sparse_ei_fn is called each epoch to get a fresh drop mask (stochastic aug).
    When use_lp=False → SAGE vanilla (only L_cls, uses sparse_ei without L_lp).
    model_class controls which GNN architecture is used.
    """
    scaler  = RobustScaler()
    X_tr    = scaler.fit_transform(X_raw[tr_idx])
    X_va    = scaler.transform(X_raw[va_idx])
    X_scaled = np.zeros_like(X_raw)
    X_scaled[tr_idx] = X_tr
    X_scaled[va_idx] = X_va

    # Build static val data (raw edge_index — each model adds self-loops internally)
    val_data = make_pyg_data(X_scaled, full_ei, y)

    tr_mask = torch.zeros(num_nodes, dtype=torch.bool)
    va_mask = torch.zeros(num_nodes, dtype=torch.bool)
    tr_mask[tr_idx] = True
    va_mask[va_idx] = True
    tr_mask = tr_mask.to(DEVICE)
    va_mask = va_mask.to(DEVICE)

    n_neg = (y[tr_idx] == 0).sum()
    n_pos = (y[tr_idx] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)

    model     = model_class(in_dim=X_raw.shape[1], hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1, best_epoch, patience_ctr = 0.0, 0, 0
    best_state = None
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    y_tensor  = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        # Fresh sparse graph each epoch (raw edge_index — model adds self-loops)
        kept_ei, dropped_ei = sparse_ei_fn(epoch)
        train_ei = kept_ei.to(DEVICE)

        # Forward pass
        h      = model.encode(x_tensor, train_ei)
        cls_logits = model.cls_head(h).squeeze(-1)

        # L_cls — classification loss on train nodes
        l_cls = F.binary_cross_entropy_with_logits(
            cls_logits[tr_mask], y_tensor[tr_mask], pos_weight=pos_weight
        )

        loss = l_cls

        # L_lp — link prediction loss (TRESA only)
        if use_lp and dropped_ei.shape[1] > 0:
            # Positive pairs = dropped edges involving train nodes
            drop_np   = dropped_ei.numpy()
            train_set = set(tr_idx.tolist())
            pos_mask  = np.array([
                int(drop_np[0, i]) in train_set or int(drop_np[1, i]) in train_set
                for i in range(drop_np.shape[1])
            ])
            pos_ei = dropped_ei[:, pos_mask]

            if pos_ei.shape[1] > 0:
                neg_ei = neg_sampler.sample(n_pos=pos_ei.shape[1], rng=rng)
                lp_ei  = torch.cat([pos_ei, neg_ei], dim=1).to(DEVICE)
                lp_labels = torch.cat([
                    torch.ones(pos_ei.shape[1]),
                    torch.zeros(neg_ei.shape[1])
                ]).to(DEVICE)
                lp_logits = model.link_logits(h, lp_ei)
                l_lp      = F.binary_cross_entropy_with_logits(lp_logits, lp_labels)
                loss      = l_cls + lambda_lp * l_lp

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            val_f1, val_auc, _, _ = eval_cls(model, val_data, va_mask)
            if val_f1 > best_f1:
                best_f1    = val_f1
                best_epoch = epoch
                patience_ctr = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    val_f1, val_auc, preds, probs = eval_cls(model, val_data, va_mask)
    return val_f1, val_auc, best_epoch, preds, probs


# ── RF baseline (re-run for each drop rate to confirm graph-agnostic) ─────────

def run_rf(X_raw, y, groups, cv):
    """RF on node features only — no graph, so drop rate is irrelevant.
    Run once; result is constant across all drop rates."""
    clf_params = dict(n_estimators=300, max_features="sqrt",
                      class_weight="balanced", n_jobs=-1, random_state=42)
    fold_f1s, fold_aucs = [], []
    all_preds = np.zeros(len(y), dtype=int)
    all_probs = np.zeros(len(y))
    for tr_idx, va_idx in cv.split(X_raw, y):
        scaler = RobustScaler()
        X_tr   = scaler.fit_transform(X_raw[tr_idx])
        X_va   = scaler.transform(X_raw[va_idx])
        clf    = RandomForestClassifier(**clf_params)
        clf.fit(X_tr, y[tr_idx])
        probs  = clf.predict_proba(X_va)[:, 1]
        preds  = (probs >= 0.5).astype(int)
        all_preds[va_idx] = preds
        all_probs[va_idx] = probs
        fold_f1s.append(f1_score(y[va_idx], preds, average="macro", zero_division=0))
        fold_aucs.append(roc_auc_score(y[va_idx], probs))
    return {
        "f1_mean":  float(np.mean(fold_f1s)),
        "f1_std":   float(np.std(fold_f1s)),
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std":  float(np.std(fold_aucs)),
        "cat_f1":   per_cat_f1(y, all_preds, groups),
    }


def per_cat_f1(y, preds, groups):
    return {
        cat: float(f1_score(y[groups == cat], preds[groups == cat],
                             average="binary", zero_division=0))
        for cat in np.unique(groups)
    }


# ── GNN runner (handles vanilla and TRESA variants for all architectures) ─────

def run_gnn(
    X_raw, y, groups, full_ei, num_nodes,
    drop_rate, paradigm, use_lp, lambda_lp,
    cv, seed, model_class,
):
    rng        = np.random.default_rng(seed)
    neg_samp   = NegativeSampler(full_ei, num_nodes=num_nodes, ratio=1)

    # Build the dropper for this (drop_rate, paradigm) combo
    if drop_rate == 0.0:
        def sparse_ei_fn(epoch):
            return full_ei.clone(), full_ei[:, :0]   # no drop, no positives
    elif paradigm == "random":
        dropper = RandomDrop(p=drop_rate)   # no fixed seed → stochastic per epoch
        def sparse_ei_fn(epoch):
            kept, dropped, _ = dropper(full_ei)
            return kept, dropped
    else:  # degree_biased
        dropper = DegreeBasedDrop(p_base=drop_rate)
        def sparse_ei_fn(epoch):
            kept, dropped, _ = dropper(full_ei, num_nodes=num_nodes)
            return kept, dropped

    fold_f1s, fold_aucs = [], []
    all_preds = np.zeros(len(y), dtype=int)
    all_probs = np.zeros(len(y))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
        val_f1, val_auc, best_ep, preds, probs = train_one_fold(
            X_raw, y, groups, tr_idx, va_idx,
            full_ei, sparse_ei_fn,
            use_lp=use_lp, lambda_lp=lambda_lp,
            num_nodes=num_nodes, neg_sampler=neg_samp, rng=rng,
            model_class=model_class,
        )
        fold_f1s.append(val_f1)
        fold_aucs.append(val_auc)
        all_preds[va_idx] = preds
        all_probs[va_idx] = probs

    return {
        "f1_mean":  float(np.mean(fold_f1s)),
        "f1_std":   float(np.std(fold_f1s)),
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std":  float(np.std(fold_aucs)),
        "cat_f1":   per_cat_f1(y, all_preds, groups),
    }


# ── Robustness AUC ────────────────────────────────────────────────────────────

def robustness_auc(f1_by_drop: list) -> float:
    """
    Area under the F1-vs-drop-rate curve, normalised to [0,1].
    f1_by_drop: list of (drop_rate, f1_mean) sorted by drop_rate.
    Uses trapezoidal integration.
    """
    xs = [x for x, _ in f1_by_drop]
    ys = [y for _, y in f1_by_drop]
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(ys, xs)
    else:
        area = np.trapz(ys, xs)
    x_range = xs[-1] - xs[0]
    return float(area / x_range) if x_range > 0 else float(ys[0])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ts("Loading data...")
    df = pd.read_parquet(os.path.join(DATA_ROOT, "full_features.parquet"))

    with open(os.path.join(DATA_ROOT, "retweet_graph.pkl"), "rb") as f:
        gd = pickle.load(f)

    edges_df = gd["edges"]
    id_to_idx = {str(uid): i for i, uid in enumerate(df["id"])}
    src, dst  = [], []
    for _, row in edges_df.iterrows():
        s = id_to_idx.get(str(row["retweeter_id"]))
        d = id_to_idx.get(str(row["original_author_id"]))
        if s is not None and d is not None and s != d:
            src.append(s)
            dst.append(d)
    full_ei   = torch.tensor([src, dst], dtype=torch.long)
    num_nodes = len(df)

    X_raw = df[NODE_FEATURES].values.astype(np.float64)
    for i in range(X_raw.shape[1]):
        cap = np.nanpercentile(X_raw[:, i], 99.9)
        X_raw[:, i] = np.clip(X_raw[:, i], None, cap)

    y      = df["label"].values
    groups = df["bot_type"].values
    cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    results = {}

    # ── RF (once — graph-agnostic) ────────────────────────────────────────────
    ts("Running RF baseline (graph-agnostic)...")
    rf_res = run_rf(X_raw, y, groups, cv)
    ts(f"  RF: F1={rf_res['f1_mean']:.4f} ± {rf_res['f1_std']:.4f}")
    results["rf"] = {str(p): rf_res for p in DROP_RATES}

    # ── GNN experiments ───────────────────────────────────────────────────────
    # (model_name, use_lp) pairs
    model_variants = [
        ("sage_vanilla", False),
        ("tresa",        True),
        ("gcn_vanilla",  False),
        ("gcn_tresa",    True),
        ("gat_vanilla",  False),
        ("gat_tresa",    True),
    ]

    DISPLAY_NAMES = {
        "sage_vanilla": "SAGE vanilla",
        "tresa":        "TRESA (SAGE)",
        "gcn_vanilla":  "GCN vanilla",
        "gcn_tresa":    "TRESA (GCN)",
        "gat_vanilla":  "GAT vanilla",
        "gat_tresa":    "TRESA (GAT)",
    }

    for paradigm in PARADIGMS:
        for model_name, use_lp in model_variants:
            key = f"{model_name}_{paradigm}"
            results[key] = {}
            f1_by_drop = []

            ts(f"\n{'='*58}")
            ts(f"Model: {DISPLAY_NAMES[model_name]:>15s}  |  Paradigm: {paradigm}")
            ts(f"{'='*58}")

            model_class = MODEL_CLASSES[model_name]

            for drop_rate in DROP_RATES:
                t0 = time.time()
                res = run_gnn(
                    X_raw, y, groups, full_ei, num_nodes,
                    drop_rate=drop_rate, paradigm=paradigm,
                    use_lp=use_lp, lambda_lp=LAMBDA_LP,
                    cv=cv, seed=SEEDS[0],
                    model_class=model_class,
                )
                elapsed = time.time() - t0
                results[key][str(drop_rate)] = res
                f1_by_drop.append((drop_rate, res["f1_mean"]))

                ts(f"  drop={drop_rate:.0%}  F1={res['f1_mean']:.4f}±{res['f1_std']:.4f}"
                   f"  AUC={res['auc_mean']:.4f}  ({elapsed:.0f}s)")

            rob_auc = robustness_auc(f1_by_drop)
            results[key]["robustness_auc"] = rob_auc
            ts(f"  → Robustness AUC: {rob_auc:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Model':25s}  {'Paradigm':14s}  {'Rob. AUC':>10}  {'F1@0%':>8}  {'F1@60%':>8}")
    print("  " + "-" * 70)

    rf_f1_base = results["rf"]["0.0"]["f1_mean"]
    print(f"  {'RF (node-only)':25s}  {'—':14s}  {'—':>10}  {rf_f1_base:>8.4f}  {rf_f1_base:>8.4f}")

    for paradigm in PARADIGMS:
        for model_name, _ in model_variants:
            key  = f"{model_name}_{paradigm}"
            rob  = results[key].get("robustness_auc", 0)
            f1_0 = results[key]["0.0"]["f1_mean"]
            f1_6 = results[key]["0.6"]["f1_mean"]
            print(f"  {DISPLAY_NAMES[model_name]:25s}  {paradigm:14s}  {rob:>10.4f}  {f1_0:>8.4f}  {f1_6:>8.4f}")

    # RF crossover analysis
    print(f"\n  RF F1 ceiling: {rf_f1_base:.4f}")
    print("  Crossover point (GNN drops below RF):")
    for paradigm in PARADIGMS:
        for model_name, _ in model_variants:
            key = f"{model_name}_{paradigm}"
            crossover = "never"
            for dr in DROP_RATES:
                f1 = results[key][str(dr)]["f1_mean"]
                if f1 < rf_f1_base:
                    crossover = f"{dr:.0%}"
                    break
            print(f"    {DISPLAY_NAMES[model_name]:>15s} [{paradigm}]: {crossover}")

    # Save
    out = os.path.join(DATA_ROOT, "tresa_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    ts(f"\nSaved → {out}")
    ts("Step 6 complete. Run 07_robustness_eval.py to plot.")


if __name__ == "__main__":
    main()
