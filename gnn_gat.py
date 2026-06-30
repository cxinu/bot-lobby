"""
cresci-2017 baseline pipeline — GAT baseline
=============================================
Architecture: GAT (Veličković et al. 2018)
  - Multi-head attention: 4 heads × 64 → 256, 4 heads × 32 → 128, 1 head × 64
  - 3 layers, ReLU + dropout, binary output head
  - Self-loops added for isolated nodes (same as GraphSAGE)
  - Node features: same 26-dim node+graph vector as RF ablation B

Training:
  - Same 5-fold stratified split as RF (same random_state=42) for fair comparison
  - AdamW + cosine LR schedule, early stopping on val F1 (patience=20)
  - Class-weighted BCE loss (mirrors RF's balanced class_weight)
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
import warnings
import time
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Feature set (mirrors RF ablation B) ──────────────────────────────────────
FEATURES = [
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

# ── Model ─────────────────────────────────────────────────────────────────────
class BotGAT(nn.Module):
    """
    3-layer GAT with:
      - 4 attention heads in first two layers, 1 head in final layer
      - BatchNorm after each conv
      - Dropout for regularisation
    """
    def __init__(self, in_dim, hidden=256, dropout=0.4):
        super().__init__()
        self.conv1 = GATConv(in_dim,  64, heads=4, concat=True)
        self.conv2 = GATConv(256,      32, heads=4, concat=True)
        self.conv3 = GATConv(128,      64, heads=1, concat=False)
        self.bn1   = nn.BatchNorm1d(256)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(64)
        self.head  = nn.Linear(64, 1)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.drop(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.drop(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.drop(F.relu(self.bn3(self.conv3(x, edge_index))))
        return self.head(x).squeeze(-1)


# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(os.path.join(DATA_ROOT, "full_features.parquet"))

with open(os.path.join(DATA_ROOT, "retweet_graph.pkl"), "rb") as f:
    graph_data = pickle.load(f)
edges_df = graph_data["edges"]

df = df.reset_index(drop=True)
id_to_idx = {str(uid): i for i, uid in enumerate(df["id"])}

src, dst = [], []
for _, row in edges_df.iterrows():
    s = id_to_idx.get(str(row["retweeter_id"]))
    d = id_to_idx.get(str(row["original_author_id"]))
    if s is not None and d is not None:
        src.append(s)
        dst.append(d)

edge_index = torch.tensor([src, dst], dtype=torch.long)
# Add self-loops so isolated nodes still receive their own features in aggregation
edge_index, _ = add_self_loops(edge_index, num_nodes=len(df))
print(f"  Nodes: {len(df):,}  |  Edges (incl. self-loops): {edge_index.shape[1]:,}")

# Feature matrix
X_raw = df[FEATURES].values.astype(np.float64)
for i in range(X_raw.shape[1]):
    cap = np.nanpercentile(X_raw[:, i], 99.9)
    X_raw[:, i] = np.clip(X_raw[:, i], None, cap)

y_all     = df["label"].values
groups    = df["bot_type"].values
bot_types = df["bot_type"].values


# ── Training helpers ──────────────────────────────────────────────────────────
def make_data(X_scaled, edge_index, y):
    return Data(
        x          = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE),
        edge_index = edge_index.to(DEVICE),
        y          = torch.tensor(y, dtype=torch.float32).to(DEVICE),
    )


def train_epoch(model, data, optimizer, train_mask, pos_weight):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.binary_cross_entropy_with_logits(
        logits[train_mask], data.y[train_mask],
        pos_weight=pos_weight
    )
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs  = torch.sigmoid(logits[mask]).cpu().numpy()
    preds  = (probs >= 0.5).astype(int)
    labels = data.y[mask].cpu().numpy().astype(int)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    return f1, auc, preds, probs


# ── Cross-validation ──────────────────────────────────────────────────────────
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

EPOCHS   = 200
PATIENCE = 20
LR       = 3e-3
HIDDEN   = 256
DROPOUT  = 0.4

fold_metrics   = []
all_oof_preds  = np.zeros(len(df), dtype=int)
all_oof_probs  = np.zeros(len(df))

print(f"\nTraining GAT ({EPOCHS} epochs max, patience={PATIENCE})...")
print(f"  Hidden={HIDDEN}  Dropout={DROPOUT}  LR={LR}\n")

for fold, (tr_idx, va_idx) in enumerate(CV.split(X_raw, y_all)):
    t0 = time.time()

    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X_raw[tr_idx])
    X_va = scaler.transform(X_raw[va_idx])
    X_scaled = np.zeros_like(X_raw)
    X_scaled[tr_idx] = X_tr
    X_scaled[va_idx] = X_va

    data = make_data(X_scaled, edge_index, y_all)

    tr_mask = torch.zeros(len(df), dtype=torch.bool)
    va_mask = torch.zeros(len(df), dtype=torch.bool)
    tr_mask[tr_idx] = True
    va_mask[va_idx] = True

    n_neg = (y_all[tr_idx] == 0).sum()
    n_pos = (y_all[tr_idx] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)

    model     = BotGAT(in_dim=len(FEATURES), hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1, best_epoch, patience_ctr = 0.0, 0, 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, data, optimizer, tr_mask, pos_weight)
        scheduler.step()

        if epoch % 5 == 0:
            val_f1, val_auc, _, _ = evaluate(model, data, va_mask)
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
    val_f1, val_auc, va_preds, va_probs = evaluate(model, data, va_mask)

    all_oof_preds[va_idx] = va_preds
    all_oof_probs[va_idx] = va_probs

    elapsed = time.time() - t0
    print(f"  Fold {fold+1}: best_epoch={best_epoch:3d}  "
          f"f1={val_f1:.4f}  auc={val_auc:.4f}  ({elapsed:.1f}s)")

    fold_metrics.append({"fold": fold+1, "f1_macro": val_f1, "auc_roc": val_auc})

# ── Aggregate results ─────────────────────────────────────────────────────────
fm = pd.DataFrame(fold_metrics)
gat_results = {
    "f1_macro":  {"mean": fm["f1_macro"].mean(), "std": fm["f1_macro"].std()},
    "auc_roc":   {"mean": fm["auc_roc"].mean(),  "std": fm["auc_roc"].std()},
    "accuracy":  {"mean": accuracy_score(y_all, all_oof_preds)},
}

print(f"\n{'='*60}")
print("GAT RESULTS")
print("="*60)
print(f"  F1 (macro): {gat_results['f1_macro']['mean']:.4f} ± {gat_results['f1_macro']['std']:.4f}")
print(f"  AUC-ROC:    {gat_results['auc_roc']['mean']:.4f} ± {gat_results['auc_roc']['std']:.4f}")
print(f"  Accuracy:   {gat_results['accuracy']['mean']:.4f}")

print("\n  Per-class report (OOF):")
print(classification_report(y_all, all_oof_preds,
                             target_names=["genuine", "bot"], digits=4))

print("  Per-category F1 (OOF):")
cat_f1_gat = {}
for cat in sorted(set(bot_types)):
    mask = bot_types == cat
    f1 = f1_score(y_all[mask], all_oof_preds[mask], average="binary", zero_division=0)
    cat_f1_gat[cat] = float(f1)
    bar = "█" * int(f1 * 30)
    print(f"    {cat:25s}: {f1:.4f}  {bar}")

cm = confusion_matrix(y_all, all_oof_preds)
print("\n  Confusion matrix (OOF):")
print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

# ── Final five-way comparison ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL BASELINE COMPARISON")
print("="*60)

rf_results = json.load(open(os.path.join(DATA_ROOT, "baseline_results.json")))

rows = [
    ("RF node-only",   rf_results["node_only"]["metrics"]),
    ("RF node+graph",  rf_results["node+graph"]["metrics"]),
]
print(f"  {'Model':20s}  {'F1 macro':>12s}  {'AUC-ROC':>12s}  {'Accuracy':>10s}")
print("  " + "-" * 60)
for name, m in rows:
    f1  = m["f1_macro"]
    auc = m["auc_roc"]
    acc = m["accuracy"]
    print(f"  {name:20s}  {f1['mean']:.4f}±{f1['std']:.4f}  "
          f"{auc['mean']:.4f}±{auc['std']:.4f}  {acc['mean']:.4f}")

# GraphSAGE row
sg_m  = rf_results["graphsage"]["metrics"]
sg_f1 = sg_m["f1_macro"]
sg_au = sg_m["auc_roc"]
sg_ac = sg_m["accuracy"]
print(f"  {'GraphSAGE':20s}  {sg_f1['mean']:.4f}±{sg_f1['std']:.4f}  "
      f"{sg_au['mean']:.4f}±{sg_au['std']:.4f}  {sg_ac['mean']:.4f}")

# GCN row (if available)
if "gcn" in rf_results:
    gc_m  = rf_results["gcn"]["metrics"]
    gc_f1 = gc_m["f1_macro"]
    gc_au = gc_m["auc_roc"]
    gc_ac = gc_m["accuracy"]
    print(f"  {'GCN':20s}  {gc_f1['mean']:.4f}±{gc_f1['std']:.4f}  "
          f"{gc_au['mean']:.4f}±{gc_au['std']:.4f}  {gc_ac['mean']:.4f}")

# GAT row
gt_f1 = gat_results["f1_macro"]
gt_au = gat_results["auc_roc"]
gt_ac = gat_results["accuracy"]
print(f"  {'GAT':20s}  {gt_f1['mean']:.4f}±{gt_f1['std']:.4f}  "
      f"{gt_au['mean']:.4f}±{gt_au['std']:.4f}  {gt_ac['mean']:.4f}")

print("\n  Per-category F1 comparison:")
all_cats = sorted(set(bot_types))
print(f"  {'category':25s}  {'RF-node':>10s}  {'RF+graph':>10s}  {'GraphSAGE':>10s}  {'GCN':>10s}  {'GAT':>10s}")
print("  " + "-" * 95)
for cat in all_cats:
    rf_n = rf_results["node_only"]["category_f1"].get(cat, 0.0)
    rf_g = rf_results["node+graph"]["category_f1"].get(cat, 0.0)
    sg   = rf_results["graphsage"]["category_f1"].get(cat, 0.0)
    gc   = rf_results.get("gcn", {}).get("category_f1", {}).get(cat, 0.0)
    gt   = cat_f1_gat.get(cat, 0.0)
    print(f"  {cat:25s}  {rf_n:10.4f}  {rf_g:10.4f}  {sg:10.4f}  {gc:10.4f}  {gt:10.4f}")

# Save updated results
rf_results["gat"] = {
    "metrics": {k: {"mean": float(v["mean"]), "std": float(v.get("std", 0))}
                for k, v in gat_results.items()},
    "category_f1": cat_f1_gat,
}
out = os.path.join(DATA_ROOT, "baseline_results.json")
with open(out, "w") as f:
    json.dump(rf_results, f, indent=2)
print(f"\nSaved → {out}")
print("\nBaseline pipeline complete.")
