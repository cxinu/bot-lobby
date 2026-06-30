"""
MGTAB pipeline — Baseline + TRESA robustness experiment
========================================================
Validates the density prerequisite finding from cresci-2017:
  "Graph robustness techniques only matter when the graph is connected."

MGTAB stats:
  10,199 nodes · 1,700,108 edges · 7 relation types
  density ~1.6%  (vs cresci-2017: 0.001%)
  labels: 7,451 genuine / 2,748 bot

Pipeline:
  1. Load MGTAB tensors, inspect graph structure
  2. RF baseline (node features only — 788-dim)
  3. GraphSAGE vanilla across sparsification grid
  4. TRESA (SAGE + L_lp) across sparsification grid
  5. Robustness AUC comparison → validate/refute cresci-2017 finding

The critical comparison:
  cresci-2017: SAGE flat under drop (already graph-agnostic)
               TRESA hurts (L_lp has no signal to work with)
  MGTAB:       SAGE should degrade under drop (graph actually matters)
               TRESA should degrade more slowly (L_lp now has signal)

If that pattern holds → density prerequisite confirmed.
If not → finding is dataset-specific, not structural.

Usage:
  python mgtab_pipeline.py                        # MGTAB/ in current dir
  MGTAB_DIR=./MGTAB python mgtab_pipeline.py
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparsification import RandomDrop, DegreeBasedDrop, NegativeSampler

MGTAB_DIR = os.environ.get("MGTAB_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "MGTAB"))
OUT_DIR = os.environ.get("OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ts(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
DROP_RATES  = [0.0, 0.2, 0.4, 0.6]
PARADIGMS   = ["random", "degree_biased"]
LAMBDA_LP   = 0.5
EPOCHS      = 200
PATIENCE    = 20
LR          = 3e-3
HIDDEN      = 256
DROPOUT     = 0.4
N_FOLDS     = 5
SEED        = 42

# 788-dim features are large — project down before GNN to save VRAM
PROJ_DIM    = 128    # Linear projection: 788 → 128

# ── Load MGTAB ────────────────────────────────────────────────────────────────
ts("Loading MGTAB tensors...")
edge_index  = torch.load(os.path.join(MGTAB_DIR, "edge_index.pt"),
                         map_location="cpu", weights_only=True)
edge_type   = torch.load(os.path.join(MGTAB_DIR, "edge_type.pt"),
                         map_location="cpu", weights_only=True)
edge_weight = torch.load(os.path.join(MGTAB_DIR, "edge_weight.pt"),
                         map_location="cpu", weights_only=True)
features    = torch.load(os.path.join(MGTAB_DIR, "features.pt"),
                         map_location="cpu", weights_only=True)
labels      = torch.load(os.path.join(MGTAB_DIR, "labels_bot.pt"),
                         map_location="cpu", weights_only=True)

N          = features.shape[0]
E          = edge_index.shape[1]
n_types    = int(edge_type.max().item()) + 1
n_bot      = int((labels == 1).sum())
n_genuine  = int((labels == 0).sum())

ts(f"  Nodes: {N:,}  |  Edges: {E:,}  |  Relation types: {n_types}")
ts(f"  Genuine: {n_genuine:,}  |  Bot: {n_bot:,}  |  Ratio: 1:{n_bot/n_genuine:.2f}")
ts(f"  Feature dim: {features.shape[1]}  |  Device: {DEVICE}")

# Edge type distribution
print("\n  Edge type distribution:")
for t in range(n_types):
    cnt = int((edge_type == t).sum())
    bar = "█" * int(cnt / E * 40)
    print(f"    type {t}: {cnt:>7,}  ({100*cnt/E:.1f}%)  {bar}")

# Degree stats
deg = torch.zeros(N, dtype=torch.long)
deg.scatter_add_(0, edge_index[0], torch.ones(E, dtype=torch.long))
deg.scatter_add_(0, edge_index[1], torch.ones(E, dtype=torch.long))
isolated = int((deg == 0).sum())
density  = E / (N * (N - 1))
print(f"\n  Isolated nodes: {isolated} ({100*isolated/N:.1f}%)")
print(f"  Graph density:  {density:.5f}  ({density/0.000099:.0f}× cresci-2017)")
print(f"  Avg degree:     {deg.float().mean():.1f}")
print(f"  Max degree:     {deg.max().item()}")

# ── Feature matrix ────────────────────────────────────────────────────────────
X_np = features.numpy().astype(np.float64)
y_np = labels.numpy().astype(int)

# Clip 99.9th pct outliers (same protocol as cresci pipeline)
for i in range(X_np.shape[1]):
    cap = np.nanpercentile(X_np[:, i], 99.9)
    X_np[:, i] = np.clip(X_np[:, i], None, cap)

# ── Helpers ───────────────────────────────────────────────────────────────────

def per_cat_f1(y_true, y_pred):
    """Binary bot F1, genuine F1, macro F1."""
    return {
        "bot":     float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "genuine": float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)),
        "macro":   float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def robustness_auc(f1_by_drop):
    xs = [x for x, _ in f1_by_drop]
    ys = [y for _, y in f1_by_drop]
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(ys, xs)
    else:
        area = np.trapz(ys, xs)
    xr   = xs[-1] - xs[0]
    return float(area / xr) if xr > 0 else float(ys[0])


# ── RF baseline ───────────────────────────────────────────────────────────────
ts("\nRunning RF baseline (788-dim features, no graph)...")
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

rf_f1s, rf_aucs = [], []
rf_preds_oof = np.zeros(N, dtype=int)
rf_probs_oof = np.zeros(N)

for tr_idx, va_idx in cv.split(X_np, y_np):
    scaler = RobustScaler()
    X_tr   = scaler.fit_transform(X_np[tr_idx])
    X_va   = scaler.transform(X_np[va_idx])
    clf    = RandomForestClassifier(
        n_estimators=300, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=SEED
    )
    clf.fit(X_tr, y_np[tr_idx])
    probs = clf.predict_proba(X_va)[:, 1]
    preds = (probs >= 0.5).astype(int)
    rf_preds_oof[va_idx] = preds
    rf_probs_oof[va_idx] = probs
    rf_f1s.append(f1_score(y_np[va_idx], preds, average="macro", zero_division=0))
    rf_aucs.append(roc_auc_score(y_np[va_idx], probs))

rf_result = {
    "f1_mean":  float(np.mean(rf_f1s)),
    "f1_std":   float(np.std(rf_f1s)),
    "auc_mean": float(np.mean(rf_aucs)),
    "auc_std":  float(np.std(rf_aucs)),
    "cat_f1":   per_cat_f1(y_np, rf_preds_oof),
}
ts(f"  RF: F1={rf_result['f1_mean']:.4f} ± {rf_result['f1_std']:.4f}  "
   f"AUC={rf_result['auc_mean']:.4f}")
ts(f"  Cat F1 → bot={rf_result['cat_f1']['bot']:.4f}  "
   f"genuine={rf_result['cat_f1']['genuine']:.4f}")


# ── GNN model ─────────────────────────────────────────────────────────────────

class BotSAGE(nn.Module):
    """
    Same architecture as cresci pipeline with one addition:
    input projection (788 → PROJ_DIM) before the SAGE layers.
    Without this, 788-dim × 256 hidden × 10k nodes strains 6GB VRAM.
    """
    def __init__(self, in_dim, proj_dim=128, hidden=256, dropout=0.4):
        super().__init__()
        self.proj  = nn.Linear(in_dim, proj_dim)
        self.bn_p  = nn.BatchNorm1d(proj_dim)
        self.conv1 = SAGEConv(proj_dim,      hidden,      aggr="mean", normalize=True)
        self.conv2 = SAGEConv(hidden,        hidden // 2, aggr="mean", normalize=True)
        self.conv3 = SAGEConv(hidden // 2,   64,          aggr="mean", normalize=True)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden // 2)
        self.bn3   = nn.BatchNorm1d(64)
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.drop = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        x = self.drop(F.relu(self.bn_p(self.proj(x))))
        x = self.drop(F.relu(self.bn1(self.conv1(x, edge_index))))
        x = self.drop(F.relu(self.bn2(self.conv2(x, edge_index))))
        x = self.drop(F.relu(self.bn3(self.conv3(x, edge_index))))
        return x

    def forward(self, x, edge_index):
        return self.cls_head(self.encode(x, edge_index)).squeeze(-1)

    def link_logits(self, h, lp_ei):
        return (h[lp_ei[0]] * h[lp_ei[1]]).sum(dim=-1)


# ── GNN fold trainer ──────────────────────────────────────────────────────────

def train_fold(X_scaled, y, tr_idx, va_idx, full_ei, sparse_ei_fn,
               use_lp, lambda_lp, num_nodes, neg_sampler, rng):

    x_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    tr_mask = torch.zeros(num_nodes, dtype=torch.bool)
    va_mask = torch.zeros(num_nodes, dtype=torch.bool)
    tr_mask[tr_idx] = True
    va_mask[va_idx] = True
    tr_mask_d = tr_mask.to(DEVICE)
    va_mask_d = va_mask.to(DEVICE)

    n_neg_cls = int((y[tr_idx] == 0).sum())
    n_pos_cls = int((y[tr_idx] == 1).sum())
    pos_weight = torch.tensor([n_neg_cls / n_pos_cls], dtype=torch.float32).to(DEVICE)

    # Full graph with self-loops for validation
    full_sl, _ = add_self_loops(full_ei, num_nodes=num_nodes)
    full_sl_d  = full_sl.to(DEVICE)

    model     = BotSAGE(X_scaled.shape[1], PROJ_DIM, HIDDEN, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1, best_epoch, wait = 0.0, 0, 0
    best_state = None
    tr_set = set(tr_idx.tolist())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        kept_ei, dropped_ei = sparse_ei_fn(epoch)
        kept_sl, _ = add_self_loops(kept_ei, num_nodes=num_nodes)
        train_ei_d = kept_sl.to(DEVICE)

        h          = model.encode(x_t, train_ei_d)
        cls_logits = model.cls_head(h).squeeze(-1)
        l_cls = F.binary_cross_entropy_with_logits(
            cls_logits[tr_mask_d], y_t[tr_mask_d], pos_weight=pos_weight
        )
        loss = l_cls

        if use_lp and dropped_ei.shape[1] > 0:
            drop_np  = dropped_ei.numpy()
            pos_mask = np.array([
                int(drop_np[0, i]) in tr_set or int(drop_np[1, i]) in tr_set
                for i in range(min(drop_np.shape[1], 4000))  # cap for speed
            ])
            pos_ei = dropped_ei[:, :4000][:, pos_mask]
            if pos_ei.shape[1] > 0:
                neg_ei    = neg_sampler.sample(pos_ei.shape[1], rng)
                lp_ei     = torch.cat([pos_ei, neg_ei], dim=1).to(DEVICE)
                lp_labels = torch.cat([
                    torch.ones(pos_ei.shape[1]),
                    torch.zeros(neg_ei.shape[1])
                ]).to(DEVICE)
                l_lp = F.binary_cross_entropy_with_logits(
                    model.link_logits(h, lp_ei), lp_labels
                )
                loss = l_cls + lambda_lp * l_lp

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(x_t, full_sl_d)
                probs  = torch.sigmoid(logits[va_mask_d]).cpu().numpy()
                preds  = (probs >= 0.5).astype(int)
                val_f1 = f1_score(y[va_idx], preds, average="macro", zero_division=0)
            if val_f1 > best_f1:
                best_f1    = val_f1
                best_epoch = epoch
                wait       = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        logits = model(x_t, full_sl_d)
        probs  = torch.sigmoid(logits[va_mask_d]).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        val_f1 = f1_score(y[va_idx], preds, average="macro", zero_division=0)
        val_auc = roc_auc_score(y[va_idx], probs)

    return val_f1, val_auc, best_epoch, preds, probs


def run_gnn(X_np, y_np, full_ei, num_nodes, drop_rate, paradigm, use_lp):
    rng      = np.random.default_rng(SEED)
    neg_samp = NegativeSampler(full_ei, num_nodes=num_nodes, ratio=1)

    if drop_rate == 0.0:
        def sparse_ei_fn(ep): return full_ei.clone(), full_ei[:, :0]
    elif paradigm == "random":
        dropper = RandomDrop(p=drop_rate)
        def sparse_ei_fn(ep):
            k, d, _ = dropper(full_ei)
            return k, d
    else:
        dropper = DegreeBasedDrop(p_base=drop_rate)
        def sparse_ei_fn(ep):
            k, d, _ = dropper(full_ei, num_nodes=num_nodes)
            return k, d

    fold_f1s, fold_aucs = [], []
    oof_preds = np.zeros(num_nodes, dtype=int)
    oof_probs = np.zeros(num_nodes)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_np, y_np)):
        scaler   = RobustScaler()
        X_scaled = np.zeros_like(X_np)
        X_scaled[tr_idx] = scaler.fit_transform(X_np[tr_idx])
        X_scaled[va_idx] = scaler.transform(X_np[va_idx])

        f1, auc, ep, preds, probs = train_fold(
            X_scaled, y_np, tr_idx, va_idx,
            full_ei, sparse_ei_fn, use_lp, LAMBDA_LP,
            num_nodes, neg_samp, rng
        )
        fold_f1s.append(f1)
        fold_aucs.append(auc)
        oof_preds[va_idx] = preds
        oof_probs[va_idx] = probs

    return {
        "f1_mean":  float(np.mean(fold_f1s)),
        "f1_std":   float(np.std(fold_f1s)),
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std":  float(np.std(fold_aucs)),
        "cat_f1":   per_cat_f1(y_np, oof_preds),
    }


# ── Main experiment loop ──────────────────────────────────────────────────────
results = {"rf": rf_result}

for paradigm in PARADIGMS:
    for model_name, use_lp in [("sage_vanilla", False), ("tresa", True)]:
        key = f"{model_name}_{paradigm}"
        results[key] = {}
        f1_curve = []

        ts(f"\n{'='*58}")
        ts(f"Model: {model_name.upper()}  |  Paradigm: {paradigm}")
        ts(f"{'='*58}")

        for drop_rate in DROP_RATES:
            t0  = time.time()
            res = run_gnn(X_np, y_np, edge_index, N,
                          drop_rate, paradigm, use_lp)
            elapsed = time.time() - t0
            results[key][str(drop_rate)] = res
            f1_curve.append((drop_rate, res["f1_mean"]))
            ts(f"  drop={drop_rate:.0%}  F1={res['f1_mean']:.4f}±{res['f1_std']:.4f}"
               f"  AUC={res['auc_mean']:.4f}  bot={res['cat_f1']['bot']:.4f}"
               f"  ({elapsed:.0f}s)")

        results[key]["robustness_auc"] = robustness_auc(f1_curve)
        ts(f"  → Robustness AUC: {results[key]['robustness_auc']:.4f}")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("MGTAB ROBUSTNESS SUMMARY")
print(f"{'='*72}")
print(f"  {'Model':22s}  {'Paradigm':14s}  {'F1@0%':>7}  {'F1@20%':>7}  "
      f"{'F1@40%':>7}  {'F1@60%':>7}  {'RobAUC':>8}")
print("  " + "-" * 70)

rf_f1 = rf_result["f1_mean"]
print(f"  {'RF (node-only)':22s}  {'—':14s}  {rf_f1:.4f}  {rf_f1:.4f}  "
      f"{rf_f1:.4f}  {rf_f1:.4f}  {'—':>8}")

for paradigm in PARADIGMS:
    for model_name in ["sage_vanilla", "tresa"]:
        key   = f"{model_name}_{paradigm}"
        label = "TRESA (ours)" if model_name == "tresa" else "SAGE vanilla"
        rob   = results[key]["robustness_auc"]
        vals  = [results[key][str(dr)]["f1_mean"] for dr in DROP_RATES]
        print(f"  {label:22s}  {paradigm:14s}  " +
              "  ".join(f"{v:.4f}" for v in vals) + f"  {rob:.4f}")

# Crossover analysis
print(f"\n  RF F1: {rf_f1:.4f}")
print("  Crossover point (GNN drops below RF):")
for paradigm in PARADIGMS:
    for model_name in ["sage_vanilla", "tresa"]:
        key = f"{model_name}_{paradigm}"
        crossover = "never"
        for dr in DROP_RATES:
            if results[key][str(dr)]["f1_mean"] < rf_f1:
                crossover = f"{dr:.0%}"
                break
        label = "TRESA      " if model_name == "tresa" else "SAGE vanilla"
        print(f"    {label} [{paradigm}]: {crossover}")

# Cresci vs MGTAB comparison
print(f"\n{'='*72}")
print("CRESCI-2017 vs MGTAB — KEY COMPARISON")
print(f"{'='*72}")
print(f"  {'':30s}  {'cresci-2017':>14}  {'MGTAB':>10}")
print("  " + "-" * 58)
print(f"  {'Nodes':30s}  {'14,368':>14}  {'10,199':>10}")
print(f"  {'Edges':30s}  {'1,423':>14}  {'1,700,108':>10}")
print(f"  {'Density':30s}  {'0.001%':>14}  {'1.6%':>10}")
print(f"  {'Isolated nodes':30s}  {'96%':>14}  {'?':>10}")
print(f"  {'RF F1':30s}  {'0.9827':>14}  {rf_f1:>10.4f}")
print(f"  {'SAGE F1 @ 0% drop':30s}  {'0.9795':>14}  "
      f"{results['sage_vanilla_random']['0.0']['f1_mean']:>10.4f}")
print(f"  {'SAGE F1 @ 60% drop (random)':30s}  {'0.9780':>14}  "
      f"{results['sage_vanilla_random']['0.6']['f1_mean']:>10.4f}")
print(f"  {'SAGE degradation 0→60%':30s}  {'−0.0015':>14}  "
      f"{results['sage_vanilla_random']['0.6']['f1_mean'] - results['sage_vanilla_random']['0.0']['f1_mean']:>+10.4f}")
print(f"  {'TRESA vs SAGE (Rob.AUC delta)':30s}  "
      f"  {'−0.0126 (hurts)':>14}  "
      f"{results['tresa_random']['robustness_auc'] - results['sage_vanilla_random']['robustness_auc']:>+10.4f}")

# Save
out = os.path.join(OUT_DIR, "mgtab_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
ts(f"\nSaved → {out}")
ts("MGTAB pipeline complete.")
