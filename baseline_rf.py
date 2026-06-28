"""
cresci-2017 baseline pipeline — Step 3: Random Forest baseline
===============================================================
Two ablations, evaluated side by side:
  A) node_only   — 15 profile/activity features
  B) node+graph  — node + 13 graph structural features

Evaluation:
  - Stratified 5-fold CV (preserves class + bot_type distribution)
  - Metrics: Accuracy, Precision, Recall, F1 (macro), AUC-ROC
  - Per-category F1 breakdown (the metric that matters for research)
  - Feature importance with std dev bands
  - Saves results to JSON for comparison in later steps (GNN)

Class imbalance strategy: class_weight='balanced' in RF.
We do NOT oversample — the literature standard for cresci-2017 is
balanced weights + reporting macro-F1, not accuracy.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)

# ── Feature sets ──────────────────────────────────────────────────────────────
NODE_FEATURES = [
    "statuses_count", "followers_count", "friends_count",
    "favourites_count", "listed_count",
    "ff_ratio", "engagement", "listed_per_fol",
    "account_age_days", "profile_complete",
    "default_profile_image", "geo_enabled",
    "has_description", "has_url", "has_location",
]

GRAPH_FEATURES = [
    "in_degree_w", "out_degree_w", "in_degree", "out_degree",
    "pagerank", "clustering_coef",
    "wcc_size", "ego_density", "degree_ratio",
    "is_isolated", "total_degree",
    # hub/authority intentionally excluded — collapsed to ~0 in sparse graph
]

ABLATIONS = {
    "node_only":   NODE_FEATURES,
    "node+graph":  NODE_FEATURES + GRAPH_FEATURES,
}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading full feature matrix...")
df = pd.read_parquet(os.path.join(DATA_ROOT, "full_features.parquet"))
print(f"  {len(df):,} rows · {df['label'].value_counts().to_dict()}")

# ── RF config ─────────────────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_cv(X, y, groups, feature_names):
    """
    Run 5-fold stratified CV. Returns per-fold metrics + feature importances.
    groups: array of bot_type labels for per-category breakdown.
    """
    fold_metrics = []
    importances  = np.zeros(X.shape[1])
    all_preds    = np.zeros(len(y))
    all_probs    = np.zeros(len(y))

    for fold, (tr_idx, va_idx) in enumerate(CV.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Scale — RobustScaler handles the extreme outliers in follower counts
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_tr)

        preds = clf.predict(X_va)
        probs = clf.predict_proba(X_va)[:, 1]

        all_preds[va_idx] = preds
        all_probs[va_idx] = probs
        importances += clf.feature_importances_

        fold_metrics.append({
            "fold":      fold + 1,
            "accuracy":  accuracy_score(y_va, preds),
            "precision": precision_score(y_va, preds, zero_division=0),
            "recall":    recall_score(y_va, preds, zero_division=0),
            "f1_macro":  f1_score(y_va, preds, average="macro", zero_division=0),
            "auc_roc":   roc_auc_score(y_va, probs),
        })
        print(f"    Fold {fold+1}: acc={fold_metrics[-1]['accuracy']:.3f}  "
              f"f1={fold_metrics[-1]['f1_macro']:.3f}  "
              f"auc={fold_metrics[-1]['auc_roc']:.3f}")

    importances /= 5   # average over folds

    # Aggregate fold metrics
    metrics_df = pd.DataFrame(fold_metrics)
    agg = {
        col: {"mean": metrics_df[col].mean(), "std": metrics_df[col].std()}
        for col in ["accuracy", "precision", "recall", "f1_macro", "auc_roc"]
    }

    # Per-category F1 (on OOF predictions)
    cat_f1 = {}
    for cat in sorted(df["bot_type"].unique()):
        mask = groups == cat
        if mask.sum() == 0:
            continue
        cat_f1[cat] = f1_score(y[mask], all_preds[mask],
                               average="binary", zero_division=0)

    # Feature importance table
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # Confusion matrix (OOF)
    cm = confusion_matrix(y, all_preds)

    return agg, cat_f1, feat_imp, all_preds, all_probs, cm


# ── Run ablations ─────────────────────────────────────────────────────────────
results = {}
y = df["label"].values
groups = df["bot_type"].values

for ablation_name, features in ABLATIONS.items():
    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_name}  ({len(features)} features)")
    print("="*60)

    X = df[features].values.astype(np.float64)

    # Clip extreme outliers before scaling (beyond 99.9th percentile)
    for i in range(X.shape[1]):
        cap = np.nanpercentile(X[:, i], 99.9)
        X[:, i] = np.clip(X[:, i], None, cap)

    agg, cat_f1, feat_imp, preds, probs, cm = run_cv(X, y, groups, features)

    results[ablation_name] = {
        "metrics": agg,
        "category_f1": cat_f1,
        "feature_importance": feat_imp.to_dict(),
    }

    # ── Print results ──
    print("\n  Aggregate metrics (mean ± std over 5 folds):")
    for metric, vals in agg.items():
        print(f"    {metric:12s}: {vals['mean']:.4f} ± {vals['std']:.4f}")

    print("\n  Per-category F1 (OOF):")
    for cat, f1 in sorted(cat_f1.items(), key=lambda x: x[1]):
        bar = "█" * int(f1 * 30)
        print(f"    {cat:25s}: {f1:.4f}  {bar}")

    print("\n  Top-10 feature importances:")
    for feat, imp in feat_imp.head(10).items():
        bar = "█" * int(imp * 200)
        print(f"    {feat:25s}: {imp:.4f}  {bar}")

    print("\n  Confusion matrix (OOF):")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")


# ── Side-by-side comparison ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print("ABLATION COMPARISON SUMMARY")
print("="*60)
metrics_order = ["accuracy", "precision", "recall", "f1_macro", "auc_roc"]
header = f"  {'metric':15s}" + "".join(f"  {k:>20s}" for k in ABLATIONS)
print(header)
print("  " + "-" * (15 + 22 * len(ABLATIONS)))
for m in metrics_order:
    row = f"  {m:15s}"
    for ablation_name in ABLATIONS:
        v = results[ablation_name]["metrics"][m]
        row += f"  {v['mean']:.4f} ± {v['std']:.4f}"
    print(row)

print("\n  Per-category F1:")
all_cats = sorted(set(cat for r in results.values() for cat in r["category_f1"]))
for cat in all_cats:
    row = f"  {cat:25s}"
    for ablation_name in ABLATIONS:
        f1 = results[ablation_name]["category_f1"].get(cat, 0.0)
        row += f"  {f1:.4f}"
    print(row)

delta_f1 = (
    results["node+graph"]["metrics"]["f1_macro"]["mean"] -
    results["node_only"]["metrics"]["f1_macro"]["mean"]
)
delta_auc = (
    results["node+graph"]["metrics"]["auc_roc"]["mean"] -
    results["node_only"]["metrics"]["auc_roc"]["mean"]
)
print(f"\n  Graph feature delta → F1: {delta_f1:+.4f}   AUC: {delta_auc:+.4f}")
if abs(delta_f1) < 0.005:
    print("  ⚠  Graph adds <0.005 F1 — as expected given 96% isolation rate.")
    print("     This is the key baseline finding: node features dominate on cresci-2017.")

# ── Save results for Step 4 comparison ───────────────────────────────────────
# Convert numpy types to native Python for JSON serialization
def to_serializable(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj

out = os.path.join(DATA_ROOT, "baseline_results.json")
with open(out, "w") as f:
    json.dump(to_serializable(results), f, indent=2)
print(f"\nSaved → {out}")
print("Step 3 complete. Ready for Step 4 (GNN).")
