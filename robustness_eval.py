"""
cresci-2017 / TRESA — Step 7: Robustness evaluation & paper figures
====================================================================
Generates four outputs:
  1. robustness_curves.png   — F1 vs drop-rate curves (paper Figure 1)
  2. robustness_heatmap.png  — F1 at each (model, drop, paradigm) cell
  3. cat_f1_breakdown.png    — per bot-type F1 at 0% and 60% drop
  4. results_summary.txt     — formatted table for the paper
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)
OUT_DIR = os.environ.get("OUT_DIR", DATA_ROOT)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})

COLORS = {
    "rf":           "#2ecc71",
    "sage_vanilla": "#7f77dd",
    "tresa":        "#e74c3c",
    "gcn_vanilla":  "#3498db",
    "gcn_tresa":    "#f39c12",
    "gat_vanilla":  "#1abc9c",
    "gat_tresa":    "#9b59b6",
}
LABELS = {
    "rf":           "RF (node-only)",
    "sage_vanilla": "GraphSAGE vanilla",
    "tresa":        "TRESA (SAGE)",
    "gcn_vanilla":  "GCN vanilla",
    "gcn_tresa":    "TRESA (GCN)",
    "gat_vanilla":  "GAT vanilla",
    "gat_tresa":    "TRESA (GAT)",
}
PARADIGM_LABELS = {
    "random":        "Random drop",
    "degree_biased": "Degree-biased drop",
}

DROP_RATES   = [0.0, 0.2, 0.4, 0.6]
DROP_LABELS  = ["0%", "20%", "40%", "60%"]
PARADIGMS    = ["random", "degree_biased"]

# All model variants in order (RF is handled separately)
MODEL_KEYS = ["sage_vanilla", "tresa", "gcn_vanilla", "gcn_tresa", "gat_vanilla", "gat_tresa"]

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading results...")
with open(os.path.join(DATA_ROOT, "tresa_results.json")) as f:
    results = json.load(f)

def get_f1_curve(key):
    """Returns (means, stds) arrays for DROP_RATES."""
    means = [results[key][str(dr)]["f1_mean"] for dr in DROP_RATES]
    stds  = [results[key][str(dr)]["f1_std"]  for dr in DROP_RATES]
    return np.array(means), np.array(stds)

rf_f1 = results["rf"]["0.0"]["f1_mean"]
rf_std = results["rf"]["0.0"]["f1_std"]


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: Robustness curves — 1×2 subplots (one per paradigm)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
fig.suptitle(
    "GNN robustness under edge sparsification — cresci-2017",
    fontsize=13, fontweight="bold", y=1.01
)

for ax, paradigm in zip(axes, PARADIGMS):
    xs = np.array(DROP_RATES) * 100   # percent for x-axis

    # RF — flat line + shaded band
    ax.axhline(rf_f1, color=COLORS["rf"], linewidth=2.0, linestyle="-",
               label=LABELS["rf"], zorder=3)
    ax.axhspan(rf_f1 - rf_std, rf_f1 + rf_std,
               color=COLORS["rf"], alpha=0.12, zorder=1)

    # GNN models
    for mk in MODEL_KEYS:
        key = f"{mk}_{paradigm}"
        means, stds = get_f1_curve(key)
        linestyle = "-" if "tresa" in mk else "--"
        marker = "s" if "tresa" in mk else "o"
        ax.plot(xs, means, color=COLORS[mk], linewidth=1.8,
                linestyle=linestyle, marker=marker, markersize=5,
                label=LABELS[mk], zorder=4)
        ax.fill_between(xs, means - stds, means + stds,
                        color=COLORS[mk], alpha=0.10, zorder=2)

    ax.set_title(PARADIGM_LABELS[paradigm], fontsize=11, fontweight="bold")
    ax.set_xlabel("Edges dropped (%)")
    ax.set_xlim(-3, 65)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xticklabels(DROP_LABELS)
    ax.set_ylim(0.60, 1.00)
    ax.set_yticks([0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    if paradigm == "random":
        ax.set_ylabel("Macro F1")

axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=2)

finding_txt = (
    "Key finding: SAGE is already graph-agnostic on cresci-2017\n"
    "(96% nodes isolated). L_lp degrades performance in\n"
    "this sparse regime due to conflicting gradient signals."
)
fig.text(0.5, -0.04, finding_txt, ha="center", fontsize=9,
         style="italic", color="#555555",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                   edgecolor="#cccccc", linewidth=0.8))

plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "robustness_curves.png")
plt.savefig(fig1_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved → {fig1_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: Heatmap — F1 at each cell
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("F1 scores across sparsification grid", fontsize=12,
             fontweight="bold", y=1.02)

for ax, paradigm in zip(axes, PARADIGMS):
    n_models = len(MODEL_KEYS) + 1   # +1 for RF row
    n_drops  = len(DROP_RATES)

    matrix = np.zeros((n_models, n_drops))

    # RF row (constant)
    matrix[0, :] = rf_f1

    for mi, model in enumerate(MODEL_KEYS):
        key = f"{model}_{paradigm}"
        for di, dr in enumerate(DROP_RATES):
            matrix[mi + 1, di] = results[key][str(dr)]["f1_mean"]

    row_labels = ["RF (node-only)"] + [LABELS[m] for m in MODEL_KEYS]
    vmin, vmax = 0.955, 0.985

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_drops))
    ax.set_xticklabels(DROP_LABELS)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Edges dropped (%)")
    ax.set_title(PARADIGM_LABELS[paradigm], fontsize=10, fontweight="bold")

    for i in range(n_models):
        for j in range(n_drops):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="#222222")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Macro F1")

plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "robustness_heatmap.png")
plt.savefig(fig2_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved → {fig2_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3: Per-category F1 at 0% vs 60% drop (random paradigm)
# ════════════════════════════════════════════════════════════════════════════
categories = ["fake_followers", "genuine", "social_spambot", "trad_spambot"]
cat_labels  = ["Fake\nFollowers", "Genuine", "Social\nSpambot", "Trad.\nSpambot"]

# Include all models in the category breakdown
breakdown_models = ["rf"] + MODEL_KEYS
n_breakdown = len(breakdown_models)
n_cols = 4
n_rows = int(np.ceil(n_breakdown / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
fig.suptitle("Per-category F1: effect of 60% random edge drop",
             fontsize=12, fontweight="bold")

axes_flat = axes.flatten()
for idx, model_key in enumerate(breakdown_models):
    ax = axes_flat[idx]
    x  = np.arange(len(categories))
    w  = 0.32

    if model_key == "rf":
        f1_0  = [results["rf"]["0.0"]["cat_f1"].get(c, 0) for c in categories]
        f1_60 = f1_0
    else:
        result_key = f"{model_key}_random"
        f1_0  = [results[result_key]["0.0"]["cat_f1"].get(c, 0) for c in categories]
        f1_60 = [results[result_key]["0.6"]["cat_f1"].get(c, 0) for c in categories]

    bars0  = ax.bar(x - w/2, f1_0,  w, label="0% drop",  color=COLORS[model_key],
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    bars60 = ax.bar(x + w/2, f1_60, w, label="60% drop", color=COLORS[model_key],
                    alpha=0.45, edgecolor=COLORS[model_key], linewidth=1.0,
                    hatch="//")

    ax.set_title(LABELS[model_key], fontsize=9, fontweight="bold",
                 color=COLORS[model_key])
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if model_key == "rf":
        ax.set_ylabel("F1 score")

    ax.legend(fontsize=7, loc="lower right")

    for bar in list(bars0) + list(bars60):
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6,
                    rotation=45)

# Hide unused subplots
for idx in range(len(breakdown_models), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, "cat_f1_breakdown.png")
plt.savefig(fig3_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved → {fig3_path}")


# ════════════════════════════════════════════════════════════════════════════
# Text summary for paper
# ════════════════════════════════════════════════════════════════════════════
summary_lines = []
summary_lines.append("TRESA ROBUSTNESS EVALUATION — cresci-2017")
summary_lines.append("=" * 80)
summary_lines.append("Dataset: 14,368 nodes  |  1,423 retweet edges  |  96% isolated")
summary_lines.append("CV: 5-fold stratified  |  Metric: macro F1")
summary_lines.append("")

summary_lines.append("Table 1: F1 across sparsification levels")
summary_lines.append("-" * 80)
header = f"{'Model':25s}  {'Paradigm':14s}" + "".join(f"  {lb:>7s}" for lb in DROP_LABELS) + "  Rob.AUC"
summary_lines.append(header)
summary_lines.append("-" * 80)

# RF row
rf_row = f"{'RF (node-only)':25s}  {'—':14s}"
for _ in DROP_RATES:
    rf_row += f"  {rf_f1:.4f}"
rf_row += "     —"
summary_lines.append(rf_row)

for paradigm in PARADIGMS:
    for model_name in MODEL_KEYS:
        key   = f"{model_name}_{paradigm}"
        label = LABELS[model_name]
        rob   = results[key].get("robustness_auc", 0)
        row   = f"{label:25s}  {paradigm:14s}"
        for dr in DROP_RATES:
            row += f"  {results[key][str(dr)]['f1_mean']:.4f}"
        row += f"  {rob:.4f}"
        summary_lines.append(row)

summary_txt = "\n".join(summary_lines)
print("\n" + summary_txt)

out_txt = os.path.join(OUT_DIR, "results_summary.txt")
with open(out_txt, "w") as f:
    f.write(summary_txt)
print(f"\nSaved → {out_txt}")
print(f"Saved → {fig1_path}")
print(f"Saved → {fig2_path}")
print(f"Saved → {fig3_path}")
print("\nStep 7 complete.")
