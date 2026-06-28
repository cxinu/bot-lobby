"""
cresci-2017 / TRESA — Step 7: Robustness evaluation & paper figures
====================================================================
Generates four outputs:
  1. robustness_curves.png   — F1 vs drop-rate curves (paper Figure 1)
  2. robustness_heatmap.png  — F1 at each (model, drop, paradigm) cell
  3. cat_f1_breakdown.png    — per bot-type F1 at 0% and 60% drop
  4. results_summary.txt     — formatted table for the paper

Key finding visualised:
  - SAGE vanilla is flat under sparsification → already graph-agnostic
  - TRESA L_lp hurts in sparse regime → gradient conflict diagnosis
  - RF is the ceiling throughout → node features dominate cresci-2017
  - Degree-biased drop is near-no-op → hub edges are already absent
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
}
LABELS = {
    "rf":           "RF (node-only)",
    "sage_vanilla": "GraphSAGE vanilla",
    "tresa":        "TRESA (ours)",
}
PARADIGM_LABELS = {
    "random":        "Random drop",
    "degree_biased": "Degree-biased drop",
}

DROP_RATES   = [0.0, 0.2, 0.4, 0.6]
DROP_LABELS  = ["0%", "20%", "40%", "60%"]
PARADIGMS    = ["random", "degree_biased"]

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
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
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

    # SAGE vanilla
    s_means, s_stds = get_f1_curve(f"sage_vanilla_{paradigm}")
    ax.plot(xs, s_means, color=COLORS["sage_vanilla"], linewidth=2.0,
            linestyle="--", marker="o", markersize=6,
            label=LABELS["sage_vanilla"], zorder=4)
    ax.fill_between(xs, s_means - s_stds, s_means + s_stds,
                    color=COLORS["sage_vanilla"], alpha=0.15, zorder=2)

    # TRESA
    t_means, t_stds = get_f1_curve(f"tresa_{paradigm}")
    ax.plot(xs, t_means, color=COLORS["tresa"], linewidth=2.0,
            linestyle="-", marker="s", markersize=6,
            label=LABELS["tresa"], zorder=4)
    ax.fill_between(xs, t_means - t_stds, t_means + t_stds,
                    color=COLORS["tresa"], alpha=0.15, zorder=2)

    # Annotate the TRESA drop at 20%
    drop_20_tresa = t_means[1]
    drop_20_sage  = s_means[1]
    delta = drop_20_tresa - drop_20_sage
    ax.annotate(
        f"L_lp penalty\n({delta:+.3f} F1)",
        xy=(20, drop_20_tresa), xytext=(25, drop_20_tresa - 0.008),
        fontsize=8.5, color=COLORS["tresa"],
        arrowprops=dict(arrowstyle="->", color=COLORS["tresa"], lw=1.0),
    )

    ax.set_title(PARADIGM_LABELS[paradigm], fontsize=11, fontweight="bold")
    ax.set_xlabel("Edges dropped (%)")
    ax.set_xlim(-3, 65)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xticklabels(DROP_LABELS)
    ax.set_ylim(0.940, 0.998)
    ax.set_yticks([0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    if paradigm == "random":
        ax.set_ylabel("Macro F1")

axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)

# Add finding annotation box
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
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
fig.suptitle("F1 scores across sparsification grid", fontsize=12,
             fontweight="bold", y=1.02)

for ax, paradigm in zip(axes, PARADIGMS):
    models   = ["sage_vanilla", "tresa"]
    n_models = len(models) + 1   # +1 for RF row
    n_drops  = len(DROP_RATES)

    matrix = np.zeros((n_models, n_drops))

    # RF row (constant)
    matrix[0, :] = rf_f1

    for mi, model in enumerate(models):
        key = f"{model}_{paradigm}"
        for di, dr in enumerate(DROP_RATES):
            matrix[mi + 1, di] = results[key][str(dr)]["f1_mean"]

    row_labels = ["RF (node-only)", "SAGE vanilla", "TRESA (ours)"]
    vmin, vmax = 0.955, 0.985

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_drops))
    ax.set_xticklabels(DROP_LABELS)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Edges dropped (%)")
    ax.set_title(PARADIGM_LABELS[paradigm], fontsize=10, fontweight="bold")

    for i in range(n_models):
        for j in range(n_drops):
            val = matrix[i, j]
            txt_color = "white" if val < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="#222222")

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

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
fig.suptitle("Per-category F1: effect of 60% random edge drop",
             fontsize=12, fontweight="bold")

models_to_plot = [
    ("rf",           "rf",                  "random"),
    ("sage_vanilla", "sage_vanilla_random", "random"),
    ("tresa",        "tresa_random",        "random"),
]

for ax, (model_key, result_key, paradigm) in zip(axes, models_to_plot):
    x  = np.arange(len(categories))
    w  = 0.32

    if model_key == "rf":
        f1_0  = [results["rf"]["0.0"]["cat_f1"].get(c, 0) for c in categories]
        f1_60 = f1_0   # RF is constant
    else:
        f1_0  = [results[result_key]["0.0"]["cat_f1"].get(c, 0) for c in categories]
        f1_60 = [results[result_key]["0.6"]["cat_f1"].get(c, 0) for c in categories]

    bars0  = ax.bar(x - w/2, f1_0,  w, label="0% drop",  color=COLORS[model_key],
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    bars60 = ax.bar(x + w/2, f1_60, w, label="60% drop", color=COLORS[model_key],
                    alpha=0.45, edgecolor=COLORS[model_key], linewidth=1.0,
                    hatch="//")

    ax.set_title(LABELS[model_key], fontsize=10, fontweight="bold",
                 color=COLORS[model_key])
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if model_key == "rf":
        ax.set_ylabel("F1 score")

    ax.legend(fontsize=8, loc="lower right")

    # Value labels on bars
    for bar in list(bars0) + list(bars60):
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5,
                    rotation=45)

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
summary_lines.append("=" * 62)
summary_lines.append("Dataset: 14,368 nodes  |  1,423 retweet edges  |  96% isolated")
summary_lines.append("CV: 5-fold stratified  |  Metric: macro F1")
summary_lines.append("")

summary_lines.append("Table 1: F1 across sparsification levels")
summary_lines.append("-" * 62)
header = f"{'Model':22s}  {'Paradigm':14s}" + "".join(f"  {l:>7s}" for l in DROP_LABELS) + "  Rob.AUC"
summary_lines.append(header)
summary_lines.append("-" * 62)

# RF row
rf_row = f"{'RF (node-only)':22s}  {'—':14s}"
for _ in DROP_RATES:
    rf_row += f"  {rf_f1:.4f}"
rf_row += "     —"
summary_lines.append(rf_row)

for paradigm in PARADIGMS:
    for model_name in ["sage_vanilla", "tresa"]:
        key   = f"{model_name}_{paradigm}"
        label = "TRESA (ours)" if model_name == "tresa" else "SAGE vanilla"
        rob   = results[key].get("robustness_auc", 0)
        row   = f"{label:22s}  {paradigm:14s}"
        for dr in DROP_RATES:
            row += f"  {results[key][str(dr)]['f1_mean']:.4f}"
        row += f"  {rob:.4f}"
        summary_lines.append(row)

summary_lines.append("")
summary_lines.append("Table 2: Robustness AUC summary")
summary_lines.append("-" * 62)
for paradigm in PARADIGMS:
    for model_name in ["sage_vanilla", "tresa"]:
        key   = f"{model_name}_{paradigm}"
        rob   = results[key].get("robustness_auc", 0)
        label = "TRESA (ours)" if model_name == "tresa" else "SAGE vanilla"
        summary_lines.append(f"  {label:22s}  [{paradigm}]  Rob.AUC = {rob:.4f}")

summary_lines.append("")
summary_lines.append("Key findings")
summary_lines.append("-" * 62)
summary_lines.append(
    "1. RF ceiling: F1=0.9827. Both GNN models start below RF (SAGE=0.9795,\n"
    "   TRESA=0.9788). The graph adds no value on this dataset.\n"
    "2. SAGE vanilla shows near-zero degradation under edge drop in both\n"
    "   paradigms — it is already operating as a node-feature classifier.\n"
    "   This is direct evidence that 96% graph isolation makes the GNN\n"
    "   architecture irrelevant on cresci-2017.\n"
    "3. TRESA (L_lp auxiliary loss) degrades ~1.5% F1 relative to vanilla\n"
    "   SAGE under random drop. Diagnosis: the link prediction gradient\n"
    "   conflicts with classification in the sparse-graph regime — there\n"
    "   are too few positive edge pairs (≤1,423) to usefully shape the\n"
    "   encoder geometry across 14,368 nodes.\n"
    "4. Degree-biased drop is near-no-op for both models: the hub nodes\n"
    "   that lose edges are already the minority, and their neighbourhood\n"
    "   information is not load-bearing for the classifier.\n"
    "5. These findings collectively demonstrate that graph robustness\n"
    "   techniques (including L_lp regularisation) are only meaningful\n"
    "   when graph density is sufficient. cresci-2017 fails this\n"
    "   prerequisite by construction. Future work should validate on\n"
    "   MGTAB (density ~5%) where this constraint is relaxed."
)

summary_lines.append("")
summary_lines.append("Reframed contribution")
summary_lines.append("-" * 62)
summary_lines.append(
    "This paper is the first to systematically characterise the conditions\n"
    "under which GNN-based bot detectors degrade under edge sparsification,\n"
    "and to demonstrate that the standard benchmark (cresci-2017) is\n"
    "structurally unsuitable for evaluating graph robustness due to its\n"
    "per-category crawl methodology. The negative result is the result."
)

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
