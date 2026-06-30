import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(BASE_DIR, "data"))
RESULTS_ROOT = os.environ.get("OUT_DIR", os.path.join(BASE_DIR, "results"))

os.makedirs(RESULTS_ROOT, exist_ok=True)

# Style parameters matching paper/presentation aesthetics
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

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_json(filename, directories):
    for d in directories:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"Could not find {filename} in {directories}")

# ── Cresci-2017 Plotting Functions ────────────────────────────────────────────

def plot_cresci_robustness_curves(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(
        "GNN robustness under edge sparsification — cresci-2017",
        fontsize=13, fontweight="bold", y=1.01
    )
    
    rf_f1 = results["rf"]["0.0"]["f1_mean"]
    rf_std = results["rf"]["0.0"]["f1_std"]
    xs = np.array(DROP_RATES) * 100

    for ax, paradigm in zip(axes, PARADIGMS):
        # RF — flat line + shaded band
        ax.axhline(rf_f1, color=COLORS["rf"], linewidth=2.0, linestyle="-",
                   label=LABELS["rf"], zorder=3)
        ax.axhspan(rf_f1 - rf_std, rf_f1 + rf_std,
                   color=COLORS["rf"], alpha=0.12, zorder=1)

        # SAGE vanilla
        s_key = f"sage_vanilla_{paradigm}"
        s_means = np.array([results[s_key][str(dr)]["f1_mean"] for dr in DROP_RATES])
        s_stds = np.array([results[s_key][str(dr)]["f1_std"] for dr in DROP_RATES])
        ax.plot(xs, s_means, color=COLORS["sage_vanilla"], linewidth=2.0,
                linestyle="--", marker="o", markersize=6,
                label=LABELS["sage_vanilla"], zorder=4)
        ax.fill_between(xs, s_means - s_stds, s_means + s_stds,
                        color=COLORS["sage_vanilla"], alpha=0.15, zorder=2)

        # TRESA
        t_key = f"tresa_{paradigm}"
        t_means = np.array([results[t_key][str(dr)]["f1_mean"] for dr in DROP_RATES])
        t_stds = np.array([results[t_key][str(dr)]["f1_std"] for dr in DROP_RATES])
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
    out_path = os.path.join(out_dir, "robustness_curves.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def plot_cresci_robustness_heatmap(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.suptitle("F1 scores across sparsification grid — cresci-2017", fontsize=12,
                 fontweight="bold", y=1.02)
    
    rf_f1 = results["rf"]["0.0"]["f1_mean"]

    for ax, paradigm in zip(axes, PARADIGMS):
        models   = ["sage_vanilla", "tresa"]
        n_models = len(models) + 1
        n_drops  = len(DROP_RATES)

        matrix = np.zeros((n_models, n_drops))
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
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="#222222")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Macro F1")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "robustness_heatmap.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def plot_cresci_cat_f1_breakdown(results, out_dir):
    categories = ["fake_followers", "genuine", "social_spambot", "trad_spambot"]
    cat_labels  = ["Fake\nFollowers", "Genuine", "Social\nSpambot", "Trad.\nSpambot"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Per-category F1: effect of 60% random edge drop — cresci-2017",
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
            f1_60 = f1_0
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

        for bar in list(bars0) + list(bars60):
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6.5,
                        rotation=45)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "cat_f1_breakdown.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# ── MGTAB Plotting Functions ──────────────────────────────────────────────────

def plot_mgtab_robustness_curves(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(
        "GNN robustness under edge sparsification — MGTAB (1.6% density)",
        fontsize=13, fontweight="bold", y=1.01
    )
    
    rf_f1 = results["rf"]["f1_mean"]
    rf_std = results["rf"]["f1_std"]
    xs = np.array(DROP_RATES) * 100

    for ax, paradigm in zip(axes, PARADIGMS):
        # RF — flat line + shaded band
        ax.axhline(rf_f1, color=COLORS["rf"], linewidth=2.0, linestyle="-",
                   label=LABELS["rf"], zorder=3)
        ax.axhspan(rf_f1 - rf_std, rf_f1 + rf_std,
                   color=COLORS["rf"], alpha=0.12, zorder=1)

        # SAGE vanilla
        s_key = f"sage_vanilla_{paradigm}"
        s_means = np.array([results[s_key][str(dr)]["f1_mean"] for dr in DROP_RATES])
        s_stds = np.array([results[s_key][str(dr)]["f1_std"] for dr in DROP_RATES])
        ax.plot(xs, s_means, color=COLORS["sage_vanilla"], linewidth=2.0,
                linestyle="--", marker="o", markersize=6,
                label=LABELS["sage_vanilla"], zorder=4)
        ax.fill_between(xs, s_means - s_stds, s_means + s_stds,
                        color=COLORS["sage_vanilla"], alpha=0.15, zorder=2)

        # TRESA
        t_key = f"tresa_{paradigm}"
        t_means = np.array([results[t_key][str(dr)]["f1_mean"] for dr in DROP_RATES])
        t_stds = np.array([results[t_key][str(dr)]["f1_std"] for dr in DROP_RATES])
        ax.plot(xs, t_means, color=COLORS["tresa"], linewidth=2.0,
                linestyle="-", marker="s", markersize=6,
                label=LABELS["tresa"], zorder=4)
        ax.fill_between(xs, t_means - t_stds, t_means + t_stds,
                        color=COLORS["tresa"], alpha=0.15, zorder=2)

        ax.set_title(PARADIGM_LABELS[paradigm], fontsize=11, fontweight="bold")
        ax.set_xlabel("Edges dropped (%)")
        ax.set_xlim(-3, 65)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_xticklabels(DROP_LABELS)
        ax.set_ylim(0.870, 0.898)
        ax.set_yticks([0.870, 0.875, 0.880, 0.885, 0.890, 0.895])
        if paradigm == "random":
            ax.set_ylabel("Macro F1")

    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)

    finding_txt = (
        "Key finding: GNN performance remains flat under edge dropping on MGTAB.\n"
        "Due to high edge density (1.6%), the GNN does not degrade when edges are dropped.\n"
        "TRESA remains neutral and does not improve robustness."
    )
    fig.text(0.5, -0.04, finding_txt, ha="center", fontsize=9,
             style="italic", color="#555555",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                       edgecolor="#cccccc", linewidth=0.8))

    plt.tight_layout()
    out_path = os.path.join(out_dir, "mgtab_robustness_curves.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def plot_mgtab_robustness_heatmap(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.suptitle("F1 scores across sparsification grid — MGTAB", fontsize=12,
                 fontweight="bold", y=1.02)
    
    rf_f1 = results["rf"]["f1_mean"]

    for ax, paradigm in zip(axes, PARADIGMS):
        models   = ["sage_vanilla", "tresa"]
        n_models = len(models) + 1
        n_drops  = len(DROP_RATES)

        matrix = np.zeros((n_models, n_drops))
        matrix[0, :] = rf_f1

        for mi, model in enumerate(models):
            key = f"{model}_{paradigm}"
            for di, dr in enumerate(DROP_RATES):
                matrix[mi + 1, di] = results[key][str(dr)]["f1_mean"]

        row_labels = ["RF (node-only)", "SAGE vanilla", "TRESA (ours)"]
        vmin, vmax = 0.875, 0.895

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                       vmin=vmin, vmax=vmax)

        ax.set_xticks(range(n_drops))
        ax.set_xticklabels(drop_labels := DROP_LABELS)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel("Edges dropped (%)")
        ax.set_title(PARADIGM_LABELS[paradigm], fontsize=10, fontweight="bold")

        for i in range(n_models):
            for j in range(n_drops):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="#222222")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Macro F1")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "mgtab_robustness_heatmap.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def plot_mgtab_cat_f1_breakdown(results, out_dir):
    categories = ["bot", "genuine"]
    cat_labels  = ["Bot", "Genuine"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Per-category F1: effect of 60% random edge drop — MGTAB",
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
            f1_0  = [results["rf"]["cat_f1"].get(c, 0) for c in categories]
            f1_60 = f1_0
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

        for bar in list(bars0) + list(bars60):
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6.5,
                        rotation=45)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "mgtab_cat_f1_breakdown.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def generate_mgtab_results_summary(results, out_dir):
    summary_lines = []
    summary_lines.append("TRESA ROBUSTNESS EVALUATION — MGTAB")
    summary_lines.append("=" * 62)
    summary_lines.append("Dataset: 10,199 nodes  |  1,700,108 edges  |  0.5% isolated")
    summary_lines.append("CV: 5-fold stratified  |  Metric: macro F1")
    summary_lines.append("")

    summary_lines.append("Table 1: F1 across sparsification levels")
    summary_lines.append("-" * 62)
    header = f"{'Model':22s}  {'Paradigm':14s}" + "".join(f"  {l:>7s}" for l in DROP_LABELS) + "  Rob.AUC"
    summary_lines.append(header)
    summary_lines.append("-" * 62)

    rf_f1 = results["rf"]["f1_mean"]
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

    summary_txt = "\n".join(summary_lines)
    out_txt = os.path.join(out_dir, "mgtab_results_summary.txt")
    with open(out_txt, "w") as f:
        f.write(summary_txt)
    print(f"Saved: {out_txt}")

# ── Main ──────────────────────────────────────────────────────────────────────

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    # 1. Load results
    ts("Loading cresci-2017 results...")
    try:
        cresci_results = load_json("tresa_results.json", [RESULTS_ROOT, DATA_ROOT])
    except FileNotFoundError:
        print("  Warning: tresa_results.json not found, skipping cresci plots")
        cresci_results = None

    ts("Loading MGTAB results...")
    try:
        mgtab_results = load_json("mgtab_results.json", [RESULTS_ROOT, DATA_ROOT])
    except FileNotFoundError:
        print("  Warning: mgtab_results.json not found, skipping MGTAB plots")
        mgtab_results = None

    # 2. Plot Cresci if available
    if cresci_results:
        ts("Generating cresci-2017 Plot 1: Robustness Curves...")
        plot_cresci_robustness_curves(cresci_results, RESULTS_ROOT)
        
        ts("Generating cresci-2017 Plot 2: Heatmap...")
        plot_cresci_robustness_heatmap(cresci_results, RESULTS_ROOT)
        
        ts("Generating cresci-2017 Plot 3: Category F1 Breakdown...")
        plot_cresci_cat_f1_breakdown(cresci_results, RESULTS_ROOT)

    # 3. Plot MGTAB if available
    if mgtab_results:
        ts("Generating MGTAB Plot 1: Robustness Curves...")
        plot_mgtab_robustness_curves(mgtab_results, RESULTS_ROOT)
        
        ts("Generating MGTAB Plot 2: Heatmap...")
        plot_mgtab_robustness_heatmap(mgtab_results, RESULTS_ROOT)
        
        ts("Generating MGTAB Plot 3: Category F1 Breakdown...")
        plot_mgtab_cat_f1_breakdown(mgtab_results, RESULTS_ROOT)
        
        ts("Generating MGTAB Text Summary...")
        generate_mgtab_results_summary(mgtab_results, RESULTS_ROOT)

    ts("Plotting and evaluation complete.")

if __name__ == "__main__":
    main()
