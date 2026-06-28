import json
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)

def load_results():
    with open(os.path.join(DATA_ROOT, "tresa_results.json"), "r") as f:
        return json.load(f)

def plot_robustness_curves(results):
    drop_rates = [0.0, 0.2, 0.4, 0.6]
    drop_labels = ["0%", "20%", "40%", "60%"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Colors and styles
    styles = {
        "rf": {"color": "#e74c3c", "linestyle": "--", "marker": "o", "label": "RF (node-only)"},
        "sage": {"color": "#3498db", "linestyle": "-", "marker": "s", "label": "SAGE vanilla"},
        "tresa": {"color": "#2ecc71", "linestyle": "-", "marker": "^", "label": "TRESA (ours)"}
    }
    
    for ax, paradigm in zip(axes, ["random", "degree_biased"]):
        # RF is flat
        rf_f1s = [results["rf"][str(dr)]["f1_mean"] for dr in drop_rates]
        rf_stds = [results["rf"][str(dr)]["f1_std"] for dr in drop_rates]
        ax.errorbar(drop_rates, rf_f1s, yerr=rf_stds, **styles["rf"], capsize=4)
        
        # SAGE vanilla
        sage_key = f"sage_vanilla_{paradigm}"
        sage_f1s = [results[sage_key][str(dr)]["f1_mean"] for dr in drop_rates]
        sage_stds = [results[sage_key][str(dr)]["f1_std"] for dr in drop_rates]
        ax.errorbar(drop_rates, sage_f1s, yerr=sage_stds, **styles["sage"], capsize=4)
        
        # TRESA
        tresa_key = f"tresa_{paradigm}"
        tresa_f1s = [results[tresa_key][str(dr)]["f1_mean"] for dr in drop_rates]
        tresa_stds = [results[tresa_key][str(dr)]["f1_std"] for dr in drop_rates]
        ax.errorbar(drop_rates, tresa_f1s, yerr=tresa_stds, **styles["tresa"], capsize=4)
        
        ax.set_title(f"Sparsification: {paradigm.replace('_', ' ').capitalize()}", fontsize=14, pad=10)
        ax.set_xlabel("Edge Drop Rate", fontsize=12)
        ax.set_xticks(drop_rates)
        ax.set_xticklabels(drop_labels)
        ax.set_ylim(0.95, 0.99)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=11, loc="lower left")
        
    axes[0].set_ylabel("Macro F1 Score", fontsize=12)
    plt.suptitle("Topological Robustness: Macro F1 vs. Edge Drop Rate", fontsize=16, weight="bold", y=0.98)
    plt.tight_layout()
    
    out_path = os.path.join(DATA_ROOT, "robustness_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def plot_robustness_heatmap(results):
    drop_rates = [0.0, 0.2, 0.4, 0.6]
    drop_labels = ["0%", "20%", "40%", "60%"]
    models = ["RF (node-only)", "SAGE vanilla", "TRESA (ours)"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, paradigm in enumerate(["random", "degree_biased"]):
        # Build grid
        grid = np.zeros((3, 4))
        # Row 0: RF
        grid[0, :] = [results["rf"][str(dr)]["f1_mean"] for dr in drop_rates]
        # Row 1: SAGE
        grid[1, :] = [results[f"sage_vanilla_{paradigm}"][str(dr)]["f1_mean"] for dr in drop_rates]
        # Row 2: TRESA
        grid[2, :] = [results[f"tresa_{paradigm}"][str(dr)]["f1_mean"] for dr in drop_rates]
        
        ax = axes[i]
        im = ax.imshow(grid, cmap="YlGnBu", vmin=0.95, vmax=0.99)
        
        # Show values
        for r in range(3):
            for c in range(4):
                val = grid[r, c]
                color = "white" if val < 0.965 or val > 0.98 else "black"
                ax.text(c, r, f"{val:.4f}", ha="center", va="center", color=color, fontweight="bold")
                
        ax.set_title(f"{paradigm.replace('_', ' ').capitalize()} Drop Heatmap", fontsize=13, pad=10)
        ax.set_xticks(range(4))
        ax.set_xticklabels(drop_labels)
        ax.set_yticks(range(3))
        ax.set_yticklabels(models)
        
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Macro F1 Score")
    
    out_path = os.path.join(DATA_ROOT, "robustness_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_cat_f1_breakdown(results):
    # Categories of interest
    categories = ["fake_followers", "social_spambot", "trad_spambot"]
    cat_labels = ["Fake Followers", "Social Spambots", "Traditional Spambots"]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    x = np.arange(2) # 0% vs 60%
    width = 0.2
    
    for idx, (cat, label) in enumerate(zip(categories, cat_labels)):
        ax = axes[idx]
        
        # RF is constant
        rf_val = results["rf"]["0.0"]["cat_f1"][cat]
        
        # SAGE random
        sage_r_0 = results["sage_vanilla_random"]["0.0"]["cat_f1"][cat]
        sage_r_6 = results["sage_vanilla_random"]["0.6"]["cat_f1"][cat]
        
        # TRESA random
        tresa_r_0 = results["tresa_random"]["0.0"]["cat_f1"][cat]
        tresa_r_6 = results["tresa_random"]["0.6"]["cat_f1"][cat]

        # SAGE degree
        sage_d_0 = results["sage_vanilla_degree_biased"]["0.0"]["cat_f1"][cat]
        sage_d_6 = results["sage_vanilla_degree_biased"]["0.6"]["cat_f1"][cat]
        
        # TRESA degree
        tresa_d_0 = results["tresa_degree_biased"]["0.0"]["cat_f1"][cat]
        tresa_d_6 = results["tresa_degree_biased"]["0.6"]["cat_f1"][cat]
        
        rects1 = ax.bar(x - 2*width, [sage_r_0, sage_r_6], width, label="SAGE (random)", color="#3498db")
        rects2 = ax.bar(x - width, [tresa_r_0, tresa_r_6], width, label="TRESA (random)", color="#2ecc71")
        rects3 = ax.bar(x, [sage_d_0, sage_d_6], width, label="SAGE (degree-biased)", color="#2980b9")
        rects4 = ax.bar(x + width, [tresa_d_0, tresa_d_6], width, label="TRESA (degree-biased)", color="#27ae60")
        
        # Plot RF line
        ax.axhline(rf_val, color="#e74c3c", linestyle="--", label="RF (node-only)", alpha=0.8)
        
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["0% Drop", "60% Drop"], fontsize=11)
        ax.set_ylim(0.95, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        
    axes[0].set_ylabel("F1 Score (OOF)", fontsize=12)
    axes[2].legend(fontsize=10, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    
    plt.suptitle("Per-Category F1 Score Breakdown: 0% vs 60% Sparsification", fontsize=15, weight="bold", y=0.98)
    plt.tight_layout()
    
    out_path = os.path.join(DATA_ROOT, "cat_f1_breakdown.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def generate_results_summary(results):
    summary = []
    summary.append("=" * 80)
    summary.append("TRESA ROBUSTNESS BENCHMARK SUMMARY")
    summary.append("=" * 80)
    
    rf_base = results["rf"]["0.0"]["f1_mean"]
    summary.append("\nRandom Forest Baseline (Graph-Agnostic):")
    summary.append(f"  F1 Macro:  {rf_base:.4f} \u00b1 {results['rf']['0.0']['f1_std']:.4f}")
    summary.append(f"  AUC-ROC:   {results['rf']['0.0']['auc_mean']:.4f} \u00b1 {results['rf']['0.0']['auc_std']:.4f}")
    summary.append(f"  Categories F1: {results['rf']['0.0']['cat_f1']}")
    
    for paradigm in ["random", "degree_biased"]:
        summary.append("\n" + "-" * 50)
        summary.append(f"Paradigm: {paradigm.upper()}")
        summary.append("-" * 50)
        
        for model in ["sage_vanilla", "tresa"]:
            key = f"{model}_{paradigm}"
            summary.append(f"\nModel: {model.upper()}")
            summary.append(f"  Robustness AUC: {results[key]['robustness_auc']:.4f}")
            for dr in ["0.0", "0.2", "0.4", "0.6"]:
                m_res = results[key][dr]
                summary.append(f"    Drop={float(dr):.0%}: F1={m_res['f1_mean']:.4f} \u00b1 {m_res['f1_std']:.4f} | AUC={m_res['auc_mean']:.4f}")
                summary.append(f"              Categories: {m_res['cat_f1']}")
                
    out_path = os.path.join(DATA_ROOT, "results_summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"Saved: {out_path}")

def main():
    ts("Loading results...")
    results = load_results()
    
    ts("Generating Plot 1: Robustness Curves...")
    plot_robustness_curves(results)
    
    ts("Generating Plot 2: Heatmap...")
    plot_robustness_heatmap(results)
    
    ts("Generating Plot 3: Category F1 Breakdown...")
    plot_cat_f1_breakdown(results)
    
    ts("Generating Text Summary: results_summary.txt...")
    generate_results_summary(results)
    
    ts("Plotting and evaluation complete.")

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

import time
if __name__ == "__main__":
    main()
