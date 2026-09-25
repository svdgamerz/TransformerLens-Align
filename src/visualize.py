"""
Visualization module for high-DPI publication figures (arXiv standard).
Generates:
1. Figure 1: Layer-wise MLP Output Patching Recovery Curve (300 DPI).
2. Figure 2: Attention Head Attribution / Suppression Heatmap (300 DPI).
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional

def set_publication_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
    })

def plot_mlp_patching_recovery(
    mlp_results: Dict[str, any],
    output_path: str = "outputs/mlp_layer_patching_recovery.png",
    model_name: str = "Model"
) -> str:
    set_publication_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_layers = mlp_results["n_layers"]
    layers = np.arange(n_layers)
    means = np.array(mlp_results["mean_layer_patches"])
    stds = np.array(mlp_results["std_layer_patches"])
    clean_baseline = mlp_results["mean_clean_diff"]
    syc_baseline = mlp_results["mean_syc_diff"]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Baselines
    ax.axhline(clean_baseline, color="#2a9d8f", linestyle="--", linewidth=1.8, label=f"Clean Baseline ΔL ({clean_baseline:.2f})")
    ax.axhline(syc_baseline, color="#e76f51", linestyle="--", linewidth=1.8, label=f"Sycophantic Baseline ΔL ({syc_baseline:.2f})")
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)

    # Patched Curve
    ax.plot(layers, means, marker="o", color="#264653", linewidth=2.2, markersize=5.5, label="MLP Output Patched ΔL")
    ax.fill_between(layers, means - stds, means + stds, color="#264653", alpha=0.15, label="±1 Std Dev")

    # Annotate peak recovery
    peak_idx = int(np.argmax(means))
    peak_val = means[peak_idx]
    ax.annotate(
        f"Peak MLP Recovery: L{peak_idx} ({peak_val:.2f})",
        xy=(peak_idx, peak_val),
        xytext=(peak_idx + 0.8, peak_val + (0.5 if peak_val < clean_baseline else -0.5)),
        arrowprops=dict(facecolor="#264653", shrink=0.08, width=1.2, headwidth=6),
        fontsize=9.5,
        fontweight="bold"
    )

    ax.set_title(f"Layer-wise MLP Output Patching Recovery: {model_name}", pad=14, fontweight="bold")
    ax.set_xlabel("Layer Index (Residual Stream Flow →)", labelpad=8)
    ax.set_ylabel("Logit Difference ΔL = Logit(Truth) - Logit(Lie)", labelpad=8)
    ax.set_xticks(layers)
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Visuals] Saved Figure 1: {output_path}")
    return output_path

def plot_attention_head_heatmap(
    attribution_matrix: np.ndarray,
    output_path: str = "outputs/attention_suppression_heatmap.png",
    model_name: str = "Model",
    top_k: int = 5
) -> str:
    set_publication_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_layers, n_heads = attribution_matrix.shape
    fig, ax = plt.subplots(figsize=(10, 6))

    # Diverging heatmap centered at zero
    vmax = max(abs(float(np.min(attribution_matrix))), abs(float(np.max(attribution_matrix))), 0.1)
    vmin = -vmax

    sns.heatmap(
        attribution_matrix,
        cmap="vlag",
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Mean ΔL Shift (Recovery toward Truth)"},
        ax=ax,
        linewidths=0.4,
        linecolor="#f0f0f0"
    )

    # Highlight top suppressor heads (largest positive shift toward truth when clean activations restored)
    flat_indices = np.argsort(attribution_matrix.flatten())[::-1][:top_k]
    for idx in flat_indices:
        l = idx // n_heads
        h = idx % n_heads
        val = attribution_matrix[l, h]
        rect = plt.Rectangle((h, l), 1, 1, fill=False, edgecolor="#e63946", lw=2)
        ax.add_patch(rect)
        ax.text(h + 0.5, l + 0.5, f"+{val:.1f}", color="#e63946", ha="center", va="center", fontsize=7.5, fontweight="bold")

    ax.set_title(f"Attention Head Attribution Heatmap (Sycophancy Suppressor Heads): {model_name}", pad=14, fontweight="bold")
    ax.set_xlabel("Head Index", labelpad=8)
    ax.set_ylabel("Layer Index", labelpad=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Visuals] Saved Figure 2: {output_path}")
    return output_path
