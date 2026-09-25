"""
Circuit dissection module for isolating the internal mechanics of
Sycophancy Suppressor Heads (L15H7, L13H6, L13H13).
Performs:
1. Attention Pattern Extraction: What tokens do suppressor heads attend to?
2. Direct Logit Attribution (DLA): Does the head promote the Lie or suppress the Truth?
3. Generates Figures 3 & 4 (300 DPI).
"""

import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Prioritize site-packages for transformer_lens (v3.9.0)
sys.path = [p for p in sys.path if p not in ("", "C:\\TransformerLens", "c:\\TransformerLens", "c:/TransformerLens", "C:/TransformerLens")]
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
from transformer_lens import HookedTransformer
from src.dataset import get_validated_dataset, SycophancyPair
from src.patching_engine import load_target_model, clean_vram

TOP_HEADS = [(15, 7), (13, 6), (13, 13)]

def set_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
    })

def analyze_attention_patterns(
    model: HookedTransformer,
    dataset: List[Tuple[SycophancyPair, int, int]],
    output_path: str = "outputs/suppressor_attention_patterns.png"
) -> Dict[str, any]:
    set_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    example_pair, _, _ = dataset[0]
    prompt = example_pair.sycophantic_prompt
    tokens = model.to_str_tokens(prompt)

    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)

    fig, axes = plt.subplots(len(TOP_HEADS), 1, figsize=(11, 3.2 * len(TOP_HEADS)), sharex=True)
    if len(TOP_HEADS) == 1:
        axes = [axes]

    colors = ["#e76f51", "#f4a261", "#2a9d8f"]

    for idx, (layer, head) in enumerate(TOP_HEADS):
        pattern_hook = f"blocks.{layer}.attn.hook_pattern"
        attn_pattern = cache[pattern_hook][0, head, -1, :].cpu().numpy()

        ax = axes[idx]
        x_indices = np.arange(len(tokens))
        bars = ax.bar(x_indices, attn_pattern, color=colors[idx % len(colors)], alpha=0.85, edgecolor="#222222", linewidth=0.8)

        max_idx = int(np.argmax(attn_pattern))
        bars[max_idx].set_color("#d90429")
        bars[max_idx].set_edgecolor("#000000")
        bars[max_idx].set_linewidth(1.5)

        ax.set_title(f"Suppressor Head L{layer}H{head} Attention Distribution at Final Token (Top: '{tokens[max_idx]}' = {attn_pattern[max_idx]:.1%})", fontweight="bold", pad=8)
        ax.set_ylabel("Attention Weight", labelpad=6)
        ax.set_ylim(0, max(0.5, float(np.max(attn_pattern)) * 1.25))

    axes[-1].set_xticks(range(len(tokens)))
    axes[-1].set_xticklabels([repr(t)[1:-1] for t in tokens], rotation=45, ha="right", fontsize=9.5)
    axes[-1].set_xlabel("Prompt Tokens (Sycophantic Input Sequence)", labelpad=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Dissection] Saved Figure 3: {output_path}", flush=True)

    del cache
    clean_vram()
    return {"prompt": prompt, "tokens": tokens}

def analyze_direct_logit_attribution(
    model: HookedTransformer,
    dataset: List[Tuple[SycophancyPair, int, int]],
    output_path: str = "outputs/direct_logit_attribution_decomposition.png"
) -> Dict[str, Dict[str, float]]:
    set_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    head_results = {f"L{l}H{h}": {"dla_truth": [], "dla_lie": [], "dla_diff": []} for l, h in TOP_HEADS}

    print(f"\n[Dissection] Computing Direct Logit Attribution (DLA) across {len(dataset)} prompt pairs...", flush=True)

    for pair, truth_id, lie_id in dataset:
        with torch.no_grad():
            _, cache = model.run_with_cache(pair.sycophantic_prompt)

            for layer, head in TOP_HEADS:
                z_token = cache[f"blocks.{layer}.attn.hook_z"][0, -1, head, :]  # [d_head]
                W_O = model.blocks[layer].attn.W_O[head, :, :]  # [d_head, d_model]
                head_out = z_token @ W_O  # [d_model]

                W_U_truth = model.W_U[:, truth_id]  # [d_model]
                W_U_lie = model.W_U[:, lie_id]      # [d_model]

                dla_truth = (head_out @ W_U_truth).item()
                dla_lie = (head_out @ W_U_lie).item()
                dla_diff = dla_truth - dla_lie

                key = f"L{layer}H{head}"
                head_results[key]["dla_truth"].append(dla_truth)
                head_results[key]["dla_lie"].append(dla_lie)
                head_results[key]["dla_diff"].append(dla_diff)

            del cache
            clean_vram()

    aggregated = {}
    for key, vals in head_results.items():
        aggregated[key] = {
            "mean_truth": float(np.mean(vals["dla_truth"])),
            "mean_lie": float(np.mean(vals["dla_lie"])),
            "mean_diff": float(np.mean(vals["dla_diff"])),
        }
        print(f"  {key}: DLA(Truth)={aggregated[key]['mean_truth']:+.3f} | DLA(Lie)={aggregated[key]['mean_lie']:+.3f} | Net DLA(Truth-Lie)={aggregated[key]['mean_diff']:+.3f}", flush=True)

    head_names = list(aggregated.keys())
    truth_vals = [aggregated[h]["mean_truth"] for h in head_names]
    lie_vals = [aggregated[h]["mean_lie"] for h in head_names]
    diff_vals = [aggregated[h]["mean_diff"] for h in head_names]

    x = np.arange(len(head_names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - width, truth_vals, width, label="DLA(Truth)", color="#2a9d8f", edgecolor="#222222", alpha=0.9)
    ax.bar(x, lie_vals, width, label="DLA(Lie)", color="#e76f51", edgecolor="#222222", alpha=0.9)
    ax.bar(x + width, diff_vals, width, label="Net ΔL Contribution (Truth - Lie)", color="#264653", edgecolor="#222222", alpha=0.9)

    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_title("Direct Logit Attribution (DLA) Decomposition of Suppressor Heads", pad=14, fontweight="bold")
    ax.set_ylabel("Logit Projection Unit", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(head_names, fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Dissection] Saved Figure 4: {output_path}", flush=True)

    return aggregated

def main():
    print("=" * 65, flush=True)
    print("  PHASE 2: CIRCUIT DISSECTION OF SYCOPHANCY SUPPRESSOR HEADS", flush=True)
    print("=" * 65, flush=True)

    model, loaded_name = load_target_model()
    dataset = get_validated_dataset(model)
    print(f"[Dissection] Using model {loaded_name} with {len(dataset)} validated pairs.", flush=True)

    output_dir = os.path.join(base_dir, "outputs")
    fig3_path = os.path.join(output_dir, "suppressor_attention_patterns.png")
    fig4_path = os.path.join(output_dir, "direct_logit_attribution_decomposition.png")

    attn_info = analyze_attention_patterns(model, dataset, output_path=fig3_path)
    dla_info = analyze_direct_logit_attribution(model, dataset, output_path=fig4_path)

    print("\n" + "=" * 65, flush=True)
    print("  CIRCUIT DISSECTION SUMMARY REPORT", flush=True)
    print("=" * 65, flush=True)
    for head_name, metrics in dla_info.items():
        is_lie_booster = metrics["mean_lie"] > abs(metrics["mean_truth"])
        role = "Active Lie Booster (Directly elevates sycophantic prediction)" if is_lie_booster else "Truth Depressor (Suppresses factual activation)"
        print(f"* {head_name}: Net Contribution = {metrics['mean_diff']:+.3f} | Classified Role: {role}", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
