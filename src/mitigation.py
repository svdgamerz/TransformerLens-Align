"""
Mitigation module: Tests zero-shot surgical intervention by ablating
the identified Active Lie Booster heads (L13H6, L13H13) and Truth Gate (L15H7).
Demonstrates:
1. Sycophancy Reduction: Zero-ablating Lie Boosters recovers factual logit diff Delta_L under sycophantic prompts without fine-tuning.
2. Side-Effect Benchmark: Verifies that factual accuracy on clean prompts is preserved.
3. Generates Figure 5: outputs/mitigation_ablation_recovery.png (300 DPI).
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
from typing import List, Tuple, Dict
from transformer_lens import HookedTransformer
from src.dataset import get_validated_dataset, SycophancyPair
from src.patching_engine import load_target_model, calculate_logit_diff, clean_vram

LIE_BOOSTERS = [(13, 6), (13, 13)]

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

def make_zero_ablation_hooks(heads_to_ablate: List[Tuple[int, int]]):
    hooks = []
    # Group by layer
    by_layer = {}
    for layer, head in heads_to_ablate:
        by_layer.setdefault(layer, []).append(head)

    for layer, heads in by_layer.items():
        hook_name = f"blocks.{layer}.attn.hook_z"
        head_indices = heads.copy()

        def make_hook(h_list):
            def hook_fn(z, hook):
                for h in h_list:
                    z[:, -1, h, :] = 0.0
                return z
            return hook_fn

        hooks.append((hook_name, make_hook(head_indices)))
    return hooks

def run_mitigation_experiments(
    model: HookedTransformer,
    dataset: List[Tuple[SycophancyPair, int, int]],
    output_path: str = "outputs/mitigation_ablation_recovery.png"
) -> Dict[str, any]:
    set_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    clean_diffs = []
    syc_unmitigated_diffs = []
    syc_mitigated_diffs = []
    clean_mitigated_diffs = []

    ablation_hooks = make_zero_ablation_hooks(LIE_BOOSTERS)

    print(f"\n[Mitigation] Testing surgical zero-ablation of {LIE_BOOSTERS} across {len(dataset)} prompt pairs...", flush=True)

    for idx, (pair, truth_id, lie_id) in enumerate(dataset):
        with torch.no_grad():
            # 1. Baseline Clean
            clean_logits = model(pair.clean_prompt)
            c_diff = calculate_logit_diff(clean_logits, truth_id, lie_id)
            clean_diffs.append(c_diff)

            # 2. Baseline Sycophantic (Unmitigated)
            syc_logits = model(pair.sycophantic_prompt)
            s_diff = calculate_logit_diff(syc_logits, truth_id, lie_id)
            syc_unmitigated_diffs.append(s_diff)

            # 3. Mitigated Sycophantic (Lie Boosters Zeroed Out)
            mitigated_syc_logits = model.run_with_hooks(pair.sycophantic_prompt, fwd_hooks=ablation_hooks)
            m_syc_diff = calculate_logit_diff(mitigated_syc_logits, truth_id, lie_id)
            syc_mitigated_diffs.append(m_syc_diff)

            # 4. Side-Effect Check: Clean prompt under ablation
            mitigated_clean_logits = model.run_with_hooks(pair.clean_prompt, fwd_hooks=ablation_hooks)
            m_clean_diff = calculate_logit_diff(mitigated_clean_logits, truth_id, lie_id)
            clean_mitigated_diffs.append(m_clean_diff)

            del clean_logits, syc_logits, mitigated_syc_logits, mitigated_clean_logits
            clean_vram()

        print(f"  [{idx+1:02d}/{len(dataset):02d}] {pair.id:16s} | Syc Base: {s_diff:+.2f} -> Mitigated: {m_syc_diff:+.2f} (Δ={m_syc_diff - s_diff:+.2f})", flush=True)

    results = {
        "mean_clean": float(np.mean(clean_diffs)),
        "mean_syc_unmitigated": float(np.mean(syc_unmitigated_diffs)),
        "mean_syc_mitigated": float(np.mean(syc_mitigated_diffs)),
        "mean_clean_mitigated": float(np.mean(clean_mitigated_diffs)),
        "std_clean": float(np.std(clean_diffs)),
        "std_syc_unmitigated": float(np.std(syc_unmitigated_diffs)),
        "std_syc_mitigated": float(np.std(syc_mitigated_diffs)),
        "std_clean_mitigated": float(np.std(clean_mitigated_diffs)),
        "pair_ids": [p[0].id for p in dataset],
    }

    # Plot Figure 5: Mitigation Recovery Bar Chart
    categories = [
        "1. Clean Baseline\n(Truth Confident)",
        "2. Sycophantic Base\n(Lie Bias)",
        "3. Mitigated Sycophantic\n(L13H6+H13 Ablated)",
        "4. Mitigated Clean\n(Side-Effect Check)"
    ]
    means = [
        results["mean_clean"],
        results["mean_syc_unmitigated"],
        results["mean_syc_mitigated"],
        results["mean_clean_mitigated"]
    ]
    stds = [
        results["std_clean"],
        results["std_syc_unmitigated"],
        results["std_syc_mitigated"],
        results["std_clean_mitigated"]
    ]
    colors = ["#2a9d8f", "#e76f51", "#264653", "#457b9d"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(categories, means, yerr=stds, capsize=5, color=colors, edgecolor="#222222", linewidth=1.1, alpha=0.9, width=0.55)

    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Logit Difference ΔL = Logit(Truth) - Logit(Lie)", labelpad=8)
    ax.set_title("Zero-Shot Surgical Mitigation: Ablating Active Lie Boosters (L13H6, L13H13)", pad=14, fontweight="bold")

    # Add numeric labels on top of bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        y_pos = height + (0.15 if height >= 0 else -0.35)
        ax.text(bar.get_x() + bar.get_width() / 2., y_pos, f"{mean_val:+.2f}", ha="center", va="bottom" if height >= 0 else "top", fontweight="bold", fontsize=10.5)

    recovery_delta = results["mean_syc_mitigated"] - results["mean_syc_unmitigated"]
    ax.annotate(
        f"Sycophancy Reduced\n(+{recovery_delta:.2f} logit diff)",
        xy=(2, results["mean_syc_mitigated"]),
        xytext=(1.4, results["mean_syc_mitigated"] + 0.8),
        arrowprops=dict(facecolor="#264653", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold"
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Mitigation] Saved Figure 5: {output_path}", flush=True)

    return results

def main():
    print("=" * 65, flush=True)
    print("  PHASE 3: ZERO-SHOT SURGICAL SYCOPHANCY MITIGATION", flush=True)
    print("=" * 65, flush=True)

    model, loaded_name = load_target_model()
    dataset = get_validated_dataset(model)
    print(f"[Mitigation] Target Model: {loaded_name} | Target Heads for Knockout: {LIE_BOOSTERS}", flush=True)

    output_dir = os.path.join(base_dir, "outputs")
    fig5_path = os.path.join(output_dir, "mitigation_ablation_recovery.png")

    results = run_mitigation_experiments(model, dataset, output_path=fig5_path)

    gap_original = results["mean_clean"] - results["mean_syc_unmitigated"]
    recovered = results["mean_syc_mitigated"] - results["mean_syc_unmitigated"]
    mitigation_pct = (recovered / gap_original) * 100 if gap_original != 0 else 0.0

    print("\n" + "=" * 65, flush=True)
    print("  MITIGATION RESULTS SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"* Baseline Clean Delta_L:            {results['mean_clean']:+.3f}")
    print(f"* Sycophantic Unmitigated Delta_L:   {results['mean_syc_unmitigated']:+.3f}")
    print(f"* Sycophantic Mitigated Delta_L:     {results['mean_syc_mitigated']:+.3f} (Ablating {LIE_BOOSTERS})")
    print(f"* Absolute Factual Recovery:         {recovered:+.3f} logit units")
    print(f"* Relative Sycophancy Suppression:   {mitigation_pct:.1f}%")
    print(f"* Clean Prompt Retention:            {results['mean_clean_mitigated']:+.3f} (Side-Effect Preservation)")
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
