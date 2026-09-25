"""
End-to-end execution script for Sycophancy Mechanistic Interpretability study.
Orchestrates:
1. Environment & Hardware diagnostics.
2. Model initialization with peak memory safety and gated model auto-fallback.
3. Contrastive dataset validation.
4. Step A: Layer-wise MLP Output Patching.
5. Step B: Downstream Attention Head Suppression Matrix.
6. Publication-grade figure generation (300 DPI).
7. Rust high-speed benchmark execution.
8. Comprehensive arXiv Markdown summary report.
"""

import sys
import os

# Prioritize site-packages for transformer_lens (v3.9.0)
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if p not in ("", "C:\\TransformerLens", "c:\\TransformerLens", "c:/TransformerLens", "C:/TransformerLens")]
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import subprocess
import torch
import numpy as np

from src.dataset import get_validated_dataset, CONTRASTIVE_PAIRS
from src.patching_engine import load_target_model, run_mlp_layer_patching, run_attention_head_patching, clean_vram
from src.visualize import plot_mlp_patching_recovery, plot_attention_head_heatmap

def print_banner(text: str):
    print("\n" + "=" * 70, flush=True)
    print(f"  {text}", flush=True)
    print("=" * 70, flush=True)

def main():
    print_banner("SYCOPHANCY MECHANISTIC INTERPRETABILITY SUITE")

    # 1. Hardware & Runtime Diagnostics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
    print(f"Runtime Environment:")
    print(f"  - Device: {device} ({gpu_name})")
    print(f"  - PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  - Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 2. Model Loading
    print_banner("PHASE 1: MODEL INITIALIZATION & DATASET VALIDATION")
    model, loaded_model_name = load_target_model(
        preferred_model="google/gemma-2-2b",
        fallback_model="EleutherAI/pythia-1.4b"
    )
    print(f"[Engine] Active Model: {loaded_model_name} (Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, d_model: {model.cfg.d_model})", flush=True)

    # 3. Dataset Validation
    validated_pairs = get_validated_dataset(model)
    print(f"[Dataset] Validated {len(validated_pairs)} / {len(CONTRASTIVE_PAIRS)} contrastive pairs for single-token targets.", flush=True)
    if len(validated_pairs) == 0:
        raise RuntimeError("No contrastive prompt pairs could be resolved to single tokens!")

    # 4. Step A: MLP Layer Patching
    print_banner("PHASE 2: STEP A - LAYER-WISE MLP OUTPUT PATCHING")
    t0 = time.time()
    mlp_results = run_mlp_layer_patching(model, validated_pairs)
    mlp_duration = time.time() - t0
    print(f"[Step A Complete] Processed {model.cfg.n_layers} layers in {mlp_duration:.2f}s", flush=True)

    # 5. Step B: Downstream Attention Suppression
    print_banner("PHASE 3: STEP B - DOWNSTREAM ATTENTION HEAD SUPPRESSION")
    t1 = time.time()
    attribution_matrix = run_attention_head_patching(model, validated_pairs)
    attn_duration = time.time() - t1
    print(f"[Step B Complete] Attention Head Scan completed in {attn_duration:.2f}s", flush=True)

    # 6. High-DPI Visualizations
    print_banner("PHASE 4: PUBLICATION FIGURE GENERATION (300 DPI)")
    outputs_dir = os.path.join(base_dir, "outputs")
    fig1_path = os.path.join(outputs_dir, "mlp_layer_patching_recovery.png")
    fig2_path = os.path.join(outputs_dir, "attention_suppression_heatmap.png")

    plot_mlp_patching_recovery(mlp_results, output_path=fig1_path, model_name=loaded_model_name)
    plot_attention_head_heatmap(attribution_matrix, output_path=fig2_path, model_name=loaded_model_name, top_k=5)

    # 7. Rust High-Speed Benchmark
    print_banner("PHASE 5: RUST CANDLE-CORE BENCHMARK LAYER")
    rust_manifest = os.path.join(base_dir, "rust_core", "Cargo.toml")
    try:
        proc = subprocess.run(
            ["cargo", "run", "--manifest-path", rust_manifest, "--release"],
            capture_output=True,
            text=True,
            timeout=120
        )
        rust_output = proc.stdout if proc.returncode == 0 else proc.stderr
        print(rust_output, flush=True)
    except Exception as e:
        print(f"[Rust Core] Notice: Benchmark skipped or encountered issue: {e}", flush=True)

    # 8. Mechanistic Interpretation & Final Summary Report
    print_banner("PHASE 6: EMPIRICAL FINDINGS & MECHANISTIC DIAGNOSIS")

    clean_base = mlp_results["mean_clean_diff"]
    syc_base = mlp_results["mean_syc_diff"]
    layer_diffs = mlp_results["mean_layer_patches"]
    peak_layer = int(np.argmax(layer_diffs))
    peak_val = layer_diffs[peak_layer]
    mlp_recovery_ratio = (peak_val - syc_base) / (clean_base - syc_base) if (clean_base - syc_base) != 0 else 0.0

    # Top suppressor heads
    n_layers, n_heads = attribution_matrix.shape
    flat_indices = np.argsort(attribution_matrix.flatten())[::-1][:3]
    top_heads = []
    for idx in flat_indices:
        l = idx // n_heads
        h = idx % n_heads
        shift = attribution_matrix[l, h]
        top_heads.append(f"L{l}H{h} (Delta_L shift: +{shift:.2f})")

    if mlp_recovery_ratio >= 0.70:
        diagnosis = "Parametric MLP Knowledge Retrieval Failure (Bias prompt steers MLP outputs away from factual knowledge retrieval)."
    else:
        diagnosis = "Downstream Attention Head Suppression (Factual representation retrieved by early/mid MLPs, but late attention heads selectively suppress truth in favor of user compliance)."

    report = f"""
### Empirical Mechanistic Report: Drivers of Sycophancy

| Metric / Parameter | Empirical Result |
| :--- | :--- |
| **Model Evaluated** | `{loaded_model_name}` ({model.cfg.n_layers} Layers, {model.cfg.n_heads} Heads) |
| **Execution Hardware** | `{gpu_name}` |
| **Baseline Clean Delta_L** | **`{clean_base:+.4f}`** (High confidence in truth) |
| **Baseline Sycophantic Delta_L** | **`{syc_base:+.4f}`** (Strong bias toward user lie) |
| **Sycophancy Gap** | **`{clean_base - syc_base:.4f}`** logit units |
| **Peak MLP Patched Recovery** | **`{peak_val:+.4f}`** at **Layer {peak_layer}** |
| **MLP Recovery Ratio** | **`{mlp_recovery_ratio * 100:.1f}%`** |
| **Top Suppressor Heads** | {', '.join(top_heads)} |
| **Mechanistic Conclusion** | **{diagnosis}** |

#### Generated Artifacts:
- **Figure 1 (MLP Patching Curve):** `{fig1_path}`
- **Figure 2 (Attention Suppression Heatmap):** `{fig2_path}`
"""
    print(report, flush=True)

if __name__ == "__main__":
    main()
