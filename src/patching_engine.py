"""
Patching engine module for causal activation patching in TransformerLens.
Implements:
1. Dynamic model loader with strict peak VRAM safety and auto-fallback.
2. Baseline Logit Difference measurement: Delta_L = Logit(Truth) - Logit(Lie).
3. Step A: Layer-wise MLP Output Patching at final prompt token.
4. Step B: Downstream Attention Head Attribution / Suppression Analysis across heads.
"""

import sys
import os

# Prioritize site-packages for transformer_lens (v3.9.0)
sys.path = [p for p in sys.path if p not in ("", "C:\\TransformerLens", "c:\\TransformerLens", "c:/TransformerLens", "C:/TransformerLens")]
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import gc
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformer_lens import HookedTransformer
from src.dataset import SycophancyPair, get_validated_dataset

def has_hf_token() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    token_file = os.path.expanduser("~/.cache/huggingface/token")
    return os.path.exists(token_file) and os.path.getsize(token_file) > 0

def load_target_model(preferred_model: str = "google/gemma-2-2b", fallback_model: str = "EleutherAI/pythia-1.4b") -> Tuple[HookedTransformer, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)

    model_to_try = preferred_model if has_hf_token() else fallback_model
    print(f"[Model Loader] Target device: {device} | Dtype: {dtype}")
    print(f"[Model Loader] HF Token detected: {has_hf_token()} | Selected model: {model_to_try}")

    try:
        model = HookedTransformer.from_pretrained(model_to_try, device=device, torch_dtype=dtype)
        return model, model_to_try
    except Exception as e:
        print(f"[Model Loader] Failed loading '{model_to_try}': {e}")
        print(f"[Model Loader] Falling back to open model: '{fallback_model}'")
        try:
            model = HookedTransformer.from_pretrained(fallback_model, device=device, torch_dtype=dtype)
            return model, fallback_model
        except Exception as e2:
            print(f"[Model Loader] Fallback to pythia failed ({e2}). Loading 'gpt2-medium'...")
            model = HookedTransformer.from_pretrained("gpt2-medium", device=device, torch_dtype=dtype)
            return model, "gpt2-medium"

def clean_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def calculate_logit_diff(logits: torch.Tensor, truth_token: int, lie_token: int) -> float:
    final_logits = logits[0, -1, :]
    return (final_logits[truth_token] - final_logits[lie_token]).item()

def run_mlp_layer_patching(
    model: HookedTransformer,
    dataset: List[Tuple[SycophancyPair, int, int]]
) -> Dict[str, any]:
    """
    Step A: Intercept final prompt token position.
    Patch hook_mlp_out layer-by-layer from clean run into sycophantic run.
    """
    n_layers = model.cfg.n_layers
    all_clean_diffs = []
    all_syc_diffs = []
    layer_patch_diffs = [[] for _ in range(n_layers)]

    print(f"\n[Patching Engine] Running MLP Output Patching across {len(dataset)} pairs ({n_layers} layers)...")

    for idx, (pair, truth_id, lie_id) in enumerate(dataset):
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(pair.clean_prompt)
            syc_logits, _ = model.run_with_cache(pair.sycophantic_prompt)

            c_diff = calculate_logit_diff(clean_logits, truth_id, lie_id)
            s_diff = calculate_logit_diff(syc_logits, truth_id, lie_id)
            all_clean_diffs.append(c_diff)
            all_syc_diffs.append(s_diff)

            for layer in range(n_layers):
                hook_name = f"blocks.{layer}.hook_mlp_out"
                clean_act = clean_cache[hook_name][:, -1, :].clone()

                def make_hook(saved_act):
                    def hook_fn(mlp_out, hook):
                        mlp_out[:, -1, :] = saved_act
                        return mlp_out
                    return hook_fn

                patched_logits = model.run_with_hooks(
                    pair.sycophantic_prompt,
                    fwd_hooks=[(hook_name, make_hook(clean_act))]
                )
                p_diff = calculate_logit_diff(patched_logits, truth_id, lie_id)
                layer_patch_diffs[layer].append(p_diff)

            del clean_cache, clean_logits, syc_logits
            clean_vram()

        print(f"  [{idx+1:02d}/{len(dataset):02d}] {pair.id}: Clean Delta_L={c_diff:+.2f} | Syc Delta_L={s_diff:+.2f}", flush=True)

    mean_clean = float(np.mean(all_clean_diffs))
    mean_syc = float(np.mean(all_syc_diffs))
    mean_layer_patches = [float(np.mean(diffs)) for diffs in layer_patch_diffs]
    std_layer_patches = [float(np.std(diffs)) for diffs in layer_patch_diffs]

    return {
        "n_layers": n_layers,
        "mean_clean_diff": mean_clean,
        "mean_syc_diff": mean_syc,
        "mean_layer_patches": mean_layer_patches,
        "std_layer_patches": std_layer_patches,
        "pair_ids": [p[0].id for p in dataset],
    }

def run_attention_head_patching(
    model: HookedTransformer,
    dataset: List[Tuple[SycophancyPair, int, int]],
    layers_to_scan: Optional[List[int]] = None
) -> np.ndarray:
    """
    Step B: Downstream Attention Head Attribution & Suppression Analysis.
    Intervene on hook_z [batch, pos, head, d_head] at position -1.
    Patches clean head activation into sycophantic prompt.
    Returns: Attribution matrix of shape [n_layers, n_heads] representing mean Delta_L shift.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if layers_to_scan is None:
        # Default to downstream layers (second half of model)
        layers_to_scan = list(range(n_layers // 2, n_layers))

    attribution_matrix = np.zeros((n_layers, n_heads), dtype=np.float32)
    print(f"\n[Patching Engine] Scanning Downstream Attention Heads ({len(layers_to_scan)} layers x {n_heads} heads)...", flush=True)

    # Use first 5 diverse pairs for high-precision attention head scan
    eval_subset = dataset[:5]

    for p_idx, (pair, truth_id, lie_id) in enumerate(eval_subset):
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(pair.clean_prompt)
            base_syc_logits = model(pair.sycophantic_prompt)
            base_syc_diff = calculate_logit_diff(base_syc_logits, truth_id, lie_id)

            for layer in layers_to_scan:
                hook_name = f"blocks.{layer}.attn.hook_z"
                clean_head_acts = clean_cache[hook_name][:, -1, :, :].clone()

                for head in range(n_heads):
                    clean_head = clean_head_acts[:, head, :].clone()

                    def make_head_hook(saved_head, h_idx):
                        def head_hook_fn(z, hook):
                            z[:, -1, h_idx, :] = saved_head
                            return z
                        return head_hook_fn

                    patched_logits = model.run_with_hooks(
                        pair.sycophantic_prompt,
                        fwd_hooks=[(hook_name, make_head_hook(clean_head, head))]
                    )
                    p_diff = calculate_logit_diff(patched_logits, truth_id, lie_id)
                    attribution_matrix[layer, head] += (p_diff - base_syc_diff)

            del clean_cache, base_syc_logits
            clean_vram()

        print(f"  [Attn Scan Pair {p_idx+1}/{len(eval_subset)}] {pair.id} complete.", flush=True)

    attribution_matrix /= len(eval_subset)
    return attribution_matrix
