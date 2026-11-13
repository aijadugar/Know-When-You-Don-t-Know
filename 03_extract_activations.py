"""03_extract_activations.py — extract last-prompt-token hidden states."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ACTS_DIR,
    DATA_DIR,
    MODEL_ID,
    SEED,
    SMOKE_MODEL_ID,
    SYSTEM_PROMPT,
)
from common import build_prompt, load_jsonl, set_seed, load_model_and_tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model_id", type=str, default=None)
    return parser.parse_args()

def verify_hidden_state_structure(model: Any) -> dict:
    """Verify hidden_states structure in installed transformers.
    Returns dict with layer info for documentation.
    """
    import inspect
    from transformers import Qwen2Model

    # Check the docstring or source
    src = inspect.getsource(Qwen2Model.forward)
    lines = [l.strip() for l in src.split("\n")]

    # Look for output_hidden_states references
    hs_lines = [l for l in lines if "hidden_states" in l.lower()]
    print(f"  [verify] Found {len(hs_lines)} lines mentioning hidden_states in Qwen2Model.forward")

    # The base class PreTrainedModel documents:
    # hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True)
    # - output of embeddings, i.e. before first transformer block (index 0)
    # - output of last transformer block (index n_layers)
    # - output of final norm (index n_layers + 1? or n_layers is final?)
    #
    # For Qwen2Model specifically: output is a tuple where:
    # - index 0: hidden state after embedding layer (before first block)
    # - indices 1 through n_layers: each transformer block output
    # - index n_layers+1: final hidden state after all blocks and final norm
    #
    # Actually let me check: for Qwen2Model the hidden_states tuple length is n_layers + 1
    # index 0 = embeddings output, index 1..n_layers-1 = block outputs, index n_layers = final norm
    # Wait, let me be more careful. The spec says "n_layers+1 entries" and "index 0 is the embedding
    # output and the last entry is after the final norm". So the tuple has n_layers+1 elements.

    config = model.config
    n_layers = getattr(config, "num_hidden_layers", None)
    d_model = getattr(config, "hidden_size", None)

    print(f"  [verify] config.num_hidden_layers = {n_layers}")
    print(f"  [verify] config.hidden_size = {d_model}")
    print(f"  [verify] Expected hidden_states tuple length: {n_layers + 1}")
    print(f"  [verify] index 0 = embedding output, index {n_layers} = after final norm")

    return {
        "n_layers": n_layers,
        "d_model": d_model,
        "expected_length": n_layers + 1,
    }

def extract_activations(
    model: Any,
    tokenizer: Any,
    split_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    layer_info: dict,
) -> tuple[np.ndarray, str, float]:
    """Extract activations for one split.
    Returns: (activations_array, dtype_str, non_finite_fraction)
    """
    rows = load_jsonl(str(DATA_DIR / f"{split_name}_labeled.jsonl"))

    # Rebuild prompts (same as P3)
    questions = [r["question"] for r in rows]
    prompts = [build_prompt(tokenizer, q, SYSTEM_PROMPT) for q in questions]

    # Sort by prompt length for efficient batching
    sorted_data = sorted(
        zip(prompts, rows),
        key=lambda x: len(x[0]),
    )
    sorted_prompts = [d[0] for d in sorted_data]
    sorted_indices = [rows.index(d[1]) for d in sorted_data]

    n_layers = layer_info["n_layers"]
    d_model = layer_info["d_model"]

    # Pre-allocate output array
    n_samples = len(rows)
    activations = np.zeros((n_samples, n_layers + 1, d_model), dtype=np.float32)

    n_batches = (len(sorted_prompts) + batch_size - 1) // batch_size
    for batch_start in tqdm(
        range(0, len(sorted_prompts), batch_size),
        desc=f"  {split_name}",
        total=n_batches,
    ):
        batch_end = min(batch_start + batch_size, len(sorted_prompts))
        batch_prompts = sorted_prompts[batch_start:batch_end]

        # Tokenize with left padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Assert left padding: last position is real token for all rows
        assert attention_mask[:, -1].all(), (
            f"Batch {batch_start}: last position is padding for some row. "
            f"attention_mask[:, -1] = {attention_mask[:, -1]}"
        )

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states  # tuple of n_layers+1 tensors

        # Verify structure
        actual_layers = len(hidden_states)
        assert actual_layers == n_layers + 1, (
            f"Expected {n_layers + 1} hidden states, got {actual_layers}"
        )

        # Extract last token position for each layer
        # hidden_states[i] shape: [batch, seq_len, d_model]
        # Under left padding, position -1 is the last real token
        batch_size_actual = input_ids.shape[0]
        for layer_idx in range(n_layers + 1):
            hs = hidden_states[layer_idx].float()  # [batch, seq_len, d_model]
            last_hidden = hs[:, -1, :]  # [batch, d_model]
            activations[
                batch_start:batch_end, layer_idx, :
            ] = last_hidden.cpu().numpy()

    # Try casting to float16
    activations_f16 = activations.astype(np.float16)
    non_finite = ~np.isfinite(activations_f16)
    non_finite_count = non_finite.sum()
    non_finite_fraction = non_finite_count / activations_f16.size

    print(f"  {split_name}: non-finite after float16 cast: {non_finite_count} "
          f"({non_finite_fraction:.2e})")

    if non_finite_fraction > 0:
        print(f"\n*** WARNING: {non_finite_fraction*100:.2f}% of activations are "
              f"non-finite in float16. Using float32 instead. ***\n")
        final_activations = activations
        final_dtype = "float32"
    else:
        final_activations = activations_f16
        final_dtype = "float16"

    return final_activations, final_dtype, non_finite_fraction

def main():
    args = parse_args()
    set_seed(SEED)

    print("=" * 70)
    print("P4: ACTIVATION EXTRACTION")
    print("=" * 70)

    # Determine model
    if args.smoke:
        model_id = SMOKE_MODEL_ID
    elif args.model_id:
        model_id = args.model_id
    else:
        model_id = MODEL_ID

    print(f"Model: {model_id}")

    # Load model (no GPU at module level for fast tests)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model, tokenizer = load_model_and_tokenizer(model_id, dtype)
    model = model.to(device)
    model.eval()

    # Verify hidden state structure
    print("\n## Verifying hidden_states structure...")
    layer_info = verify_hidden_state_structure(model)

    n_layers = layer_info["n_layers"]
    d_model = layer_info["d_model"]

    # Extract activations for each split
    split_files = ["train", "val", "test", "ood"]
    results_info = []

    for split_name in split_files:
        label_path = DATA_DIR / f"{split_name}_labeled.jsonl"
        if not label_path.exists():
            print(f"  Skipping {split_name} (not found)")
            continue

        print(f"\n## Processing {split_name}...")

        activations, dtype_used, non_finite_frac = extract_activations(
            model, tokenizer, split_name,
            batch_size=32, device=device, dtype=dtype,
            layer_info=layer_info,
        )

        # Save with memmap for low RAM
        out_path = ACTS_DIR / f"{split_name}.npy"
        np.save(str(out_path), activations)

        file_size_mb = out_path.stat().st_size / 1e6
        results_info.append({
            "split": split_name,
            "shape": activations.shape,
            "dtype": dtype_used,
            "file_size_mb": file_size_mb,
            "non_finite_frac": non_finite_frac,
        })

        # Verify shape
        assert activations.shape[1] == n_layers + 1, (
            f"Expected {n_layers + 1} layers, got {activations.shape[1]}"
        )
        assert activations.shape[2] == d_model, (
            f"Expected d_model={d_model}, got {activations.shape[2]}"
        )

    # Print summary
    print(f"\n{'=' * 70}")
    print("P4 SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Split':<8} {'Shape':>20} {'Dtype':>10} {'File (MB)':>10} {'Non-finite':>12}")
    print("-" * 62)
    for r in results_info:
        print(f"{r['split']:<8} {str(r['shape']):>20} {r['dtype']:>10} "
              f"{r['file_size_mb']:>10.2f} {r['non_finite_frac']:>12.2e}")

    # Gate assertions
    print(f"\n## GATE assertions")
    for r in results_info:
        assert r["non_finite_frac"] == 0, (
            f"{r['split']} has non-finite activations: {r['non_finite_frac']}"
        )
        assert r["shape"][1] == n_layers + 1
        assert r["shape"][2] == d_model

    print("  ✓ All non-finite fractions are 0")
    print("  ✓ All shapes match expected [N, n_layers+1, d_model]")
    print(f"\nP4 GATE: PASSED")

if __name__ == "__main__":
    main()
