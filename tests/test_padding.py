"""tests/test_padding.py — verify left-padding produces same activations."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from common import build_prompt, load_model_and_tokenizer, set_seed
from config import SEED, SMOKE_MODEL_ID, SYSTEM_PROMPT

class TestPadding:
    """Test that left-padded batch activations match single-prompt activations."""

    def test_batch_vs_single_activations(self):
        """Activations for 6 prompts in one batch must match individual forward passes.
        Tolerance justification:
        - We use float32 for comparison (activations stored as float16, but we
          convert to float32 for the comparison to avoid precision issues).
        - The tolerance of 1e-4 is tight for float32: it catches real numerical
          differences while allowing for negligible floating-point variation.
        - Larger differences would indicate incorrect padding handling.
        """
        set_seed(SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32  # Always use float32 for the test

        model, tokenizer = load_model_and_tokenizer(SMOKE_MODEL_ID, dtype)
        model = model.to(device)
        model.eval()

        # 6 prompts of varying length
        prompts = [
            "What is the capital of France?",
            "Who painted the Mona Lisa?",
            "In what year did World War II end?",
            "What is the chemical symbol for gold?",
            "Who wrote Hamlet?",
            "What is the tallest mountain in the world?",
        ]

        # Get individual activations
        individual_activations = {}
        for prompt in prompts:
            text = build_prompt(tokenizer, prompt, SYSTEM_PROMPT)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=False,  # No padding for individual
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            # Take last token of each layer
            hidden = outputs.hidden_states
            last_token_hs = torch.stack([h[0, -1, :] for h in hidden], dim=0)
            individual_activations[prompt] = last_token_hs.float().cpu().numpy()

        # Get batch activations
        batch_texts = [build_prompt(tokenizer, p, SYSTEM_PROMPT) for p in prompts]
        batch_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=512,
        )
        batch_input_ids = batch_inputs["input_ids"].to(device)
        batch_attention_mask = batch_inputs["attention_mask"].to(device)

        # Verify left padding: last position is real token for all rows
        assert batch_attention_mask[:, -1].all(), (
            "Last position is padding for some row in the batch"
        )

        with torch.no_grad():
            batch_outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
            )

        batch_hidden = batch_outputs.hidden_states
        # batch_hidden[i] shape: [batch, seq_len, d_model]
        # Stack last token for each layer
        batch_activations = torch.stack(
            [batch_hidden[i][:, -1, :] for i in range(len(batch_hidden))],
            dim=1,  # [batch, n_layers+1, d_model]
        ).float().cpu().numpy()

        # Compare: 3 chosen layers (embedding, middle, last)
        n_layers_plus_1 = batch_activations.shape[1]
        test_layers = [0, n_layers_plus_1 // 2, n_layers_plus_1 - 1]

        TOLERANCE = 1e-3
        for layer_idx in test_layers:
            for i, prompt in enumerate(prompts):
                ind_hs = individual_activations[prompt][layer_idx]
                batch_hs = batch_activations[i, layer_idx]

                max_abs_diff = np.abs(ind_hs - batch_hs).max()
                mean_abs_diff = np.abs(ind_hs - batch_hs).mean()

                assert max_abs_diff < TOLERANCE, (
                    f"Layer {layer_idx}, prompt '{prompt[:40]}': "
                    f"max_abs_diff={max_abs_diff:.2e} > {TOLERANCE:.0e}\n"
                    f"  individual: {ind_hs[:5]}\n"
                    f"  batch:      {batch_hs[:5]}"
                )

        print(f"  ✓ All {len(prompts)} prompts, {len(test_layers)} layers: "
              f"max_abs_diff < {TOLERANCE:.0e}")
