"""Gradio abstention app (HF Spaces, ZeroGPU) -- STUB, implemented in prompt P8.

Planned flow (do not trust until it exists and its test passes): build the chat
prompt -> ONE forward pass with output_hidden_states=True -> hidden state of the
LAST PROMPT TOKEN at the probe layer -> logistic-regression risk P(y=1) where
y=1 means "the greedy answer is WRONG" -> if risk > threshold calibrated for
config.TARGET_COVERAGE, ABSTAIN without ever calling model.generate().

Nothing here is wired yet: no probe artifact, no metrics.json, no UI.  See README
for what actually exists so far.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "app.py is a stub: the Gradio/ZeroGPU app arrives in prompt P8. "
        "Run `make test` for the current gate."
    )


if __name__ == "__main__":
    main()
