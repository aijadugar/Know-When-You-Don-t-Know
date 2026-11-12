"""ZeroGPU abstention demo (gate P8).

Flow: build the chat prompt -> ONE forward pass with output_hidden_states=True ->
take hidden_states[layer][:, -1, :] (last prompt token, left padding) -> probe
P(wrong).  If P(wrong) > threshold: ABSTAIN -- `model.generate` is provably never
called on that branch (tests/test_app.py asserts this with a CPU mock).  Only
confident rows reach generate().

ZeroGPU notes (from the Spaces docs): Gradio 4+, model moved to cuda at module
level, GPU work wrapped in @spaces.GPU(duration=...).  Local CPU runs fall back
to the smoke model so the mock test needs no GPU.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ABSTAIN_MSG = ("I'm not confident I know this one, so I'd rather not guess. "
               "(abstained before generating)")

_MODEL_ID = os.environ.get("ABSTAIN_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
_device = None  # resolved once in _lazy_state via config.resolve_device


def _identity_gpu(duration=None):
    """Fallback decorator when `spaces` (HF ZeroGPU) is not installed."""
    def deco(fn):
        return fn
    return deco


try:  # gradio only needed for the demo; tests import answer() without it
    import gradio as gr
except Exception:  # pragma: no cover - local/Colab fallback
    gr = None

try:
    from spaces import GPU as _gpu_decorator  # type: ignore
except Exception:
    _gpu_decorator = _identity_gpu


def _lazy_state():
    global _model, _tok, _bundle, _device, _threshold, _ready
    if globals().get("_ready"):
        return
    import json

    import joblib

    import config
    from common import load_model_and_tokenizer

    _device = config.resolve_device(None)
    _model, _tok, _dtype_used = load_model_and_tokenizer(_MODEL_ID, _device, None)
    _bundle = joblib.load(config.PROBE_PATH)
    with open(config.METRICS_PATH) as f:
        _threshold = float(json.load(f)["threshold_p_wrong"])
    _ready = True


if gr is not None and os.environ.get("ABSTAIN_ON_HUB") == "1":
    # ZeroGPU requirement: move the model to cuda at module level
    _lazy_state()
    _model.to("cuda")
    _device = "cuda"


@_gpu_decorator(duration=20)
def answer(question: str) -> str:
    """Return the answer, or ABSTAIN_MSG.  The abstain path never calls generate."""
    import torch

    import config
    from common import build_prompt, generate_and_score

    _lazy_state()
    prompt = build_prompt(question, _tok)
    enc = _tok([prompt], return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        out = _model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[_bundle["layer"]][:, -1, :].float().cpu().numpy()
    p_wrong = float(_bundle["probe"].predict_proba(hs)[0, 1])
    if p_wrong > _threshold:
        # ABSTAIN: deliberately NO call to _model.generate anywhere below.
        return f"{ABSTAIN_MSG}  [P(wrong)={p_wrong:.2f}]"
    res = generate_and_score(_model, _tok, [prompt], _device,
                             max_new_tokens=config.MAX_NEW_TOKENS, batch_size=1)
    return res["answers"][0]


def main():
    assert gr is not None, "gradio not installed; run pip install -r requirements.txt"
    demo = gr.Interface(fn=answer, inputs="textbox", outputs="text",
                        title="Know-When-You-Don't-Know",
                        description="Answers only when a linear probe on the last "
                                    "prompt-token activation predicts the answer will be right.")
    demo.launch()


if __name__ == "__main__":
    main()
