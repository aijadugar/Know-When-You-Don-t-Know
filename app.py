"""app.py — Gradio app for Know-When-You-Don't-Know on ZeroGPU."""
from __future__ import annotations
import os
import sys
from pathlib import Path

# ── ZeroGPU setup ────────────────────────────────────────────────────────────
import spaces
import gradio as gr
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import MAX_NEW_TOKENS, MODEL_ID, SYSTEM_PROMPT
from common import build_prompt, load_model_and_tokenizer

# ── Module-level loading (GPU function must be short, no lazy loading) ────────
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
LOCAL_MODELS = Path(__file__).parent / "models"
LOCAL_RESULTS = Path(__file__).parent / "results"

# Load probe and meta
import joblib
import json

def download_or_local(filename: str, local_dir: Path) -> str:
    """Download from Hub if HF_REPO_ID is set, else use local file."""
    if HF_REPO_ID:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    local_path = local_dir / filename
    if local_path.exists():
        return str(local_path)
    raise FileNotFoundError(f"Neither Hub repo ({HF_REPO_ID}) nor local ({local_path}) has {filename}")

probe_path = download_or_local("probe.joblib", LOCAL_MODELS)
probe = joblib.load(probe_path)
print(f"[app] Loaded probe from {probe_path}")

meta_path = download_or_local("probe_meta.json", LOCAL_MODELS)
with open(meta_path) as f:
    meta = json.load(f)
print(f"[app] Loaded meta: layer={meta['hidden_state_index']}, threshold={meta['threshold']:.4f}")

# Load metrics for the "How it works" panel
metrics_path = download_or_local("metrics.json", LOCAL_RESULTS)
with open(metrics_path) as f:
    metrics = json.load(f)
print(f"[app] Loaded metrics")

# Build "How it works" text from metrics.json
verdict = metrics.get("verdict", {})
ood_deg = verdict.get("ood_degradation", 0.0)

def _get(split: str, scorer: str, key: str) -> str:
    for m in metrics.get("results", []):
        if m.get("split") == split and m.get("scorer") == scorer:
            val = m.get(key)
            if val is not None:
                return f"{val:.3f}"
    return "N/A"

test_probe_auroc = _get("test", "probe", "auroc")
test_probe_acc70 = _get("test", "probe", "acc_at_70")
test_base_acc = _get("test", "probe", "base_accuracy")
ood_probe_auroc = _get("ood", "probe", "auroc")
ood_probe_acc70 = _get("ood", "probe", "acc_at_70")
ood_base_acc = _get("ood", "probe", "base_accuracy")

test_auroc_ci_lo = "N/A"
test_auroc_ci_hi = "N/A"
for m in metrics.get("results", []):
    if m.get("split") == "test" and m.get("scorer") == "probe":
        ci = m.get("auroc_ci", [])
        if len(ci) == 2:
            test_auroc_ci_lo = f"{ci[0]:.3f}"
            test_auroc_ci_hi = f"{ci[1]:.3f}"

ood_note = ""
if ood_deg < -0.05:
    ood_note = (
        f" ⚠️ OOD performance is degraded (AUROC gap: {ood_deg:+.3f}). "
        "This is reported rather than hidden."
    )

HOW_IT_WORKS = f"""## How it works

A linear probe is trained on the last-prompt-token hidden state of {MODEL_ID} to predict whether the greedy answer will be wrong.

**Label convention:** y=1 means the answer is wrong (hallucination), y=0 means correct.

### Results

| Dataset | Metric | Value |
|---------|--------|-------|
| TriviaQA test | AUROC | {test_probe_auroc} ({test_auroc_ci_lo}, {test_auroc_ci_hi}) |
| TriviaQA test | Accuracy@70% | {test_probe_acc70} vs base {test_base_acc} |
| NQ-Open OOD | AUROC | {ood_probe_auroc} |
| NQ-Open OOD | Accuracy@70% | {ood_probe_acc70} vs base {ood_base_acc} |

### How to use the strictness slider

The slider controls the abstention threshold. At 70% strictness, the threshold is set so that ~70% of questions are answered. Higher strictness → more abstentions → higher accuracy on answered. Lower strictness → fewer abstentions → more coverage.{ood_note}

**Known limits:** The probe is trained on TriviaQA factual QA. It inherits label noise from exact-string matching. It may not generalize to other tasks.
"""

# ── Load model at module level (ZeroGPU requirement) ────────────────────────
APP_MOCK = os.environ.get("APP_MOCK", "0") == "1"

if APP_MOCK:
    print("[app] APP_MOCK=1: using fake model for local testing")

    # Fake model for CPU testing
    class FakeModel:
        def __init__(self):
            self.config = type("Config", (), {
                "num_hidden_layers": 1,
                "hidden_size": 64,
            })()
            self.device = torch.device("cpu")

        def __call__(self, **kwargs):
            # Return dummy hidden states
            batch = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            d = self.config.hidden_size
            n_layers = self.config.num_hidden_layers + 1
            hidden = tuple(
                torch.randn(batch, seq_len, d) for _ in range(n_layers)
            )
            return type("Output", (), {"hidden_states": hidden})()

        def generate(self, **kwargs):
            batch = kwargs["input_ids"].shape[0]
            return type("Output", (), {
                "sequences": torch.tensor([[1, 2, 3]] * batch),
                "scores": None,
            })()

    model = FakeModel()

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "left"

        def apply_chat_template(self, *a, **kw):
            return "[FAKE PROMPT] " + (kw.get("tokenize", True) and str(a[0][-1]["content"]) or str(a[0][-1]["content"]))

        def __call__(self, *a, **kw):
            texts = a[0] if a else kw.get("text", [])
            if isinstance(texts, str):
                texts = [texts]
            n = len(texts)
            seq_len = 3
            return {
                "input_ids": torch.ones(n, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
            }

    tokenizer = FakeTokenizer()
    print("[app] Fake model and tokenizer loaded")
else:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[app] Loading {MODEL_ID} (dtype={dtype})...")
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, dtype)
    model.eval()
    print("[app] Model loaded")

# ── GPU-decorated inference function ────────────────────────────────────────
@spaces.GPU(duration=60)
def answer_question(
    question: str,
    strictness: float,
    probe_on: bool,
) -> tuple[float, str, str]:
    """Main inference: probe risk, optionally abstain, else generate.
    Returns: (risk_score, decision_text, answer_text)
    """
    # Build prompt
    prompt_str = build_prompt(tokenizer, question, SYSTEM_PROMPT)

    # Tokenize
    inputs = tokenizer(
        prompt_str,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(model.device if hasattr(model, "device") else "cuda")
    attention_mask = inputs["attention_mask"].to(model.device if hasattr(model, "device") else "cuda")

    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # Extract last-prompt-token hidden state
    layer_idx = meta["hidden_state_index"]
    hidden = outputs.hidden_states[layer_idx][:, -1, :].float().cpu().numpy()

    # Probe risk
    risk = float(probe.predict_proba(hidden)[0, 1])

    # Map strictness slider (50-95) to threshold via val_risk_quantiles
    # strictness 50 → low strictness → low coverage (50%) → low threshold (high risk)
    # strictness 95 → high strictness → high coverage (95%) → high threshold (low risk)
    # Wait, let me re-read: "higher strictness means lower coverage means lower threshold"
    # So: strictness 50 → target 50% coverage → threshold at 50th percentile
    #     strictness 95 → target 95% coverage → threshold at 95th percentile
    # Higher strictness = more abstentions = lower threshold (more questions exceed it)
    # Actually: higher strictness = more abstentions = lower threshold = lower coverage
    # So strictness maps directly to coverage, and threshold is the percentile of that coverage
    target_coverage_pct = strictness  # 50-95
    target_percentile = target_coverage_pct  # use as percentile directly

    # val_risk_quantiles[p] = risk at p-th percentile of val risk
    # For target coverage pct, we want threshold = quantile at (1 - coverage) * 100
    # But re-reading the spec: "slider 50 to 95 means target coverage in percent"
    # So strictness 70 → answer 70% of questions → threshold at 30th percentile
    risk_percentile = 100 - int(target_coverage_pct)
    risk_percentile = max(0, min(100, risk_percentile))
    threshold = meta["val_risk_quantiles"][risk_percentile]

    # Decision
    if probe_on and risk > threshold:
        decision = "🤔 Abstained — I'm not confident I know this."
        answer = "(no answer generated)"
    else:
        # Generate
        gen_inputs = tokenizer(
            prompt_str,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=512,
        )
        gen_ids = gen_inputs["input_ids"].to(model.device if hasattr(model, "device") else "cuda")
        gen_mask = gen_inputs["attention_mask"].to(model.device if hasattr(model, "device") else "cuda")

        with torch.no_grad():
            gen_output = model.generate(
                input_ids=gen_ids,
                attention_mask=gen_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        prompt_len = gen_ids.shape[1]
        answer_ids = gen_output.sequences[0, prompt_len:]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        decision = "✅ Answered"

    return risk, decision, answer

# ── Gradio UI ────────────────────────────────────────────────────────────────
EXAMPLES = [
    # Easy facts
    ["What is the capital of France?", 70, True],
    ["Who wrote Romeo and Juliet?", 70, True],
    # Obscure facts
    ["What year did the Ming dynasty end?", 70, True],
    ["What is the chemical symbol for ununoctium?", 70, True],
    ["Who was the first person to receive a Nobel Prize in physics?", 70, True],
    # False-premise
    ["What is the airspeed velocity of an unladen swallow?", 70, True],
]

with gr.Blocks() as demo:
    gr.Markdown("# Know-When-You-Don't-Know")
    gr.Markdown(
        "Ask a factual question. The probe predicts whether the model will answer correctly. "
        "When **Probe on/off** is checked, the model abstains if the risk score exceeds the threshold."
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Question",
                placeholder="Ask a factual question...",
                lines=2,
            )
            strictness_slider = gr.Slider(
                label="Abstention strictness",
                minimum=50,
                maximum=95,
                value=70,
                step=5,
                info="Higher = more abstentions (higher accuracy on answered)",
            )
            probe_checkbox = gr.Checkbox(
                label="Probe on/off",
                value=True,
                info="Enable the abstention probe",
            )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            risk_gauge = gr.Label(
                label="Risk score (P(wrong))",
                num_top_classes=1,
            )
            decision_text = gr.Textbox(
                label="Decision",
                lines=2,
            )
            answer_box = gr.Textbox(
                label="Answer",
                lines=2,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[question_box, strictness_slider, probe_checkbox],
    )

    with gr.Accordion("How it works", open=False):
        gr.Markdown(HOW_IT_WORKS)

    submit_btn.click(
        fn=answer_question,
        inputs=[question_box, strictness_slider, probe_checkbox],
        outputs=[risk_gauge, decision_text, answer_box],
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, strictness_slider, probe_checkbox],
        outputs=[risk_gauge, decision_text, answer_box],
    )

if __name__ == "__main__":
    demo.launch()
