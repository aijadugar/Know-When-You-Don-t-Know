# Know When You Don't Know — pre-generation abstention probe

**Problem.** An LLM states wrong facts as confidently as right ones. This project trains a
*linear probe* on the hidden state of the **last prompt token** of `Qwen/Qwen2.5-3B-Instruct`
to predict, **before generating**, whether its greedy answer to a factual question will be
wrong. A Gradio app on HF ZeroGPU abstains when the probe fires.

**Label convention (never inverted).** `y = 1` means the model's answer is **WRONG**
(hallucination), `y = 0` means correct. The probe's risk score is `P(y=1)`; the app abstains
when the risk exceeds a threshold calibrated for a target coverage.

## Status (honest)

Repo state: P1 scaffold + shared code + tests are in and green. Stages P2–P9 (data build,
answer generation/labelling, activation extraction, probe training, threshold evaluation,
Hub publishing, the Gradio app, README number audit) are implemented by later prompts; the
Makefile targets exist but their scripts are not written yet, so nothing here claims any
accuracy/AUROC result. `results/metrics.json` does not exist yet — no numbers are quoted.

## Layout

```
config.py                 every constant: MODEL_ID, SMOKE_MODEL_ID, SYSTEM_PROMPT, sizes, seeds, paths
common.py                 jsonl IO, build_prompt, load_model_and_tokenizer, normalize/is_match/labels, metrics
tests/test_common.py      normalization, whole-word matching, label convention, left-padding
requirements.txt          runtime (torch/transformers/datasets/sklearn/gradio/...)
requirements-dev.txt      + pytest
Makefile                  per-stage targets, plus --smoke variants
data/ acts/ models/ results/   created by the pipeline (git-ignored)
app.py                  placeholder until P8 (Gradio + ZeroGPU abstention UI)
01_build_data.py … 06_push_to_hub.py   filled in by later prompts
```

## Smoke harness

Every script takes `--model_id` and `--smoke`. `--smoke` swaps in
`Qwen/Qwen2.5-0.5B-Instruct`, tiny sizes (`SMOKE_N_*` in config.py) and separate
`data_smoke/ acts_smoke/ models_smoke/ results_smoke/` directories, so a CPU check can never
clobber a real run:

```bash
pip install -r requirements-dev.txt
python -m pytest -q            # P1 gate
make smoke_all                 # 0.5B end-to-end once stages P2+ land
```

## Environment facts verified in this repo (not assumed)

* `Qwen/Qwen2.5-3B-Instruct` and `Qwen/Qwen2.5-0.5B-Instruct` both exist on the Hub
  (`huggingface_hub.model_info`: 3,085,938,688 / 494,032,768 params, BF16).
* `transformers==4.53.2`: `GenerationMixin.compute_transition_scores(sequences, scores,
  beam_indices=None, normalize_logits=False)` exists on the model object.
* `transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward`: `hidden_states[0]` is the
  embedding output (appended before the first decoder layer) and the final entry is appended
  **after** `self.norm`, i.e. post-final-norm → `n_layers + 1` rows.
* dtype is chosen at runtime (`bfloat16` if `torch.cuda.is_bf16_supported()` else `float16`
  on CUDA, `float32` on CPU) and logged; free Colab T4s have no bf16, so the real run uses fp16.
* Deviation from the original plan: the C grid is `config.C_VALUES = [0.01, 0.1, 1.0]`.
