# Gated agent prompts P0–P9 (Know-When-You-Don't-Know)

Each prompt is gated: no prompt starts until the previous gate passes. Every stage
reads the files produced by the previous stage. `pytest -q` must stay green at every gate.

## Confirmed environment facts (verified in this repo, not assumed)
- Qwen2.5-3B-Instruct: 36 hidden layers + embedding row = 37 hidden states, d_model=2048 (read from config.json).
- TriviaQA rc.nocontext: train 138,384 / val 17,944 rows with non-empty `answer.normalized_aliases`;
  **test split answers are withheld** (all empty, `value='<unk>'`) → test split is NEVER used;
  train+validation are sampled into disjoint pools instead.
- NQ-Open: fields `{question: str, answer: list[str]}` (NOT `answer.aliases`); train 87,925 / val 3,610, all non-empty.
- Qwen2.5-0.5B-Instruct exists and loads; usable for smoke tests on CPU.
- This box has no CUDA (Colab T4 is the target): dtype chosen per-device and logged (bf16 where supported, else fp16/cuda or fp32/cpu).

## P0 — Master context
Send the project rules (below) and require the agent to restate them, especially the label convention:
**y = 1 means the greedy answer matches NO gold alias (the model is WRONG); y = 0 means correct.**
Gate: restatement matches these rules verbatim.

## P1 — Scaffold and smoke harness
Create `config.py`, `common.py`, `tests/`, and a 0.5B end-to-end dry run over 20 questions that reaches
a probe AUROC number without crashing. Gate: `pytest -q` green AND dry run prints an AUROC.

## P2 — Data
Produce `data/{train,val,test,ood}.jsonl` with question + gold alias lists.
Gate: no duplicate questions across splits; every gold list non-empty; counts match `--n_*` flags.

## P3 — Answer and label
Produce `data/*_labeled.jsonl` (greedy answer + y). Unit test asserts y=1 ⇔ wrong.
Gate: hand-check of 40 printed rows; wrong-rate between 20% and 80%.

## P4 — Activations
Produce `acts/*.npy`, float16, shape `[N, n_layers+1, d_model]` = hidden state at the LAST PROMPT TOKEN
(`output_hidden_states=True`, index `-1` on the sequence axis under left padding; NOT forward hooks).
Gate: batched activations equal single-prompt activations within tolerance.

## P5 — Probe and baselines
Per-layer logistic regression (C ∈ {0.01, 0.1, 1, 10} → "3 values" is corrected to 4; say so), plus
baseline (a) mean token log-prob of the CHOSEN token over real tokens only (mask after first EOS),
baseline (b) first-token max-softmax-prob. Produce `models/probe.joblib`, `probe_meta.json`, per-layer AUROC.
Gate: best-layer val AUROC above both baselines, or the agent says plainly it is not.

## P6 — Threshold and evaluation
Calibrate threshold on val for 70% coverage; evaluate test (ID) and OOD (NQ-Open); bootstrap 95% CI;
`results/metrics.json` + ROC and reliability PNGs. Gate: realised coverage printed next to the 70% target.

## P7 — Publish
Hub model repo + card whose numbers are GENERATED from metrics.json, never typed. Gate: card text rebuilt from JSON matches.

## P8 — App
`app.py` + `requirements.txt` for ZeroGPU (Gradio 4+, model moved to cuda at module level, `@spaces.GPU(duration=...)`).
Gate: abstained branch provably never calls `generate` (CPU mock test asserts generate call count == 0).

## P9 — README audit
Rewrite README so every number traces to a key in metrics.json; state the fp16/no-bf16 deviation and any OOD degradation plainly.
Gate: automated check — each numeric claim in README maps to a metrics.json key.
