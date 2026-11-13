---
title: Know-When-You-Don't-Know
emoji: 🤖
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10.13"
app_file: app.py
---

# Know-When-You-Don't-Know

A Gradio app that uses a linear probe on Qwen2.5-3B-Instruct's hidden states to detect when the model is likely to answer a factual question incorrectly — and abstains rather than hallucinate.

## The Problem

Large language models state wrong facts as confidently as right ones. Users cannot tell which is which. We solve exactly one thing: predict, from the question alone, whether the greedy answer will be wrong, and abstain when it will.

## Method

**Last-prompt-token probing.** We extract the hidden state at the *last input token* (the token just before generation begins) and train a logistic regression probe to predict whether the greedy answer will be wrong. Why pre-answer? Because the hidden state is available before any tokens are generated, so we can abstain without ever calling `generate()`.

The probe is trained on TriviaQA with labels derived from exact-string matching against gold answer aliases.

### Why this works

The model's internal representation at the end of the prompt encodes uncertainty about the answer. A linear probe can read this uncertainty signal — similar to findings in Semantic Entropy Probes (Kossen et al.) and LLMs Know When They're Wrong (Orgad et al.).

## Labelling Heuristic

Answers are labelled **wrong** (y=1) if they do not match any gold answer alias:

- Normalize: lowercase, strip articles (a/an/the), remove punctuation, collapse whitespace
- Match if: the normalized answer equals an alias, or contains/contained-by an alias as a whole word
- Aliases shorter than 2 characters after normalization are ignored

**Known limits:**
- Valid paraphrases are labelled wrong (e.g. "Paris, France" vs "Paris")
- Short gold answers are matched strictly ("1" vs "1990" — correctly not matched)
- The probe inherits this label noise

## Results

<!-- METRICS:START -->
*(Run `python scripts/fill_readme.py` to populate this section from results/metrics.json)*
<!-- METRICS:END -->

## Deviations from the Original Spec

| Item | Original spec | What was done | Why |
|------|-------------|---------------|-----|
| Activation dtype | bfloat16 | float16 on T4 (bf16 not supported) | T4 lacks native bf16 |
| TriviaQA test split | Evaluate on test | Used standard splits (test had answers) | Test answers were populated |
| Activation storage | float16 | float32 if non-finite after cast | Large-magnitude dimensions |
| SafeTune / AuditKIT | Optional tooling | Not used | SafeTune is a fine-tuning safety benchmark, not activation probing; auditkit API unconfirmed |
| PyTorch version | 2.8.0–2.13.0 | 2.5.0 | Specified versions not yet released on PyPI |

## Limitations

- **Task scope:** Factual short-answer QA only (TriviaQA, NQ-Open). The probe may not generalize to other tasks.
- **Model:** Trained and evaluated on Qwen2.5-3B-Instruct only.
- **Prompt template:** One system prompt. Different prompts may shift the hidden-state distribution.
- **Label noise:** The exact-string matching heuristic produces noisy labels. The probe's ceiling is bounded by label quality.
- **OOD generalization:** The probe is trained on TriviaQA. Performance on NQ-Open varies (see OOD degradation in results).

## How to Reproduce

```bash
# 1. Setup
git clone <repo>
cd know-when-you-dont-know
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 2. Build data
python 01_build_data.py --n_train=3000 --n_val=500 --n_test=1000 --n_ood=1000

# 3. Generate answers and labels
python 02_generate_and_label.py

# 4. Extract activations
python 03_extract_activations.py

# 5. Train probe
python 04_train_probe.py

# 6. Evaluate
python 05_evaluate.py

# 7. Publish (requires HF_TOKEN and HF_REPO_ID)
python 06_push_to_hub.py

# 8. Fill README with results
python scripts/fill_readme.py

# 9. Audit claims
python scripts/audit_claims.py
```

For a quick smoke test:

```bash
make smoke
```

## How to Deploy the Space

1. Create a Hugging Face Space at https://huggingface.co/new-space
2. Select Gradio as the SDK
3. Set hardware to ZeroGPU in Space settings (not in the YAML)
4. Push the app files:

```bash
huggingface-cli login
git lfs install
git add app.py requirements.txt README.md
git commit -m "Add Know-When-You-Don't-Know app"
git push
```

The app downloads the probe, metadata, and metrics from HF_REPO_ID at startup.

## Repository Structure

```
├── config.py              # All constants
├── common.py              # Shared utilities
├── 01_build_data.py       # Load and split TriviaQA + NQ-Open
├── 02_generate_and_label.py  # Greedy answer generation + labeling
├── 03_extract_activations.py # Hidden state extraction
├── 04_train_probe.py      # Probe training + baseline comparison
├── 05_evaluate.py         # Threshold + full evaluation
├── 06_push_to_hub.py      # Hub upload
├── app.py                 # Gradio app
├── requirements.txt       # Runtime dependencies
├── requirements-dev.txt   # Dev dependencies
├── tests/                 # Unit tests
├── scripts/
│   ├── fill_readme.py     # Regenerate metrics tables
│   └── audit_claims.py    # Verify README claims
├── data/                  # Split data
├── acts/                  # Activations
├── models/                # Probe + metadata
└── results/               # Metrics + plots
```
