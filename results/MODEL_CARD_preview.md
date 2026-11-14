---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: scikit-learn
tags:
- know-when-you-dont-know
- calibration
- linear-probe
- hallucination-detection
- qwen
---

# Know-When-You-Don't-Know

A linear probe trained on the last-prompt-token hidden state of Qwen/Qwen2.5-3B-Instruct to predict whether the model will answer a factual question correctly. The probe abstains when its risk score exceeds a threshold.

## Method

- **Probe input:** Hidden state at the last prompt token (index 21 of 25 layers), dimension 896.
- **Label:** y=1 if the greedy answer does not match any gold answer alias (wrong); y=0 if correct.
- **Training:** Logistic regression (C=0.01) on 25 hidden-state indices, best selected by val AUROC.
- **Threshold:** Set at the 70th percentile of val risk, frozen before evaluating on held-out data.

## Results

### TriviaQA Test

| Scorer | AUROC | Acc@70% | Realised Cov | Wrong Caught |
|--------|-------|---------|--------------|--------------|
| Probe | 0.403 (0.247, 0.574) | 0.12857142857142856 | 0.330 | 0.655 |
| −mean log-prob | 0.807 (0.680, 0.910) | 0.2142857142857143 | 0.300 | 0.774 |
| 1 − max prob | 0.807 (0.698, 0.899) | 0.22857142857142854 | 0.220 | 0.845 |

### NQ-Open (OOD)

| Scorer | AUROC | Acc@70% | Realised Cov | Wrong Caught |
|--------|-------|---------|--------------|--------------|
| Probe | 0.544 (0.296, 0.765) | 0.042857142857142816 | 0.490 | 0.515 |
| −mean log-prob | 0.711 (0.146, 1.000) | 0.02857142857142858 | 0.210 | 0.804 |
| 1 − max prob | 0.699 (0.330, 1.000) | 0.042857142857142816 | 0.190 | 0.814 |

## Label Heuristic

The probe is trained on labels derived from exact-string matching against gold answer aliases:

- Answers are normalized (lowercase, strip articles, remove punctuation).
- Match if the answer equals, or contains as a whole word, any alias.
- Aliases shorter than 2 characters after normalization are ignored.

**Known limits:** Valid paraphrases are labelled wrong; short gold answers are matched strictly; the probe inherits this label noise.

## Probe beats both baselines on TriviaQA test: no; on NQ-Open: no.
**Note on OOD Generalization:** The probe's AUROC on NQ-Open is +0.140 below TriviaQA test. OOD performance is slightly better.

## Usage

```python
from huggingface_hub import hf_hub_download
import joblib

# Download probe
probe = joblib.load(hf_hub_download(repo_id="YOUR_REPO_ID", filename="probe.joblib"))
meta = joblib.load(hf_hub_download(repo_id="YOUR_REPO_ID", filename="probe_meta.json"))

# Get hidden state at last prompt token, then:
risk = probe.predict_proba(hidden_state.reshape(1, -1))[:, 1]
if risk > meta["threshold"]:
    return "I'm not confident I know this"
else:
    # generate answer
```

## Limitations

- Trained and evaluated on factual short-answer QA only (TriviaQA, NQ-Open).
- One model (Qwen/Qwen2.5-3B-Instruct) and one prompt template.
- Label noise from the matching heuristic.
- Probe trained on TriviaQA; OOD generalization varies (AUROC gap: +0.140).
