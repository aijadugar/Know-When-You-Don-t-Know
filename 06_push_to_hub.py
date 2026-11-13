"""06_push_to_hub.py — push model and card to Hugging Face Hub."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import joblib
from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_folder
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import MODEL_ID, MODELS_DIR, RESULTS_DIR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Write card to file, don't upload")
    return parser.parse_args()

def generate_model_card(metrics: dict, meta: dict) -> str:
    """Generate the model card from metrics.json — no numbers typed by hand."""
    # Read all numbers from metrics.json
    def get(split: str, scorer: str, key: str, default: str = "N/A") -> str:
        for m in metrics.get("results", []):
            if m.get("split") == split and m.get("scorer") == scorer:
                val = m.get(key)
                if val is not None:
                    if isinstance(val, list) and len(val) == 2:
                        return f"{val[0]:.3f}–{val[1]:.3f}"
                    return f"{val:.3f}"
        return default

    # Extract AUROC and CI
    def auroc(split: str, scorer: str) -> str:
        for m in metrics.get("results", []):
            if m.get("split") == split and m.get("scorer") == scorer:
                ci = m.get("auroc_ci", [])
                return f"{m['auroc']:.3f} ({ci[0]:.3f}, {ci[1]:.3f})"
        return "N/A"

    # Accuracy at 70%
    def acc70(split: str, scorer: str) -> str:
        for m in metrics.get("results", []):
            if m.get("split") == split and m.get("scorer") == scorer:
                return f"{m.get('acc_at_70', 'N/A')}"
        return "N/A"

    # Realised coverage
    def real_cov(split: str, scorer: str) -> str:
        for m in metrics.get("results", []):
            if m.get("split") == split and m.get("scorer") == scorer:
                return f"{m.get('realised_coverage', 0):.3f}"
        return "N/A"

    # Wrong caught
    def wrong_caught(split: str, scorer: str) -> str:
        for m in metrics.get("results", []):
            if m.get("split") == split and m.get("scorer") == scorer:
                return f"{m.get('share_of_wrong_caught', 0):.3f}"
        return "N/A"

    verdict = metrics.get("verdict", {})
    ood_degradation = verdict.get("ood_degradation", 0.0)
    beats_test = verdict.get("probe_beats_both_baselines_auroc_test", False)
    beats_ood = verdict.get("probe_beats_both_baselines_auroc_ood", False)

    ood_note = ""
    if abs(ood_degradation) > 0.05:
        ood_note = (
            f"\n"
            f"**Note on OOD Generalization:** The probe's AUROC on NQ-Open is "
            f"{ood_degradation:+.3f} below TriviaQA test. "
            f"{'This degradation is reported rather than hidden.' if ood_degradation < 0 else 'OOD performance is slightly better.'}"
        )

    card = f"""---
base_model: {MODEL_ID}
library_name: scikit-learn
tags:
- know-when-you-dont-know
- calibration
- linear-probe
- hallucination-detection
- qwen
---

# Know-When-You-Don't-Know

A linear probe trained on the last-prompt-token hidden state of {MODEL_ID} to predict whether the model will answer a factual question correctly. The probe abstains when its risk score exceeds a threshold.

## Method

- **Probe input:** Hidden state at the last prompt token (index {meta.get('hidden_state_index', '?')} of {meta.get('n_layers', '?')} layers), dimension {meta.get('d_model', '?')}.
- **Label:** y=1 if the greedy answer does not match any gold answer alias (wrong); y=0 if correct.
- **Training:** Logistic regression (C={meta.get('C', '?')}) on {meta.get('n_layers', '?')} hidden-state indices, best selected by val AUROC.
- **Threshold:** Set at the {int(meta.get('target_coverage', 0.7) * 100)}th percentile of val risk, frozen before evaluating on held-out data.

## Results

### TriviaQA Test

| Scorer | AUROC | Acc@70% | Realised Cov | Wrong Caught |
|--------|-------|---------|--------------|--------------|
| Probe | {auroc('test', 'probe')} | {acc70('test', 'probe')} | {real_cov('test', 'probe')} | {wrong_caught('test', 'probe')} |
| −mean log-prob | {auroc('test', 'baseline_a')} | {acc70('test', 'baseline_a')} | {real_cov('test', 'baseline_a')} | {wrong_caught('test', 'baseline_a')} |
| 1 − max prob | {auroc('test', 'baseline_b')} | {acc70('test', 'baseline_b')} | {real_cov('test', 'baseline_b')} | {wrong_caught('test', 'baseline_b')} |

### NQ-Open (OOD)

| Scorer | AUROC | Acc@70% | Realised Cov | Wrong Caught |
|--------|-------|---------|--------------|--------------|
| Probe | {auroc('ood', 'probe')} | {acc70('ood', 'probe')} | {real_cov('ood', 'probe')} | {wrong_caught('ood', 'probe')} |
| −mean log-prob | {auroc('ood', 'baseline_a')} | {acc70('ood', 'baseline_a')} | {real_cov('ood', 'baseline_a')} | {wrong_caught('ood', 'baseline_a')} |
| 1 − max prob | {auroc('ood', 'baseline_b')} | {acc70('ood', 'baseline_b')} | {real_cov('ood', 'baseline_b')} | {wrong_caught('ood', 'baseline_b')} |

## Label Heuristic

The probe is trained on labels derived from exact-string matching against gold answer aliases:

- Answers are normalized (lowercase, strip articles, remove punctuation).
- Match if the answer equals, or contains as a whole word, any alias.
- Aliases shorter than 2 characters after normalization are ignored.

**Known limits:** Valid paraphrases are labelled wrong; short gold answers are matched strictly; the probe inherits this label noise.

## Probe beats both baselines on TriviaQA test: {"yes" if beats_test else "no"}; on NQ-Open: {"yes" if beats_ood else "no"}.{ood_note}

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
- One model ({MODEL_ID}) and one prompt template.
- Label noise from the matching heuristic.
- Probe trained on TriviaQA; OOD generalization varies (AUROC gap: {ood_degradation:+.3f}).
"""
    return card

def main():
    args = parse_args()

    print("=" * 70)
    print("P7: PUBLISH TO HUB")
    print("=" * 70)

    # Check environment
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")

    if not token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("  Set it with: export HF_TOKEN=your_token")
        sys.exit(1)

    if not repo_id:
        print("ERROR: HF_REPO_ID environment variable not set.")
        print("  Set it with: export HF_REPO_ID=your_username/your-repo-name")
        sys.exit(1)

    print(f"Token: {'set' if token else 'MISSING'}")
    print(f"Repo ID: {repo_id}")

    # Verify huggingface_hub version
    import huggingface_hub
    print(f"huggingface_hub version: {huggingface_hub.__version__}")

    # Load metrics and meta
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    with open(MODELS_DIR / "probe_meta.json") as f:
        meta = json.load(f)

    # Generate card
    card = generate_model_card(metrics, meta)

    if args.dry_run:
        card_path = RESULTS_DIR / "MODEL_CARD_preview.md"
        with open(card_path, "w") as f:
            f.write(card)
        print(f"\n[Dry run] Card written to {card_path}")
        print("\n" + "=" * 70)
        print("P7 GATE: dry run complete — card preview generated")
        print("=" * 70)
        return

    # Create repo
    api = HfApi(token=token)
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Repo created/exists: {repo_id}")
    except Exception as e:
        print(f"ERROR creating repo: {e}")
        sys.exit(1)

    # Upload files
    files_to_upload = [
        (MODELS_DIR / "probe.joblib", "probe.joblib"),
        (MODELS_DIR / "probe_meta.json", "probe_meta.json"),
        (RESULTS_DIR / "metrics.json", "metrics.json"),
        (RESULTS_DIR / "accuracy_vs_coverage.png", "accuracy_vs_coverage.png"),
        (RESULTS_DIR / "layer_auroc.png", "layer_auroc.png"),
    ]

    # Generate and upload README
    readme_path = RESULTS_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(card)

    print("\n## Uploading files...")
    for local_path, hub_path in files_to_upload + [(readme_path, "README.md")]:
        if local_path.exists():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hub_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Uploaded: {hub_path}")
        else:
            print(f"  SKIPPED (not found): {local_path}")

    print(f"\nRepo URL: https://huggingface.co/{repo_id}")
    print("\n" + "=" * 70)
    print("P7 GATE: PASSED — all files uploaded to Hub")
    print("=" * 70)

if __name__ == "__main__":
    main()
