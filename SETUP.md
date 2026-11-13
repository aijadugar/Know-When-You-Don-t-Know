# Setup Instructions

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, CPU works for smoke tests)
- Hugging Face account (for publishing to Hub)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd know-when-you-dont-know

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Environment Variables

Set these before running the pipeline:

```bash
export HF_TOKEN=your_hugging_face_token
export HF_REPO_ID=your_username/your-repo-name
```

## Running the Pipeline

### Full Pipeline (3B model)

```bash
# 1. Build data
python 01_build_data.py --n_train=3000 --n_val=500 --n_test=1000 --n_ood=1000

# 2. Generate answers and labels
python 02_generate_and_label.py

# 3. Extract activations
python 03_extract_activations.py

# 4. Train probe
python 04_train_probe.py

# 5. Evaluate
python 05_evaluate.py

# 6. Publish to Hub
python 06_push_to_hub.py

# 7. Fill README
python scripts/fill_readme.py

# 8. Audit claims
python scripts/audit_claims.py
```

### Smoke Test (0.5B model)

```bash
make smoke
```

This runs the entire pipeline on a smaller model with tiny dataset sizes for quick validation.

## Running Tests

```bash
pytest tests/ -v
```

## Deploying the Gradio App

1. Create a Hugging Face Space
2. Set hardware to ZeroGPU in Space settings
3. Push the app files:

```bash
huggingface-cli login
git lfs install
git add app.py requirements.txt README.md
git commit -m "Add Know-When-You-Don't-Know app"
git push
```

## Notes

- The app requires `HF_REPO_ID` to be set to download the probe from the Hub
- For local testing, set `APP_MOCK=1` to use a fake model
- PyTorch 2.5.0 is used (ZeroGPU docs specify 2.8.0-2.13.0 which aren't released yet)
- On T4 GPUs, float16 is used instead of bfloat16 (T4 lacks native bf16 support)
