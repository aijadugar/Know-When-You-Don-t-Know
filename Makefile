.PHONY: smoke test clean

# Run the full pipeline on the smoke model
smoke:
	python -m pytest tests/ -v
	python 01_build_data.py --smoke --n_train=20 --n_val=10 --n_test=20 --n_ood=10
	python 02_generate_and_label.py --smoke
	python 03_extract_activations.py --smoke
	python 04_train_probe.py --smoke
	python 05_evaluate.py --smoke
	python 06_push_to_hub.py --dry_run

test:
	python -m pytest tests/ -v

clean:
	rm -rf data/*.jsonl acts/*.npy models/*.joblib models/*.json results/*.json results/*.png __pycache__/ */__pycache__/ */*/__pycache__
