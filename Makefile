# Know-when-you-don't-know -- pipeline driver.
# Real-run targets assume one Colab T4 (16 GB).  Every stage also has a --smoke variant
# that runs Qwen/Qwen2.5-0.5B-Instruct on CPU into *_smoke/ directories.
PYTHON ?= python

.PHONY: help install test smoke_data smoke_gen smoke_acts smoke_all data generate acts \
        probe eval card app clean

help:
	@echo "make install     - pip install -r requirements-dev.txt"
	@echo "make test        - pytest -q                       (P1 gate)"
	@echo "make smoke_all   - 0.5B/CPU end-to-end into *_smoke/ dirs"
	@echo "make data        - 01_build_data.py       -> data/{train,val,test,ood}.jsonl      (P2)"
	@echo "make generate    - 02_generate_and_label  -> data/*_labeled.jsonl                 (P3)"
	@echo "make acts        - 03_extract_activations -> acts/{split}.npy                     (P4)"
	@echo "make probe       - 04_train_probe.py      -> models/probe.joblib                  (P5)"
	@echo "make eval        - 05_evaluate.py         -> results/metrics.json                 (P6)"
	@echo "make card        - 06_push_to_hub.py      -> Hub upload                           (P7)"
	@echo "make app         - run the Gradio app locally (needs models/ + results/)"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

# ---------------- smoke (CPU, Qwen2.5-0.5B-Instruct, tiny sizes) ----------------
smoke_data:
	$(PYTHON) 01_build_data.py --smoke

smoke_gen:
	$(PYTHON) 02_generate_and_label.py --smoke

smoke_acts:
	$(PYTHON) 03_extract_activations.py --smoke

smoke_all: smoke_data smoke_gen smoke_acts

# ---------------- real run (T4) ----------------
data:
	$(PYTHON) 01_build_data.py

generate:
	$(PYTHON) 02_generate_and_label.py

acts:
	$(PYTHON) 03_extract_activations.py

probe:
	@test -f 04_train_probe.py || (echo "04_train_probe.py arrives in a later prompt" && exit 1)
	$(PYTHON) 04_train_probe.py

eval:
	@test -f 05_evaluate.py || (echo "05_evaluate.py arrives in a later prompt" && exit 1)
	$(PYTHON) 05_evaluate.py

card:
	@test -f 06_push_to_hub.py || (echo "06_push_to_hub.py arrives in a later prompt" && exit 1)
	$(PYTHON) 06_push_to_hub.py

app:
	$(PYTHON) app.py

clean:
	rm -rf data_smoke acts_smoke models_smoke results_smoke .pytest_cache __pycache__ tests/__pycache__
