.PHONY: all prepare train merge quantize eval clean help test

PYTHON ?= python3

all: prepare train merge quantize eval

prepare:
	$(PYTHON) scripts/01_prepare_datasets.py --variant all

train:
	$(PYTHON) scripts/02_train_qlora.py --variant c
	$(PYTHON) scripts/02_train_qlora.py --variant r
	$(PYTHON) scripts/02_train_qlora.py --variant g

merge:
	$(PYTHON) scripts/03_merge_weights.py --variant c
	$(PYTHON) scripts/03_merge_weights.py --variant r
	$(PYTHON) scripts/03_merge_weights.py --variant g

quantize:
	bash scripts/04_quantize_imatrix.sh c
	bash scripts/04_quantize_imatrix.sh r
	bash scripts/04_quantize_imatrix.sh g

eval:
	$(PYTHON) scripts/05_benchmark_eval.py --variant all

test:
	$(PYTHON) -m pytest tests/

clean:
	rm -rf data/processed/* checkpoints/* build/*
