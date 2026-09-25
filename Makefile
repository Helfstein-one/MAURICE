.PHONY: all prepare train merge quantize eval publish clean help test dry-run lint

PYTHON ?= python3
VARIANT ?= all

all: prepare train merge quantize eval

prepare:
	$(PYTHON) scripts/01_prepare_datasets.py --variant $(VARIANT)

train:
	@if [ "$(VARIANT)" = "all" ]; then \
		$(PYTHON) scripts/02_train_qlora.py --variant c; \
		$(PYTHON) scripts/02_train_qlora.py --variant r; \
		$(PYTHON) scripts/02_train_qlora.py --variant g; \
	else \
		$(PYTHON) scripts/02_train_qlora.py --variant $(VARIANT); \
	fi

merge:
	@if [ "$(VARIANT)" = "all" ]; then \
		$(PYTHON) scripts/03_merge_weights.py --variant c; \
		$(PYTHON) scripts/03_merge_weights.py --variant r; \
		$(PYTHON) scripts/03_merge_weights.py --variant g; \
	else \
		$(PYTHON) scripts/03_merge_weights.py --variant $(VARIANT); \
	fi

quantize:
	@if [ "$(VARIANT)" = "all" ]; then \
		bash scripts/04_quantize_imatrix.sh c; \
		bash scripts/04_quantize_imatrix.sh r; \
		bash scripts/04_quantize_imatrix.sh g; \
	else \
		bash scripts/04_quantize_imatrix.sh $(VARIANT); \
	fi

eval:
	$(PYTHON) scripts/05_benchmark_eval.py --variant $(VARIANT)

publish:
	$(PYTHON) scripts/07_publish_hub.py --repo-id $(or $(REPO),Helfstein-one/mau-llm-1.0-$(VARIANT)) --variant $(or $(VARIANT),c)

dry-run:
	$(PYTHON) scripts/01_prepare_datasets.py --variant all --dry-run
	$(PYTHON) scripts/02_train_qlora.py --variant c --dry-run
	$(PYTHON) scripts/03_merge_weights.py --variant c --dry-run
	$(PYTHON) scripts/05_benchmark_eval.py --variant all --dry-run
	$(PYTHON) scripts/07_publish_hub.py --variant all --dry-run

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .
	ruff format --check .

ui:
	streamlit run ui/app.py

clean:
	rm -rf checkpoints/ build/ results/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name '*.pyc' -delete

help:
	@echo 'MAURICE Pipeline Makefile'
	@echo ''
	@echo 'Targets:'
	@echo '  all        Run full pipeline (prepare → train → merge → quantize → eval)'
	@echo '  prepare    Prepare datasets for all variants'
	@echo '  train      Train QLoRA adapters (c, r, g)'
	@echo '  merge      Merge adapter weights into base model'
	@echo '  quantize   Convert to GGUF and quantize with imatrix'
	@echo '  eval       Run benchmark evaluation'
	@echo '  publish    Publish model weights and GGUF to Hugging Face Hub'
	@echo '  dry-run    Smoke test entire pipeline without GPU'
	@echo '  test       Run pytest suite'
	@echo '  lint       Run ruff linter'
	@echo '  ui         Run Streamlit UI'
	@echo '  clean      Remove build artifacts'
