.PHONY: all prepare train merge quantize eval serve code-review clean help test dry-run lint

PYTHON ?= python3
VARIANT ?= all
PORT ?= 8000

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

serve:
	$(PYTHON) scripts/06_serve_model.py --variant $(or $(VARIANT),c) --port $(or $(PORT),8000)

code-review:
	$(PYTHON) scripts/08_code_review.py --dry-run

dry-run:
	$(PYTHON) scripts/01_prepare_datasets.py --variant all --dry-run
	$(PYTHON) scripts/02_train_qlora.py --variant c --dry-run
	$(PYTHON) scripts/03_merge_weights.py --variant c --dry-run
	$(PYTHON) scripts/05_benchmark_eval.py --variant all --dry-run
	$(PYTHON) -m py_compile scripts/06_serve_model.py
	$(PYTHON) -m py_compile scripts/08_code_review.py
	$(PYTHON) scripts/08_code_review.py --dry-run

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
	@echo '  serve      Start FastAPI inference server for specified variant'
	@echo '  code-review Run proactive code review on PR diffs using mau-llm-1.0-c'
	@echo '  dry-run    Smoke test entire pipeline without GPU'
	@echo '  test       Run pytest suite'
	@echo '  lint       Run ruff linter'
	@echo '  ui         Run Streamlit UI'
	@echo '  clean      Remove build artifacts'
