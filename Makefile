.PHONY: all prepare train merge quantize eval serve ui clean help test dry-run lint

MAURICE ?= maurice
PORT ?= 8000
VARIANT ?= c

all: prepare train merge quantize eval

prepare:
	$(MAURICE) prepare --variant $(VARIANT)

train:
	@if [ "$(VARIANT)" = "all" ]; then \
		$(MAURICE) train --variant c; \
		$(MAURICE) train --variant r; \
		$(MAURICE) train --variant g; \
	else \
		$(MAURICE) train --variant $(VARIANT); \
	fi

merge:
	@if [ "$(VARIANT)" = "all" ]; then \
		$(MAURICE) merge --variant c; \
		$(MAURICE) merge --variant r; \
		$(MAURICE) merge --variant g; \
	else \
		$(MAURICE) merge --variant $(VARIANT); \
	fi

quantize:
	@if [ "$(VARIANT)" = "all" ]; then \
		$(MAURICE) quantize --variant c; \
		$(MAURICE) quantize --variant r; \
		$(MAURICE) quantize --variant g; \
	else \
		$(MAURICE) quantize --variant $(VARIANT); \
	fi

eval:
	$(MAURICE) eval --variant $(VARIANT)

serve:
	$(MAURICE) serve --variant $(VARIANT) --port $(PORT)

dry-run:
	$(MAURICE) prepare --variant all --dry-run
	$(MAURICE) train --variant c --dry-run
	$(MAURICE) merge --variant c --dry-run
	$(MAURICE) eval --variant all --dry-run

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .
	ruff format --check .

ui:
	$(MAURICE) ui

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
	@echo '  serve      Launch FastAPI inference server'
	@echo '  dry-run    Smoke test entire pipeline without GPU'
	@echo '  test       Run pytest suite'
	@echo '  lint       Run ruff linter'
	@echo '  ui         Run Streamlit UI'
	@echo '  clean      Remove build artifacts'
