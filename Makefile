.PHONY: all prepare train merge quantize eval serve code-review ui clean help test dry-run lint quality-gates mcp-server validate-reasoning

MAURICE ?= maurice
PYTHON ?= python3
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

validate-reasoning:
	$(PYTHON) scripts/validate_reasoning.py --dry-run

serve:
	$(MAURICE) serve --variant $(VARIANT) --port $(PORT)

code-review:
	$(PYTHON) scripts/08_code_review.py

dry-run:
	$(MAURICE) prepare --variant all --dry-run
	$(MAURICE) train --variant c --dry-run
	$(MAURICE) merge --variant c --dry-run
	$(MAURICE) eval --variant all --dry-run
	$(PYTHON) scripts/validate_reasoning.py --dry-run
	$(PYTHON) -m py_compile scripts/06_serve_model.py
	$(PYTHON) -m py_compile scripts/07_publish_hub.py
	$(PYTHON) -m py_compile scripts/08_code_review.py
	$(PYTHON) -m py_compile scripts/08_mcp_quality_gates.py

quality-gates:
	$(PYTHON) scripts/08_mcp_quality_gates.py --run-gates

mcp-server:
	$(PYTHON) scripts/08_mcp_quality_gates.py --port $(or $(PORT),8080)

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
	@echo '  validate-reasoning Validate mau-llm-1.0-r mathematical & CoT reasoning'
	@echo '  serve      Launch FastAPI inference server'
	@echo '  code-review Run proactive code review on a PR'
	@echo '  dry-run    Smoke test entire pipeline without GPU'
	@echo '  test          Run pytest suite'
	@echo '  lint          Run ruff linter'
	@echo '  quality-gates Run quality & validation gates locally'
	@echo '  mcp-server    Start local MCP Quality Gates server'
	@echo '  ui         Run Streamlit UI'
	@echo '  clean      Remove build artifacts'
