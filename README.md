# MAURICE Framework

**Minimal Adaptation for Ultra-fast Reasoning and Inference in Code Engines**

MAURICE is an end-to-end pipeline for automated training, post-training alignment, quantization, and local runtime delivery of specialized 1.5B parameter language models based on `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

---

## Model Matrix

* **`mau-llm-1.0-c` (Code & Refactor Engine):** Syntax validation, AST consistency, unified diff patches, and structural code refactoring.
* **`mau-llm-1.0-r` (Pure Reasoning & Logic):** Step-by-step chain-of-thought logic with `<think> ... </think>` calibration.
* **`mau-llm-1.0-g` (General Purpose & Adaptive Reasoner):** Balances instruction-following with thinking suppression for direct queries.

---

## Directory Structure

```text
maurice/
├── configs/
│   ├── variant_c.json
│   ├── variant_r.json
│   └── variant_g.json
├── data/
│   ├── raw/
│   └── processed/
│       ├── train_c.jsonl
│       ├── train_r.jsonl
│       └── train_g.jsonl
├── modelfiles/
│   ├── Modelfile.c
│   ├── Modelfile.r
│   └── Modelfile.g
├── scripts/
│   ├── 01_prepare_datasets.py
│   ├── 02_train_qlora.py
│   ├── 03_merge_weights.py
│   ├── 04_quantize_imatrix.sh
│   └── 05_benchmark_eval.py
├── paper/
│   └── maurice_paper.tex
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Quickstart & Usage

### 1. Dataset Preparation
```bash
python3 scripts/01_prepare_datasets.py --variant all
```

### 2. QLoRA Training
```bash
python3 scripts/02_train_qlora.py --variant c
python3 scripts/02_train_qlora.py --variant r
python3 scripts/02_train_qlora.py --variant g
```

### 3. Weight Consolidation
```bash
python3 scripts/03_merge_weights.py --variant c
python3 scripts/03_merge_weights.py --variant r
python3 scripts/03_merge_weights.py --variant g
```

### 4. GGUF Conversion & imatrix Quantization
```bash
bash scripts/04_quantize_imatrix.sh c
bash scripts/04_quantize_imatrix.sh r
bash scripts/04_quantize_imatrix.sh g
```

### 5. Benchmark & Evaluation
```bash
python3 scripts/05_benchmark_eval.py --variant all
```

### 6. Automated Pipeline Execution via Makefile
```bash
make all
```
