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

The dataset preparation pipeline downloads public HuggingFace datasets, applies variant-specific quality filters, and formats records into standard ChatML JSONL schema with `<think>` tags.

#### Sources and Quality Filters

* **Variant `c` (Code):** `iamtarun/python_code_instructions_18k_alpaca` — Validates Python code AST syntax with `validate_code_syntax()`.
* **Variant `r` (Reasoning):** `openai/gsm8k` — Filters multi-step reasoning chains (`len(steps) >= 2`) with calibrated `<think> ... </think>` tags.
* **Variant `g` (General):** `HuggingFaceH4/ultrachat_200k` — Excludes short responses (`len(response) < 50`) and injects minimal `<think>\n</think>` tags.

#### Execution Commands

To download and process datasets for all variants:

```bash
python3 scripts/01_prepare_datasets.py --variant all
```

To prepare a specific variant (e.g., reasoning):

```bash
python3 scripts/01_prepare_datasets.py --variant r --sample-size 1000
```

Raw cached datasets are saved to `data/raw/` (`python_code_instructions.jsonl`, `gsm8k_train.jsonl`, `ultrachat_sample.jsonl`) and formatted datasets are saved to `data/processed/` (`train_c.jsonl`, `train_r.jsonl`, `train_g.jsonl`).

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
