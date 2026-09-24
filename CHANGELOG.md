# Changelog

All notable changes to the MAURICE project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-23

### Added
- **Model Variant Framework**: Support for three fine-tuned model variants based on `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`:
  - `mau-llm-1.0-c`: Specialized for code refactoring, AST consistency, syntax validation, and unified diff generation.
  - `mau-llm-1.0-r`: Specialized for step-by-step chain-of-thought mathematical and logical reasoning with calibrated `<think>` tags.
  - `mau-llm-1.0-g`: General-purpose instruction following with adaptive thinking suppression.
- **Dataset Preparation Pipeline (`scripts/01_prepare_datasets.py`)**:
  - Processing Hugging Face datasets (`bigcode/the-stack-smol-xs`, `HuggingFaceH4/Bespoke-Stratos-17k`, `teknium/OpenHermes-2.5`).
  - AST syntax validation for Python, balanced brace checks for C/JS/TS.
  - ChatML JSONL formatting and synthetic fallback generation.
- **QLoRA Fine-Tuning Pipeline (`scripts/02_train_qlora.py`)**:
  - 4-bit NormalFloat (NF4) quantization via `bitsandbytes`.
  - Configurable LoRA parameters (rank `r=16`, `lora_alpha=16`) targeting query, key, value, output, and MLP projection modules.
- **Weight Consolidation (`scripts/03_merge_weights.py`)**:
  - Seamless merge of QLoRA adapters with base 16-bit model weights.
- **GGUF & Importance Matrix Quantization (`scripts/04_quantize_imatrix.sh`)**:
  - Integration with `llama.cpp` for GGUF conversion and `imatrix` (importance matrix) calibration.
- **Benchmark & Evaluation Suite (`scripts/05_benchmark_eval.py`)**:
  - Memory consumption (RSS) monitoring and throughput measurement (tokens/sec).
- **Inference Server (`scripts/06_serve_model.py`)**:
  - OpenAI-compatible REST endpoint (`/v1/chat/completions`) using FastAPI and StreamingResponse support.
- **User Interface (`ui/app.py`)**:
  - Streamlit web application with real-time parsing and formatting of `<think>` reasoning blocks.
- **Container Infrastructure (`Containerfile`)**:
  - Multi-stage glibc-compatible runtime (`debian:bookworm-slim` builder and `python:3.11-slim-bookworm` runtime) preventing glibc/musl C-runtime mismatch.
- **CI/CD DAG Pipeline (`.github/workflows/ci.yml`)**:
  - 6-layer Directed Acyclic Graph topology with path filtering, parallel fast-checks (Ruff), static typing (mypy), test matrix (pytest), and Podman container smoke tests.
- **Comprehensive Documentation**:
  - Enhanced README with Mermaid architecture diagrams and detailed Model Matrix table.
