# Operational Guidelines for Jules AI — Project MAURICE

## Project Architecture
MAURICE delivers three specialized variants based on `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`:
- `mau-llm-1.0-c`: Code refactoring and syntax-grounded diffs.
- `mau-llm-1.0-r`: Chain-of-thought mathematical and deductive reasoning (`<think>`).
- `mau-llm-1.0-g`: General-purpose instruction following with adaptive thinking.

## Quality & Validation Gates
Before opening or updating a Pull Request, run and ensure zero failures:
1. `ruff check --fix .`
2. `ruff format .`
3. `pytest tests/ -v -m "not slow"`
4. `python3 -m py_compile scripts/*.py`

## Binary Runtime Rule
Never build binaries against musl (Alpine) if the runtime image is based on glibc (Debian/Ubuntu). Ensure both stages use glibc-compatible bases.
