# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **RLAIF Synthesis Engine**: Added `maurice synth-prefs` to automatically generate preference datasets (chosen vs rejected) using LLM-as-a-Judge logic.
- **Post-SFT Alignment**: Added `maurice align` supporting Direct Preference Optimization (DPO) and Odds Ratio Preference Optimization (ORPO) via the `trl` library.
- **Streamlit Evaluation Dashboard**: New interactive UI (`ui/app.py`) for real-time benchmark visualization and `<think>` tag reasoning analysis.
- **Inference Optimization**: Integrated `vLLM` backend, `Flash Attention 2`, and `torch.compile()` for massive performance gains in PyTorch 2.x.
- **Multi-Agent Governance**: Added MCP Gates and Github Actions strict sequential DAG pipeline for CI/CD.

### Fixed
- Fixed strict type hinting warnings (`mypy`) in all modules (`scripts/validate_reasoning.py`, `maurice/synth.py`, `scripts/08_code_review.py`).
- Resolved merge conflicts and consolidated inference backends safely inside `maurice/serve.py`.
- Hardened security by blocking API credentials in `.agents/` and `.jules/` via `.gitignore`.
