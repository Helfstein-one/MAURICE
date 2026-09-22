import json

import pytest


@pytest.fixture
def tmp_variant_config(tmp_path):
    """Fixture that creates a temporary variant JSON config file and returns its path as a Path object."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "variant_c.json"
    content = {
        "variant": "c",
        "name": "mau-llm-1.0-c",
        "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "max_seq_length": 2048,
        "datasets": {"processed_file": "data/processed/train_c.jsonl"},
        "lora": {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "training": {"load_in_4bit": True},
    }
    config_file.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return config_file


@pytest.fixture
def tmp_jsonl_dataset(tmp_path):
    """Fixture that creates a temporary ChatML JSONL dataset file and returns its path as a Path object."""
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = data_dir / "train_c.jsonl"
    sample = {
        "messages": [
            {
                "role": "system",
                "content": "You are mau-llm-1.0-c, an expert code and refactoring engine.",
            },
            {"role": "user", "content": "Refactor function."},
            {
                "role": "assistant",
                "content": "<think>\nAnalyzing code...\n</think>\n```diff\n--- a/f.py\n+++ b/f.py\n```",
            },
        ]
    }
    jsonl_file.write_text(
        json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return jsonl_file
