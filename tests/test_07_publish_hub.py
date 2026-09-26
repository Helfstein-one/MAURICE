import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Dynamically import scripts/07_publish_hub.py
script_path = Path(__file__).parent.parent / "scripts" / "07_publish_hub.py"
spec = importlib.util.spec_from_file_location("publish_hub", script_path)
publish_hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish_hub)


def test_generate_model_card_yaml_frontmatter_and_variant():
    card_c = publish_hub.generate_model_card("c", benchmark_file="non_existent_file.json")
    assert "license: mit" in card_c
    assert "base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" in card_c
    assert "deepseek" in card_c
    assert "llama.cpp" in card_c
    assert "Code refactoring" in card_c
    assert "<|im_start|>system" in card_c

    card_r = publish_hub.generate_model_card("r", benchmark_file="non_existent_file.json")
    assert "mathematical reasoning" in card_r

    card_g = publish_hub.generate_model_card("g", benchmark_file="non_existent_file.json")
    assert "General-purpose instruction following" in card_g


def test_generate_model_card_with_benchmark_data(tmp_path):
    bench_data = [
        {
            "variant": "mau-llm-1.0-c",
            "hardware_acceleration": "AVX2",
            "metrics": {
                "tokens_per_second": 88.5,
                "time_to_first_token_ms": 15.2,
                "peak_rss_mb": 420.0,
                "evaluation_scores": {"humaneval_pass_at_1": 0.764},
            },
        }
    ]
    bench_file = tmp_path / "benchmark_results.json"
    bench_file.write_text(json.dumps(bench_data), encoding="utf-8")

    card = publish_hub.generate_model_card("c", benchmark_file=str(bench_file))
    assert "88.5" in card
    assert "15.2 ms" in card
    assert "420.0 MB" in card
    assert "humaneval_pass_at_1: 0.764" in card


def test_generate_model_card_missing_benchmark_fallback(tmp_path):
    bench_file = tmp_path / "missing_benchmark.json"
    card = publish_hub.generate_model_card("c", benchmark_file=str(bench_file))
    assert "Benchmark evaluation results pending or unavailable" in card


def test_publish_variant_dry_run(capsys):
    publish_hub.publish_variant(
        variant="c",
        repo_id="Helfstein-one/mau-llm-1.0-c",
        weights_dir="checkpoints/merged_c",
        gguf_dir="build",
        dry_run=True,
    )
    captured = capsys.readouterr().out
    assert "[Dry-Run] Model Card generated successfully" in captured
    assert "Simulated successful publication for repository 'Helfstein-one/mau-llm-1.0-c'" in captured


def test_publish_variant_upload_logic(tmp_path):
    weights_dir = tmp_path / "checkpoints" / "merged_c"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "config.json").write_text("{}", encoding="utf-8")
    (weights_dir / "model.safetensors").write_text("WEIGHTS", encoding="utf-8")

    gguf_dir = tmp_path / "build"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_file = gguf_dir / "mau-llm-1.0-c-q4_k_m.gguf"
    gguf_file.write_text("GGUF_BINARY", encoding="utf-8")

    mock_hf_api_class = MagicMock()
    mock_api_instance = MagicMock()
    mock_hf_api_class.return_value = mock_api_instance

    with patch("huggingface_hub.HfApi", mock_hf_api_class):
        publish_hub.publish_variant(
            variant="c",
            repo_id="Helfstein-one/mau-llm-1.0-c",
            weights_dir=str(weights_dir),
            gguf_dir=str(gguf_dir),
            token="mock_hf_token_12345",
            private=True,
            dry_run=False,
        )

    mock_hf_api_class.assert_called_once_with(token="mock_hf_token_12345")
    mock_api_instance.create_repo.assert_called_once_with(
        repo_id="Helfstein-one/mau-llm-1.0-c", repo_type="model", exist_ok=True, private=True
    )
    mock_api_instance.upload_folder.assert_called_once_with(
        folder_path=str(weights_dir),
        repo_id="Helfstein-one/mau-llm-1.0-c",
        repo_type="model",
    )
    assert mock_api_instance.upload_file.call_count == 2


def test_publish_variant_missing_token_error():
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="Hugging Face API token is required"):
        publish_hub.publish_variant(
            variant="c",
            repo_id="Helfstein-one/mau-llm-1.0-c",
            weights_dir="checkpoints/merged_c",
            gguf_dir="build",
            token=None,
            dry_run=False,
        )


def test_cli_parsing(tmp_path):
    with (
        patch(
            "sys.argv",
            [
                "07_publish_hub.py",
                "--variant",
                "c",
                "--dry-run",
                "--private",
                "--repo-id",
                "Helfstein-one/mau-llm-1.0-c",
            ],
        ),
        patch.object(publish_hub, "publish_variant") as mock_publish,
    ):
        publish_hub.main()
        mock_publish.assert_called_once_with(
            variant="c",
            repo_id="Helfstein-one/mau-llm-1.0-c",
            weights_dir="checkpoints/merged_c",
            gguf_dir="build",
            token=None,
            private=True,
            dry_run=True,
            benchmark_file="build/benchmark_results.json",
        )


def test_cli_all_variants_dry_run():
    with (
        patch(
            "sys.argv",
            [
                "07_publish_hub.py",
                "--variant",
                "all",
                "--dry-run",
            ],
        ),
        patch.object(publish_hub, "publish_variant") as mock_publish,
    ):
        publish_hub.main()
        assert mock_publish.call_count == 3
