import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Dynamically import scripts/02_train_qlora.py
script_path = Path(__file__).parent.parent / "scripts" / "02_train_qlora.py"
spec = importlib.util.spec_from_file_location("train_qlora", script_path)
train_qlora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_qlora)


def test_load_variant_config_c():
    config = train_qlora.load_variant_config("c")
    assert isinstance(config, dict)
    assert config.get("variant") == "c"
    assert "lora" in config
    assert config["lora"]["r"] == 16


def test_load_variant_config_not_found():
    with pytest.raises(FileNotFoundError):
        train_qlora.load_variant_config("non_existent_variant_xyz")


def test_run_training_dry_run(tmp_path):
    output_dir = str(tmp_path / "checkpoints" / "adapter_c")
    train_qlora.run_training(
        variant="c",
        dry_run=True,
        output_dir=output_dir,
    )

    adapter_config = Path(output_dir) / "adapter_config.json"
    adapter_model = Path(output_dir) / "adapter_model.bin"

    assert adapter_config.exists()
    assert adapter_model.exists()

    with open(adapter_config, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data.get("variant") == "c"
        assert data.get("status") == "trained_successfully"


def test_run_training_custom_config_path(tmp_path):
    cfg_file = tmp_path / "custom_config.json"
    cfg_data = {
        "variant": "c",
        "name": "custom-mau-llm",
        "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "max_seq_length": 1024,
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj"],
        },
        "training": {"load_in_4bit": True},
    }
    cfg_file.write_text(json.dumps(cfg_data), encoding="utf-8")

    out_dir = str(tmp_path / "out_custom")
    train_qlora.run_training(
        variant="c",
        config_path=str(cfg_file),
        dry_run=True,
        output_dir=out_dir,
    )

    assert (Path(out_dir) / "adapter_config.json").exists()


def test_run_training_fallback_on_exception(tmp_path):
    out_dir = str(tmp_path / "out_fallback")
    # Calling dry_run=False without GPU setup should catch exception and trigger fallback
    train_qlora.run_training(
        variant="c",
        dry_run=False,
        output_dir=out_dir,
    )

    adapter_config = Path(out_dir) / "adapter_config.json"
    assert adapter_config.exists()
    with open(adapter_config, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data.get("status") in ["trained_successfully", "trained_fallback"]


def test_main_cli(tmp_path):
    out_dir = str(tmp_path / "cli_out")
    with patch(
        "sys.argv",
        [
            "02_train_qlora.py",
            "--variant",
            "c",
            "--dry-run",
            "--output-dir",
            out_dir,
        ],
    ):
        train_qlora.main()
        assert (Path(out_dir) / "adapter_config.json").exists()


def test_attn_implementation_selection(tmp_path):
    out_dir = str(tmp_path / "out_attn")
    mock_spec = type("ModuleSpec", (), {})()

    mock_torch = type("TorchMock", (), {"cuda": type("CudaMock", (), {"is_available": lambda *args, **kwargs: False})()})()
    mock_transformers = type("TransformersMock", (), {})()
    mock_autotokenizer = type("AutoTokenizerMock", (), {"from_pretrained": MagicMock()})()
    mock_automodel = type("AutoModelForCausalLMMock", (), {"from_pretrained": MagicMock()})()
    mock_peft = type("PeftMock", (), {"LoraConfig": MagicMock(), "get_peft_model": MagicMock()})()
    mock_datasets = type("DatasetsMock", (), {"load_dataset": MagicMock(side_effect=Exception("skip training loop"))})()

    mock_transformers.AutoTokenizer = mock_autotokenizer
    mock_transformers.AutoModelForCausalLM = mock_automodel

    modules_dict = {
        "unsloth": None,
        "torch": mock_torch,
        "transformers": mock_transformers,
        "peft": mock_peft,
        "datasets": mock_datasets,
    }

    # Test when flash_attn is available
    with (
        patch.dict("sys.modules", modules_dict),
        patch("importlib.util.find_spec", return_value=mock_spec),
    ):
        train_qlora.run_training(
            variant="c",
            dry_run=False,
            output_dir=out_dir,
        )
        mock_automodel.from_pretrained.assert_called_once()
        _, kwargs = mock_automodel.from_pretrained.call_args
        assert kwargs.get("attn_implementation") == "flash_attention_2"

    # Test fallback to sdpa when flash_attn is not available
    mock_automodel.from_pretrained.reset_mock()
    out_dir_sdpa = str(tmp_path / "out_sdpa")
    with (
        patch.dict("sys.modules", modules_dict),
        patch("importlib.util.find_spec", return_value=None),
    ):
        train_qlora.run_training(
            variant="c",
            dry_run=False,
            output_dir=out_dir_sdpa,
        )
        mock_automodel.from_pretrained.assert_called_once()
        _, kwargs = mock_automodel.from_pretrained.call_args
        assert kwargs.get("attn_implementation") == "sdpa"
