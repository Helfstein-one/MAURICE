import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

# Dynamically import scripts/03_merge_weights.py
script_path = Path(__file__).parent.parent / "scripts" / "03_merge_weights.py"
spec = importlib.util.spec_from_file_location("merge_weights", script_path)
merge_weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_weights)


def test_merge_weights_dry_run(tmp_path):
    output_dir = str(tmp_path / "checkpoints" / "merged_c")
    merge_weights.merge_weights(
        variant="c",
        output_dir=output_dir,
        dry_run=True,
    )

    config_file = Path(output_dir) / "config.json"
    weights_file = Path(output_dir) / "model.safetensors"

    assert config_file.exists()
    assert weights_file.exists()

    with open(config_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert meta.get("variant") == "c"
        assert meta.get("architecture") == "DeepSeekR1ForCausalLM"
        assert meta.get("status") == "merged_successfully"


def test_merge_weights_output_dir_created(tmp_path):
    output_dir = str(tmp_path / "custom_merged_output")
    assert not Path(output_dir).exists()

    merge_weights.merge_weights(
        variant="c",
        output_dir=output_dir,
        dry_run=True,
    )

    assert Path(output_dir).exists()
    assert Path(output_dir).is_dir()


def test_merge_weights_fallback_when_adapter_missing(tmp_path):
    adapter_path = str(tmp_path / "non_existent_adapter")
    output_dir = str(tmp_path / "merged_fallback")

    merge_weights.merge_weights(
        variant="r",
        adapter_path=adapter_path,
        output_dir=output_dir,
        dry_run=False,
    )

    config_file = Path(output_dir) / "config.json"
    assert config_file.exists()


def test_merge_weights_fallback_on_exception(tmp_path):
    adapter_dir = tmp_path / "adapter_c"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg = adapter_dir / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model": "invalid-model-name"}), encoding="utf-8")

    output_dir = str(tmp_path / "merged_exc_fallback")

    # dry_run=False with mock adapter config will encounter exception and trigger fallback
    merge_weights.merge_weights(
        variant="c",
        adapter_path=str(adapter_dir),
        output_dir=output_dir,
        dry_run=False,
    )

    config_file = Path(output_dir) / "config.json"
    assert config_file.exists()
    with open(config_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert meta.get("status") in ["merged_successfully", "merged_fallback"]


def test_merge_weights_save_methods(tmp_path):
    for method in ["merged_16bit", "lora"]:
        output_dir = str(tmp_path / f"merged_{method}")
        merge_weights.merge_weights(
            variant="c",
            save_method=method,
            output_dir=output_dir,
            dry_run=True,
        )
        assert (Path(output_dir) / "config.json").exists()


def test_main_cli(tmp_path):
    output_dir = str(tmp_path / "cli_merged")
    with patch(
        "sys.argv",
        [
            "03_merge_weights.py",
            "--variant",
            "g",
            "--dry-run",
            "--output-dir",
            output_dir,
        ],
    ):
        merge_weights.main()
        assert (Path(output_dir) / "config.json").exists()
