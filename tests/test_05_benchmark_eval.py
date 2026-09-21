import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

# Dynamically import scripts/05_benchmark_eval.py
script_path = Path(__file__).parent.parent / "scripts" / "05_benchmark_eval.py"
spec = importlib.util.spec_from_file_location("benchmark_eval", script_path)
benchmark_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_eval)


def test_detect_hardware_accel_returns_string():
    result = benchmark_eval.detect_hardware_accel()
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_peak_rss_mb_positive():
    result = benchmark_eval.get_peak_rss_mb()
    assert isinstance(result, (int, float))
    assert result > 0.0


def test_run_benchmark_variant_all_variants():
    for v in ["c", "r", "g"]:
        res = benchmark_eval.run_benchmark_variant(v)
        assert isinstance(res, dict)
        assert res.get("variant") == f"mau-llm-1.0-{v}"
        assert "hardware_acceleration" in res
        assert "metrics" in res
        metrics = res["metrics"]
        assert "tokens_per_second" in metrics
        assert "time_to_first_token_ms" in metrics
        assert "peak_rss_mb" in metrics
        assert "evaluation_scores" in metrics


def test_main_cli(tmp_path):
    out_json = str(tmp_path / "bench_out.json")
    with patch(
        "sys.argv",
        [
            "05_benchmark_eval.py",
            "--variant",
            "all",
            "--output-json",
            out_json,
        ],
    ):
        benchmark_eval.main()

    assert Path(out_json).exists()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 3
