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


def test_analyze_benchmark_results():
    empty_res = benchmark_eval.analyze_benchmark_results([])
    assert empty_res["total_models"] == 0

    sample_results = [
        {
            "variant": "mau-llm-1.0-c",
            "hardware_acceleration": "x86_64 CPU (AVX2)",
            "metrics": {
                "tokens_per_second": 100.0,
                "time_to_first_token_ms": 20.0,
                "peak_rss_mb": 500.0,
                "evaluation_scores": {"ast_syntax_pass_rate": 0.98},
            },
        },
        {
            "variant": "mau-llm-1.0-r",
            "hardware_acceleration": "x86_64 CPU (AVX2)",
            "metrics": {
                "tokens_per_second": 150.0,
                "time_to_first_token_ms": 15.0,
                "peak_rss_mb": 600.0,
                "evaluation_scores": {"gsm8k_accuracy": 0.88},
            },
        },
    ]

    analysis = benchmark_eval.analyze_benchmark_results(sample_results)
    assert analysis["total_models"] == 2
    assert analysis["best_throughput"]["variant"] == "mau-llm-1.0-r"
    assert analysis["best_throughput"]["value"] == 150.0
    assert analysis["lowest_ttft_ms"]["variant"] == "mau-llm-1.0-r"
    assert analysis["lowest_ttft_ms"]["value"] == 15.0
    assert analysis["lowest_peak_rss_mb"]["variant"] == "mau-llm-1.0-c"
    assert analysis["lowest_peak_rss_mb"]["value"] == 500.0
    assert len(analysis["variant_summaries"]) == 2


def test_main_cli_analyze_input_json(tmp_path):
    input_json = str(tmp_path / "input_bench.json")
    sample_data = [
        {
            "variant": "mau-llm-1.0-c",
            "hardware_acceleration": "AVX2",
            "metrics": {
                "tokens_per_second": 90.0,
                "time_to_first_token_ms": 25.0,
                "peak_rss_mb": 400.0,
                "evaluation_scores": {},
            },
        }
    ]
    with open(input_json, "w", encoding="utf-8") as f:
        json.dump(sample_data, f)

    with patch(
        "sys.argv",
        [
            "05_benchmark_eval.py",
            "--input-json",
            input_json,
            "--analyze",
        ],
    ):
        benchmark_eval.main()
