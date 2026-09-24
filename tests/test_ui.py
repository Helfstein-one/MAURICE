import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

script_path = Path(__file__).parent.parent / "ui" / "app.py"


class MockSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value


mock_st = MagicMock()
mock_st.tabs.return_value = [MagicMock(), MagicMock()]
mock_st.columns.side_effect = lambda spec: [MagicMock() for _ in (spec if isinstance(spec, list) else range(spec))]
mock_st.chat_input.return_value = None
mock_st.session_state = MockSessionState()

with patch.dict("sys.modules", {"streamlit": mock_st}):
    spec = importlib.util.spec_from_file_location("ui_app", script_path)
    ui_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui_app)


def test_parse_benchmark_kpis_empty():
    kpis = ui_app.parse_benchmark_kpis([])
    assert kpis["model_count"] == 0
    assert kpis["max_throughput"] == 0.0
    assert kpis["min_ttft_ms"] == 0.0
    assert kpis["avg_peak_rss_mb"] == 0.0


def test_parse_benchmark_kpis_valid():
    sample_results = [
        {
            "variant": "mau-llm-1.0-c",
            "metrics": {
                "tokens_per_second": 100.0,
                "time_to_first_token_ms": 20.0,
                "peak_rss_mb": 400.0,
            },
        },
        {
            "variant": "mau-llm-1.0-r",
            "metrics": {
                "tokens_per_second": 200.0,
                "time_to_first_token_ms": 10.0,
                "peak_rss_mb": 600.0,
            },
        },
    ]

    kpis = ui_app.parse_benchmark_kpis(sample_results)
    assert kpis["model_count"] == 2
    assert kpis["max_throughput"] == 200.0
    assert kpis["min_ttft_ms"] == 10.0
    assert kpis["avg_peak_rss_mb"] == 500.0


def test_format_variant_table():
    sample_results = [
        {
            "variant": "mau-llm-1.0-c",
            "hardware_acceleration": "x86_64 CPU (AVX2)",
            "metrics": {
                "tokens_per_second": 100.0,
                "time_to_first_token_ms": 20.0,
                "peak_rss_mb": 400.0,
            },
        }
    ]

    table = ui_app.format_variant_table(sample_results)
    assert isinstance(table, list)
    assert len(table) == 1
    row = table[0]
    assert row["Variant"] == "mau-llm-1.0-c"
    assert row["Hardware"] == "x86_64 CPU (AVX2)"
    assert row["Throughput (t/s)"] == 100.0
    assert row["TTFT (ms)"] == 20.0
    assert row["Peak RSS (MB)"] == 400.0


def test_render_app_executes_without_error():
    mock_st.session_state = MockSessionState()
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        ui_app.render_app()
