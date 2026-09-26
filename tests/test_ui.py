import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

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


def test_render_assistant_content_with_think_tag():
    mock_expander = MagicMock()
    mock_st_local = MagicMock()
    mock_st_local.expander.return_value.__enter__.return_value = mock_expander

    content = "<think>\nStep-by-step reasoning...\n</think>\nFinal Answer."
    with patch.object(ui_app, "st", mock_st_local):
        ui_app.render_assistant_content(content)

    mock_st_local.expander.assert_called_with("Reasoning Process")
    mock_st_local.markdown.assert_called_with("Final Answer.")


def test_render_assistant_content_plain_text():
    mock_st_local = MagicMock()
    content = "Just a direct response."
    with patch.object(ui_app, "st", mock_st_local):
        ui_app.render_assistant_content(content)

    mock_st_local.markdown.assert_called_once_with("Just a direct response.")


def test_default_benchmark_results_kpi_calculation():
    kpis = ui_app.parse_benchmark_kpis(ui_app.DEFAULT_BENCHMARK_RESULTS)
    assert kpis["model_count"] == 3
    assert kpis["max_throughput"] == 168.0
    assert kpis["min_ttft_ms"] == 14.5
    assert kpis["avg_peak_rss_mb"] == 497.33


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
    mock_st.chat_input.return_value = None
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        ui_app.render_app()


def test_render_app_handles_connection_error():
    mock_st.session_state = MockSessionState()
    mock_st.chat_input.return_value = "Test query"
    with (
        patch.dict("sys.modules", {"streamlit": mock_st}),
        patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")),
    ):
        ui_app.render_app()

    assert len(mock_st.session_state.messages) == 2
    assert mock_st.session_state.messages[0]["role"] == "user"
    assert mock_st.session_state.messages[0]["content"] == "Test query"
    assert mock_st.session_state.messages[1]["role"] == "assistant"
    assert "Error connecting to inference server: Connection refused" in mock_st.session_state.messages[1]["content"]
