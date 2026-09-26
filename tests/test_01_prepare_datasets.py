import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import maurice.prepare

# Dynamically import scripts/01_prepare_datasets.py
script_path = Path(__file__).parent.parent / "scripts" / "01_prepare_datasets.py"
spec = importlib.util.spec_from_file_location("prepare_datasets", script_path)
prepare_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_datasets)


def test_validate_code_syntax_python_valid():
    valid_py = "def hello():\n    return 'world'"
    assert prepare_datasets.validate_code_syntax(valid_py, "python") is True
    assert prepare_datasets.validate_code_syntax(valid_py, "py") is True


def test_validate_code_syntax_python_invalid():
    invalid_py = "def hello(:\n    return 'world'"
    assert prepare_datasets.validate_code_syntax(invalid_py, "python") is False


def test_validate_code_syntax_c_balanced_braces():
    valid_c = "int main() { if (1) { return 0; } }"
    unbalanced_braces = "int main() { if (1) { return 0; }"
    unbalanced_parens = "int main() { if (1 { return 0; } }"
    negative_brace = "int main() } {"

    assert prepare_datasets.validate_code_syntax(valid_c, "c") is True
    assert prepare_datasets.validate_code_syntax(unbalanced_braces, "c") is False
    assert prepare_datasets.validate_code_syntax(unbalanced_parens, "c") is False
    assert prepare_datasets.validate_code_syntax(negative_brace, "c") is False

    # Edge cases
    assert prepare_datasets.validate_code_syntax("", "python") is False
    assert prepare_datasets.validate_code_syntax(None, "python") is False
    assert prepare_datasets.validate_code_syntax("some text", "unknown_lang") is True


def test_validate_think_tags_valid():
    valid_response = "<think>\nThinking steps here...\n</think>\nFinal answer."
    assert prepare_datasets.validate_think_tags(valid_response) is True


def test_validate_code_in_assistant_content():
    valid_content = "<think>\nTesting code\n</think>\n```python\ndef foo():\n    return 42\n```"
    invalid_content = "<think>\nTesting code\n</think>\n```python\ndef foo(:\n    return 42\n```"
    no_code = "Just plain text response."

    assert prepare_datasets.validate_code_in_assistant_content(valid_content) is True
    assert prepare_datasets.validate_code_in_assistant_content(invalid_content) is False
    assert prepare_datasets.validate_code_in_assistant_content(no_code) is True


def test_validate_think_tags_missing():
    no_start = "Thinking steps...\n</think>\nFinal answer."
    no_end = "<think>\nThinking steps...\nFinal answer."
    reversed_tags = "</think>\n<think>Reversed"
    empty = ""
    none_val = None

    assert prepare_datasets.validate_think_tags(no_start) is False
    assert prepare_datasets.validate_think_tags(no_end) is False
    assert prepare_datasets.validate_think_tags(reversed_tags) is False
    assert prepare_datasets.validate_think_tags(empty) is False
    assert prepare_datasets.validate_think_tags(none_val) is False


def test_format_chatml_example():
    sys_p = "System prompt"
    user_p = "  User question  "
    asst_r = "  Assistant answer  "

    chatml = prepare_datasets.format_chatml_example(sys_p, user_p, asst_r)

    assert "messages" in chatml
    assert len(chatml["messages"]) == 3
    assert chatml["messages"][0] == {"role": "system", "content": "System prompt"}
    assert chatml["messages"][1] == {"role": "user", "content": "User question"}
    assert chatml["messages"][2] == {"role": "assistant", "content": "Assistant answer"}


def test_generate_synthetic_samples_variant_c():
    samples = prepare_datasets.generate_synthetic_samples("c", count=2)
    assert len(samples) == 2
    for sample in samples:
        asst_msg = sample["messages"][2]["content"]
        assert "```diff" in asst_msg
        assert prepare_datasets.validate_think_tags(asst_msg) is True


def test_generate_synthetic_samples_variant_r():
    samples = prepare_datasets.generate_synthetic_samples("r", count=2)
    assert len(samples) == 2
    for sample in samples:
        asst_msg = sample["messages"][2]["content"]
        assert "<think>" in asst_msg
        assert "</think>" in asst_msg
        assert prepare_datasets.validate_think_tags(asst_msg) is True


def test_generate_synthetic_samples_variant_g():
    samples = prepare_datasets.generate_synthetic_samples("g", count=2)
    assert len(samples) == 2
    for sample in samples:
        asst_msg = sample["messages"][2]["content"]
        assert asst_msg.startswith("<think>\n</think>")


def test_process_variant(tmp_path):
    out_file = str(tmp_path / "train_c.jsonl")
    count = prepare_datasets.process_variant("c", out_file, sample_size=3, synthetic=True)
    assert count == 3
    assert Path(out_file).exists()

    lines = Path(out_file).read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    data = json.loads(lines[0])
    assert "messages" in data


def test_process_hf_dataset_mocked():
    mock_ds_c = [{"content": "def add(a, b):\n    return a + b"}]
    mock_ds_r = [
        {
            "conversations": [
                {"role": "user", "value": "What is 2+2?"},
                {"role": "assistant", "value": "<think>\n2+2=4\n</think>\n4"},
            ]
        }
    ]
    mock_ds_g = [{"instruction": "Say hi", "input": "", "output": "Hello!"}]

    with patch("datasets.load_dataset") as mock_load:
        mock_load.return_value = mock_ds_c
        recs_c = prepare_datasets.process_hf_dataset("c", sample_size=1)
        assert len(recs_c) == 1
        assert "def add" in recs_c[0]["messages"][2]["content"]

        mock_load.return_value = mock_ds_r
        recs_r = prepare_datasets.process_hf_dataset("r", sample_size=1)
        assert len(recs_r) == 1
        assert "2+2=4" in recs_r[0]["messages"][2]["content"]

        mock_load.return_value = mock_ds_g
        recs_g = prepare_datasets.process_hf_dataset("g", sample_size=1)
        assert len(recs_g) == 1
        assert "<think>" in recs_g[0]["messages"][2]["content"]


def test_process_variant_real_hf_fallback(tmp_path):
    out_file = str(tmp_path / "train_fallback.jsonl")
    # Request synthetic=False to test HF exception fallback handling without network
    count = prepare_datasets.process_variant("c", out_file, sample_size=2, synthetic=False)
    assert count > 0
    assert Path(out_file).exists()


def test_main_cli():
    with (
        patch(
            "sys.argv",
            ["01_prepare_datasets.py", "--variant", "c", "--sample-size", "2"],
        ),
        patch.object(maurice.prepare, "process_variant", return_value=2) as mock_proc,
    ):
        prepare_datasets.main()
        assert mock_proc.called
