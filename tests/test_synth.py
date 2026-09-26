"""
Unit tests for MAURICE RLAIF Preference Synthesizer (maurice/synth.py)
"""

import json
import os
import tempfile

from maurice.cli import main as cli_main
from maurice.synth import (
    generate_responses,
    judge_response,
    parse_score_from_judge_response,
    score_response_heuristically,
    synth_prefs_variant,
)


def test_parse_score_from_judge_response():
    assert parse_score_from_judge_response("Based on analysis... Score: 8.5", default_score=5.0) == 8.5
    assert parse_score_from_judge_response("Score: 12.0", default_score=5.0) == 10.0
    assert parse_score_from_judge_response("Score: 0.5", default_score=5.0) == 1.0
    assert parse_score_from_judge_response("No score here", default_score=4.0) == 4.0


def test_score_response_heuristically():
    # Code variant
    code_good = "<think>\nValidating Python AST\n</think>\n```python\ndef add(a, b):\n    return a + b\n```"
    code_bad = "def add(a, b) return a + b"
    score_good = score_response_heuristically("Write python add function", code_good, variant="c")
    score_bad = score_response_heuristically("Write python add function", code_bad, variant="c")
    assert score_good > score_bad

    # Reasoning variant
    reasoning_good = "<think>\nStep 1: Compute 2 + 2 = 4.\nStep 2: Multiply 4 * 2 = 8.\n</think>\nThe answer is 8."
    reasoning_bad = "The answer is 8."
    s_r_good = score_response_heuristically("Calculate 2+2*2", reasoning_good, variant="r")
    s_r_bad = score_response_heuristically("Calculate 2+2*2", reasoning_bad, variant="r")
    assert s_r_good > s_r_bad


def test_generate_responses():
    responses = generate_responses("Test prompt", variant="c", num_responses=2, engine_type="mock")
    assert len(responses) == 2
    assert isinstance(responses[0], str)


def test_judge_response():
    score = judge_response("Test prompt", "Test response", variant="g", engine_type="mock")
    assert 1.0 <= score <= 10.0


def test_synth_prefs_variant_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "sft.jsonl")
        output_file = os.path.join(tmpdir, "prefs.jsonl")

        sample_sft = [
            {
                "messages": [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Write Python factorial function"},
                    {"role": "assistant", "content": "Assistant answer"},
                ]
            }
        ]

        with open(input_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(s) + "\n" for s in sample_sft)

        count = synth_prefs_variant(
            variant="c",
            input_file=input_file,
            output_file=output_file,
            num_responses=2,
            dry_run=True,
            engine_type="mock",
        )

        assert count == 1
        assert os.path.exists(output_file)

        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            item = json.loads(lines[0])
            assert "prompt" in item
            assert "chosen" in item
            assert "rejected" in item
            assert item["prompt"] == "Write Python factorial function"


def test_cli_synth_prefs(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "prefs_cli.jsonl")
        cli_args = [
            "maurice",
            "synth-prefs",
            "--variant",
            "c",
            "--output-file",
            output_file,
            "--dry-run",
        ]
        cli_main(cli_args[1:])
        assert os.path.exists(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) > 0
            item = json.loads(lines[0])
            assert "prompt" in item
            assert "chosen" in item
            assert "rejected" in item
