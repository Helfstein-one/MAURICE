"""
tests/test_08_code_review.py
Comprehensive unit and integration test suite for scripts/08_code_review.py proactive code review daemon.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

script_path = Path(__file__).parent.parent / "scripts" / "08_code_review.py"
spec = importlib.util.spec_from_file_location("code_review", script_path)
code_review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_review)


def test_brace_balance_check():
    assert code_review.brace_balance_check("int main() { return (0); }") is True
    assert code_review.brace_balance_check("int main() { return (0; }") is False


def test_validate_code_syntax_python_valid():
    valid_python = "def add(a, b):\n    return a + b\n"
    assert code_review.validate_code_syntax(valid_python, "python") is True


def test_validate_code_syntax_python_invalid():
    invalid_python = "def add(a, b\n    return a +"
    assert code_review.validate_code_syntax(invalid_python, "python") is False


def test_validate_code_syntax_empty_or_non_string():
    assert code_review.validate_code_syntax("", "python") is False
    assert code_review.validate_code_syntax(None, "python") is False  # type: ignore[arg-type]


def test_extract_code_blocks():
    text = "Here is python code:\n```python\nx = 1\n```\nAnd diff:\n```diff\n--- a/f.py\n+++ b/f.py\n```"
    blocks = code_review.extract_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0] == ("python", "x = 1\n")
    assert blocks[1] == ("diff", "--- a/f.py\n+++ b/f.py\n")


def test_extract_diff_patches():
    text = "```diff\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x=1\n+x=2\n```"
    patches = code_review.extract_diff_patches(text)
    assert len(patches) == 1
    assert "--- a/f.py" in patches[0]


def test_query_mau_model_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "Mocked response from model"}}]}
    with patch("requests.post", return_value=mock_resp):
        res = code_review.query_mau_model("Test prompt", "http://localhost:8000")
        assert res == "Mocked response from model"


def test_query_mau_model_fallback_on_error():
    with patch("requests.post", side_effect=Exception("Connection refused")):
        res = code_review.query_mau_model("Test prompt", "http://localhost:8000")
        assert "Code Review Analysis" in res
        assert "<think>" in res


def test_parse_review_response():
    resp_text = (
        "<think>\nAnalyzing code...\n</think>\n"
        "### Code Review\n"
        "- **Anti-patterns**: Avoid global state\n"
        "- **Performance**: Use set for lookup\n"
        "- **Architectural Violations**: Layer coupling\n\n"
        "```python\n"
        "def safe_fn(x):\n"
        "    return x * 2\n"
        "```"
    )
    parsed = code_review.parse_review_response(resp_text)
    assert len(parsed["anti_patterns"]) >= 1
    assert len(parsed["performance_issues"]) >= 1
    assert len(parsed["architectural_violations"]) >= 1
    assert parsed["refactored_code"] == "def safe_fn(x):\n    return x * 2\n"
    assert parsed["is_safe"] is True


def test_apply_refactoring_to_files(tmp_path):
    target_file = tmp_path / "app.py"
    target_file.write_text("def old(): pass")

    review_result = {
        "refactored_code": "def new(): return True",
        "is_safe": True,
    }

    modified = code_review.apply_refactoring_to_files(
        review_result,
        file_map={"app.py": "def old(): pass"},
        target_file="app.py",
        repo_dir=str(tmp_path),
    )

    assert modified == ["app.py"]
    assert target_file.read_text() == "def new(): return True"


def test_apply_refactoring_unsafe_skips(tmp_path):
    review_result = {
        "refactored_code": "def syntax_error(",
        "is_safe": False,
    }
    modified = code_review.apply_refactoring_to_files(
        review_result,
        file_map={},
        target_file="app.py",
        repo_dir=str(tmp_path),
    )
    assert modified == []


def test_commit_refactoring_changes_success(tmp_path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        success = code_review.commit_refactoring_changes(
            branch_name="feature/refactor",
            modified_files=["app.py"],
            repo_dir=str(tmp_path),
            commit_msg="refactor(jules): code review suggestions",
        )
        assert success is True
        assert mock_run.call_count >= 2
        # Check that commit message was refactor(jules): code review suggestions
        args_list = [call.args[0] for call in mock_run.call_args_list]
        commit_cmd = next(cmd for cmd in args_list if "commit" in cmd)
        assert "refactor(jules): code review suggestions" in commit_cmd


def test_process_pr_event_dry_run():
    event_data = {
        "pr_id": "PR-123",
        "branch": "feature/test",
        "diff": "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-def f(): pass\n+def f(): return 1",
    }
    result = code_review.process_pr_event(event_data, dry_run=True)
    assert result["pr_id"] == "PR-123"
    assert result["branch"] == "feature/test"
    assert result["committed"] is False
    assert result["ready_for_human_approval"] is True


def test_handle_mcp_rpc_request_tools_list():
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    resp = code_review.handle_mcp_rpc_request(req, "http://localhost:8000")
    assert resp["jsonrpc"] == "2.0"
    assert "tools" in resp["result"]
    assert resp["result"]["tools"][0]["name"] == "review_pr_diff"


def test_handle_mcp_rpc_request_tools_call():
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "review_pr_diff",
            "arguments": {
                "pr_id": "PR-MCP-01",
                "branch": "feature/mcp",
                "diff": "diff --git a/a.py b/a.py",
                "auto_commit": False,
            },
        },
    }
    with patch.object(code_review, "process_pr_event") as mock_proc:
        mock_proc.return_value = {"pr_id": "PR-MCP-01", "committed": False}
        resp = code_review.handle_mcp_rpc_request(req, "http://localhost:8000")
        assert resp["jsonrpc"] == "2.0"
        assert resp["result"]["pr_id"] == "PR-MCP-01"


def test_handle_mcp_rpc_request_invalid_method():
    req = {"jsonrpc": "2.0", "id": 3, "method": "unknown/method"}
    resp = code_review.handle_mcp_rpc_request(req, "http://localhost:8000")
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_cli_parse_args():
    test_args = [
        "08_code_review.py",
        "--pr-event",
        "pr.json",
        "--server-url",
        "http://127.0.0.1:8000",
        "--dry-run",
    ]
    with patch("sys.argv", test_args):
        args = code_review.parse_args()
        assert args.pr_event == "pr.json"
        assert args.server_url == "http://127.0.0.1:8000"
        assert args.dry_run is True
