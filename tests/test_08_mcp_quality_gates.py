"""
tests/test_08_mcp_quality_gates.py
Unit tests for scripts/08_mcp_quality_gates.py.
"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Dynamically import scripts/08_mcp_quality_gates.py
script_path = Path(__file__).parent.parent / "scripts" / "08_mcp_quality_gates.py"
spec = importlib.util.spec_from_file_location("mcp_quality_gates", script_path)
mcp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_module)

app = mcp_module.app
commit_changes = mcp_module.commit_changes
get_git_diff = mcp_module.get_git_diff
parse_args = mcp_module.parse_args
run_quality_gates = mcp_module.run_quality_gates
validate_containerfile = mcp_module.validate_containerfile

client = TestClient(app)


def test_validate_containerfile_valid(tmp_path):
    containerfile = tmp_path / "Containerfile"
    containerfile.write_text(
        "FROM docker.io/library/debian:bookworm-slim AS builder\n"
        "RUN apt-get update && apt-get install -y build-essential\n"
        "FROM docker.io/library/python:3.11-slim-bookworm\n"
        "WORKDIR /app\n"
        "CMD ['/bin/bash']\n"
    )
    valid, msg = validate_containerfile(str(containerfile))
    assert valid is True
    assert "compliant" in msg.lower()


def test_validate_containerfile_missing():
    valid, msg = validate_containerfile("non_existent_file_path")
    assert valid is True
    assert "not found" in msg.lower()


def test_validate_containerfile_invalid_alpine(tmp_path):
    containerfile = tmp_path / "Containerfile"
    containerfile.write_text("FROM alpine:3.18\nRUN apk add --no-cache python3 gcc musl-dev\nCMD ['python3']\n")
    valid, msg = validate_containerfile(str(containerfile))
    assert valid is False
    assert "rule violation" in msg.lower()
    assert "alpine" in msg.lower()


def test_validate_containerfile_invalid_apk(tmp_path):
    containerfile = tmp_path / "Containerfile"
    containerfile.write_text("FROM debian:bookworm-slim\nRUN apk update && apk add curl\nCMD ['/bin/bash']\n")
    valid, msg = validate_containerfile(str(containerfile))
    assert valid is False
    assert "apk" in msg.lower()


@patch("subprocess.run")
def test_get_git_diff(mock_run):
    mock_run.return_value = MagicMock(stdout="diff --git a/foo.py b/foo.py\n+ # fix\n", returncode=0)
    diff = get_git_diff(".")
    assert "+ # fix" in diff


@patch("subprocess.run")
def test_commit_changes(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="[main 1234567] style fix")
    success, msg = commit_changes(".", "style fix")
    assert success is True
    assert "1234567" in msg or "style fix" in msg


@patch.object(mcp_module, "validate_containerfile")
@patch.object(mcp_module, "check_git_status")
@patch("subprocess.run")
def test_run_quality_gates_all_pass(mock_sub_run, mock_git_status, mock_validate):
    mock_validate.return_value = (True, "Compliant")
    mock_git_status.return_value = False
    mock_sub_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    res = run_quality_gates(repo_path=".", auto_commit=False, containerfile_path="Containerfile")

    assert res["success"] is True
    assert res["containerfile_valid"] is True
    assert res["ruff_check_passed"] is True
    assert res["ruff_format_passed"] is True
    assert res["pytest_passed"] is True


@patch.object(mcp_module, "validate_containerfile")
@patch.object(mcp_module, "check_git_status")
@patch("subprocess.run")
def test_run_quality_gates_pytest_failure(mock_sub_run, mock_git_status, mock_validate):
    mock_validate.return_value = (True, "Compliant")
    mock_git_status.return_value = False

    def side_effect(cmd, **kwargs):
        if "pytest" in cmd:
            return MagicMock(returncode=1, stdout="1 test failed", stderr="AssertionError")
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_sub_run.side_effect = side_effect

    res = run_quality_gates(repo_path=".", auto_commit=False, containerfile_path="Containerfile")

    assert res["success"] is False
    assert res["pytest_passed"] is False
    assert "Pytest suite failed" in res["error_log"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch.object(mcp_module, "run_quality_gates")
def test_webhook_pr_opened_success(mock_run_gates):
    mock_run_gates.return_value = {
        "success": True,
        "auto_committed": True,
        "logs": "All gates passed",
        "error_log": "",
        "suggested_diff": "",
    }

    payload = {"action": "opened", "pull_request": {"number": 42}}
    response = client.post("/webhook/pull-request", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["pr_number"] == 42


@patch.object(mcp_module, "run_quality_gates")
def test_webhook_pr_synchronize_failure(mock_run_gates):
    mock_run_gates.return_value = {
        "success": False,
        "auto_committed": False,
        "logs": "Gates failed",
        "error_log": "Pytest assertion error",
        "suggested_diff": "diff --git...",
    }

    payload = {"action": "synchronize", "pull_request": {"number": 101}}
    response = client.post("/webhook/pull-request", json=payload)

    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "failure"
    assert "comment_suggestion" in data


def test_webhook_pr_ignored_action():
    payload = {"action": "closed", "pull_request": {"number": 12}}
    response = client.post("/webhook/pull-request", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ignored"


def test_mcp_initialize():
    req = {"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1}
    response = client.post("/mcp", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["serverInfo"]["name"] == "maurice-mcp-quality-gates"


def test_mcp_ping():
    req = {"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 2}
    response = client.post("/mcp", json=req)
    assert response.status_code == 200
    assert response.json()["result"] == {}


def test_mcp_tools_list():
    req = {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 3}
    response = client.post("/mcp", json=req)
    assert response.status_code == 200
    tools = response.json()["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "run_quality_gates" in tool_names
    assert "validate_containerfile" in tool_names


@patch.object(mcp_module, "validate_containerfile")
def test_mcp_tools_call_validate_containerfile(mock_validate):
    mock_validate.return_value = (True, "Containerfile OK")
    req = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "validate_containerfile",
            "arguments": {"filepath": "Containerfile"},
        },
        "id": 4,
    }
    response = client.post("/mcp", json=req)
    assert response.status_code == 200
    content_text = response.json()["result"]["content"][0]["text"]
    parsed = json.loads(content_text)
    assert parsed["valid"] is True


def test_parse_args():
    with patch("sys.argv", ["script", "--host", "127.0.0.1", "--port", "9090", "--run-gates"]):
        args = parse_args()
        assert args.host == "127.0.0.1"
        assert args.port == 9090
        assert args.run_gates is True
