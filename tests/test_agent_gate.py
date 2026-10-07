import io

from rich.console import Console

from serana_agent.agent.audit import AuditLog
from serana_agent.agent.gate import GateRule, ScriptedApprover, TerminalApprover
from serana_agent.agent.types import ConfirmationRequest
from serana_agent.tools.protocol import Risk


def req(tool="delete_file", **args):
    return ConfirmationRequest(tool, args, Risk.DESTRUCTIVE, "will move to trash")


def test_scripted_approver_matches_rules():
    a = ScriptedApprover(
        [
            GateRule("delete_file", "old/report.md", "approve"),
            GateRule("delete_file", "keep.md", "deny"),
            GateRule("move_file", None, "approve"),
        ]
    )
    assert a.approve(req(path="old/report.md")) is True
    assert a.approve(req(path="keep.md")) is False
    assert a.approve(req("move_file", src="a", dst="b")) is True
    assert a.unexpected == []
    assert [ok for _, ok in a.opened] == [True, False, True]


def test_scripted_approver_denies_and_records_unexpected():
    a = ScriptedApprover([GateRule("delete_file", "x", "approve")])
    unexpected = req(path="y")
    assert a.approve(unexpected) is False
    assert a.unexpected == [unexpected] and a.opened == [(unexpected, False)]


def test_terminal_approver(monkeypatch):
    console = Console(file=io.StringIO(), width=80)
    approver = TerminalApprover(console)
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    assert approver.approve(req(path="[bold]a.txt")) is True
    shown = console.file.getvalue()
    assert "delete_file" in shown and "will move to trash" in shown and "[bold]a.txt" in shown
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert approver.approve(req(path="a.txt")) is False  # default is no

    def eof(*a):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert approver.approve(req(path="a.txt")) is False


def test_audit_log_writes_jsonl_and_filters_by_run(tmp_path):
    path = tmp_path / "s" / "audit.jsonl"
    log = AuditLog(path)
    log.log("tool_call", "r1", tool="list_dir")
    log.log("gate", "r2", approved=False)
    assert [e["tool"] for e in log.for_run("r1")] == ["list_dir"]
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and '"run_id": "r2"' in lines[1]


def test_scripted_approver_normalizes_paths():
    a = ScriptedApprover([GateRule("delete_file", "./old//report.md", "approve")])
    assert a.approve(req(path="old/report.md")) is True
    b = ScriptedApprover([GateRule("delete_file", "old/report.md", "approve")])
    assert b.approve(req(path="./old/x/../report.md")) is True
