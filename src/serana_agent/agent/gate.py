"""Confirm gate approvers: terminal prompt for users, scripted answers for evaluation."""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm

from serana_agent.agent.types import ConfirmationRequest

MAX_ARG_CHARS = 200


@dataclass
class GateRule:
    tool: str
    path: str | None  # matched against the call's `path` (or `src`); None matches any
    answer: Literal["approve", "deny"]


class ScriptedApprover:
    """Answers from rules. A gate no rule expects is denied and recorded in `unexpected`."""

    def __init__(self, rules: list[GateRule]):
        self.rules = rules
        self.opened: list[tuple[ConfirmationRequest, bool]] = []
        self.unexpected: list[ConfirmationRequest] = []

    def approve(self, request: ConfirmationRequest) -> bool:
        target = request.arguments.get("path", request.arguments.get("src"))
        target = posixpath.normpath(target) if isinstance(target, str) else target
        for rule in self.rules:
            wanted = posixpath.normpath(rule.path) if rule.path is not None else None
            if rule.tool == request.tool and wanted in (None, target):
                answer = rule.answer == "approve"
                break
        else:
            self.unexpected.append(request)
            answer = False
        self.opened.append((request, answer))
        return answer


def _short(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= MAX_ARG_CHARS else text[:MAX_ARG_CHARS] + "..."


class TerminalApprover:
    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def approve(self, request: ConfirmationRequest) -> bool:
        args = "\n".join(f"{k}: {_short(v)}" for k, v in request.arguments.items())
        body = f"[bold]{request.tool}[/bold]\n{escape(args)}\n\n{escape(request.change)}"
        self.console.print(Panel(body, title="확인 필요", border_style="yellow", expand=False))
        try:
            return Confirm.ask("실행할까요?", default=False, console=self.console)
        except EOFError:
            return False
