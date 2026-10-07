"""Types shared by the agent loop, CLI, and evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from serana_agent.llm.base import ToolCall
from serana_agent.tools.protocol import Risk, ToolOutcome


@dataclass
class ConfirmationRequest:
    tool: str
    arguments: dict[str, Any]
    risk: Risk
    change: str  # from ToolOutcome.change


class Approver(Protocol):
    """Answers the confirm gate. Terminal asks the user; eval answers from the task spec."""

    def approve(self, request: ConfirmationRequest) -> bool: ...


@dataclass
class Step:
    index: int
    tool_call: ToolCall | None  # None when the planner produced the final answer
    outcome: ToolOutcome | None
    gate: ConfirmationRequest | None = None
    approved: bool | None = None  # None when no gate was opened
    planner_text: str = ""
    retried_invalid_call: bool = False


StopReason = str  # "final" | "step_limit" | "repeated_call" | "invalid_tool_call" | "error"


@dataclass
class RunResult:
    task: str
    reply: str  # persona's in-character reply shown to the user
    steps: list[Step] = field(default_factory=list)
    stop_reason: StopReason = "final"
    planner_model: str = ""
    persona_model: str = ""
    latency_s: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    used_skill_ids: list[str] = field(default_factory=list)
