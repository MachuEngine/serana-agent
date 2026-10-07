"""Contract between the MCP tool server and the orchestrator.

Risk is enforced on the server: a tool call that would be destructive (delete, overwrite,
run_shell) does nothing unless its arguments include `confirmed: true`. Without it, the
server returns a ToolOutcome with status "confirmation_required" and a human-readable
description of the change. The orchestrator removes `confirmed` from the schemas shown to
the planner, asks the user through the confirm gate, and only then re-sends the call with
`confirmed: true`. The model therefore cannot approve its own destructive call.

Every tool returns ToolOutcome serialized as JSON text (one MCP TextContent item).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

CONFIRM_ARG = "confirmed"
MAX_OUTPUT_CHARS = 4000  # longer tool output is truncated with TRUNCATION_MARKER
TRUNCATION_MARKER = "\n...[truncated {omitted} chars]"


class Risk(StrEnum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


# Static risk per tool. write_file/move_file escalate to DESTRUCTIVE at call time when the
# target already exists (overwrite); the server decides that.
TOOL_RISK: dict[str, Risk] = {
    "list_dir": Risk.READ,
    "read_file": Risk.READ,
    "search_files": Risk.READ,
    "list_notes": Risk.READ,
    "write_file": Risk.WRITE,
    "edit_file": Risk.WRITE,
    "move_file": Risk.WRITE,
    "add_note": Risk.WRITE,
    "delete_file": Risk.DESTRUCTIVE,
    "run_shell": Risk.DESTRUCTIVE,
}

Status = Literal["ok", "error", "confirmation_required"]


@dataclass
class ToolOutcome:
    status: Status
    output: str = ""  # result text for the model (already truncated)
    error: str | None = None  # message the model can act on, e.g. "path is outside sandbox"
    risk: Risk = Risk.READ
    change: str | None = None  # for confirmation_required: what will happen, shown at the gate
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> ToolOutcome:
        data = json.loads(text)
        data["risk"] = Risk(data.get("risk", Risk.READ))
        return cls(**data)
