"""Append-only JSONL audit log: tool calls, gates, approvals, models used."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class AuditLog:
    """Keeps events in memory and appends them to `path` when one is given."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.events: list[dict[str, Any]] = []
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, run_id: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "event": event, "run_id": run_id, **fields}
        self.events.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def for_run(self, run_id: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["run_id"] == run_id]
