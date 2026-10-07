"""Aggregation and output: results.jsonl, summary.json and a per-level table."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from serana_agent.eval.scoring import TaskResult
from serana_agent.eval.tasks import LEVELS


@dataclass
class EvalSummary:
    results: list[TaskResult]
    summary: dict[str, Any]
    out_dir: Path
    meta: dict[str, Any] = field(default_factory=dict)


def default_out_dir(base: Path = Path("runs/eval")) -> Path:
    return base / time.strftime("%Y%m%d-%H%M%S")


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _curve(rs: list[TaskResult]) -> list[dict[str, Any]]:
    """Success rate by the minimal steps a task needs; actual calls are a secondary column."""
    by_min: dict[int, list[TaskResult]] = defaultdict(list)
    for r in rs:
        by_min[r.min_steps].append(r)
    return [
        {
            "min_steps": n,
            "tasks": len(v),
            "success": sum(r.success for r in v),
            "rate": _rate(sum(r.success for r in v), len(v)),
            "mean_actual_steps": round(sum(r.steps for r in v) / len(v), 2),
        }
        for n, v in sorted(by_min.items())
    ]


def _group(rs: list[TaskResult]) -> dict[str, Any]:
    ok = [r for r in rs if r.success]
    calls = sum(r.tool_calls_total for r in rs)
    return {
        "tasks": len(rs),
        "success": len(ok),
        "success_rate": _rate(len(ok), len(rs)),
        "mean_steps_on_success": round(sum(r.steps for r in ok) / len(ok), 2) if ok else None,
        "tool_call_accuracy": _rate(sum(r.tool_calls_valid for r in rs), calls),
        "gate_accuracy": _rate(sum(r.gate_ok for r in rs), len(rs)),
        "risky_rate": _rate(sum(r.unsafe_attempts > 0 for r in rs), len(rs)),
        "unsafe_attempts": sum(r.unsafe_attempts for r in rs),
        "sandbox_violation_attempts": sum(r.sandbox_violation_attempts for r in rs),
        "mean_latency_s": round(sum(r.latency_s for r in rs) / len(rs), 3) if rs else None,
        "total_tokens": sum(r.tokens for r in rs),
        "success_by_min_steps": _curve(rs),
    }


def summarize(results: list[TaskResult]) -> dict[str, Any]:
    by_level = {lv: _group([r for r in results if r.level == lv]) for lv in LEVELS}
    return {
        "overall": _group(results),
        "by_level": {k: v for k, v in by_level.items() if v["tasks"]},
        # Curve data (overall; each level has its own under by_level[...]["success_by_min_steps"]).
        "success_by_steps": _curve(results),
    }


def append_result(out_dir: Path, result: TaskResult) -> None:
    """Written as each task finishes so a crash later in the run loses nothing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def write_report(summary: EvalSummary) -> None:
    summary.out_dir.mkdir(parents=True, exist_ok=True)
    data = {**summary.summary, "meta": summary.meta}
    (summary.out_dir / "summary.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def print_table(summary: EvalSummary, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Eval results by level")
    for col in (
        "level",
        "tasks",
        "success",
        "tool calls",
        "gates",
        "risky",
        "steps (ok)",
        "latency s",
    ):
        table.add_column(col, justify="right" if col != "level" else "left")
    groups = {**summary.summary["by_level"], "all": summary.summary["overall"]}
    for name, g in groups.items():
        table.add_row(
            name,
            str(g["tasks"]),
            f"{g['success']}/{g['tasks']} ({_pct(g['success_rate'])})",
            _pct(g["tool_call_accuracy"]),
            _pct(g["gate_accuracy"]),
            _pct(g["risky_rate"]),
            "-" if g["mean_steps_on_success"] is None else str(g["mean_steps_on_success"]),
            "-" if g["mean_latency_s"] is None else str(g["mean_latency_s"]),
        )
    console.print(table)
