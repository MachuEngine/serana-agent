"""Runs eval tasks: fresh sandbox + fresh tool server per task, scripted gate answers."""

from __future__ import annotations

import asyncio
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

from serana_agent.agent.types import Approver, RunResult
from serana_agent.eval.report import EvalSummary, append_result, summarize, write_report
from serana_agent.eval.scoring import (
    TaskResult,
    error_result,
    make_outside,
    materialize,
    score_run,
)
from serana_agent.eval.tasks import Task, load_tasks
from serana_agent.memory.types import SkillStore
from serana_agent.tools.client import ToolClient


class RunnableAgent(Protocol):
    async def run(self, task: str) -> RunResult: ...


AgentFactory = Callable[[ToolClient, Approver, SkillStore | None], RunnableAgent]


async def _run_task(
    task: Task,
    agent_factory: AgentFactory,
    sandbox_mode: Literal["host", "docker"],
    skills: SkillStore | None,
    timeout_s: float,
) -> TaskResult:
    # Imported here so this module loads even before WP-D's gate module exists.
    from serana_agent.agent.gate import GateRule, ScriptedApprover

    with tempfile.TemporaryDirectory(prefix=f"serana-eval-{task.id}-") as tmp:
        base = Path(tmp)
        sandbox, expected, outside = base / "sandbox", base / "expected", base / "outside"
        make_outside(outside)
        materialize(task.initial_dir, sandbox, task.symlinks, outside)
        materialize(task.expected_dir or task.initial_dir, expected, task.symlinks, outside)
        approver = ScriptedApprover([GateRule(g.tool, g.path, g.answer) for g in task.gate])
        prompt = task.prompt.replace("{outside}", str(outside))

        result: RunResult | None = None
        error: str | None = None
        start = time.monotonic()
        async with ToolClient(sandbox, sandbox_mode) as client:
            specs = await client.list_tools()
            try:
                agent = agent_factory(client, approver, skills)
                result = await asyncio.wait_for(agent.run(prompt), timeout_s)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
        wall = time.monotonic() - start
        scored = score_run(
            task,
            result,
            sandbox=sandbox,
            expected_root=expected,
            outside=outside,
            specs=specs,
            approver=approver,
            error=error,
        )
        if not scored.latency_s:
            scored.latency_s = round(wall, 3)
        return scored


async def run_task(
    task: Task,
    agent_factory: AgentFactory,
    *,
    sandbox_mode: Literal["host", "docker"] = "host",
    skills: SkillStore | None = None,
    timeout_s: float = 600.0,
) -> TaskResult:
    try:
        return await _run_task(task, agent_factory, sandbox_mode, skills, timeout_s)
    except Exception as e:  # server start, list_tools or sandbox setup failed: this task only
        return error_result(task, f"{type(e).__name__}: {e}")


async def run_eval(
    tasks_dir: Path,
    agent_factory: AgentFactory,
    *,
    out_dir: Path,
    sandbox_mode: Literal["host", "docker"] = "host",
    skills: SkillStore | None = None,
    task_ids: list[str] | None = None,
    meta: dict[str, Any] | None = None,
    task_timeout_s: float = 600.0,
) -> EvalSummary:
    if skills is not None and not skills.read_only:
        raise ValueError(
            "eval needs a read-only skill store so earlier tasks cannot change later ones"
        )
    tasks = load_tasks(tasks_dir, task_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text("", encoding="utf-8")
    results = []
    for task in tasks:
        result = await run_task(
            task, agent_factory, sandbox_mode=sandbox_mode, skills=skills, timeout_s=task_timeout_s
        )
        append_result(out_dir, result)
        results.append(result)
    summary = EvalSummary(
        results,
        summarize(results),
        out_dir,
        meta={"sandbox_mode": sandbox_mode, "skills": skills is not None, **(meta or {})},
    )
    write_report(summary)
    return summary
