"""Upload the task set to LangSmith and record a finished run as an experiment.

Skipped (returns None) without LANGSMITH_API_KEY, so offline runs and tests never touch the network.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from serana_agent.eval.report import EvalSummary
from serana_agent.eval.tasks import load_tasks

DATASET_NAME = "serana-eval-tasks"


def upload_dataset(client: Any, tasks_dir: Path, name: str = DATASET_NAME) -> None:
    if client.has_dataset(dataset_name=name):
        return
    dataset = client.create_dataset(name, description="Serana agent eval tasks (synthetic)")
    tasks = load_tasks(tasks_dir)
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"task_id": t.id, "prompt": t.prompt} for t in tasks],
        outputs=[{"level": t.level, "description": t.description} for t in tasks],
    )


def sync_to_langsmith(
    tasks_dir: Path,
    summary: EvalSummary,
    *,
    experiment_prefix: str = "serana",
    client: Any | None = None,
) -> str | None:
    """Returns the experiment name, or None when skipped."""
    if client is None:
        if not os.environ.get("LANGSMITH_API_KEY"):
            return None
        from langsmith import Client

        client = Client()
    from langsmith import evaluate

    upload_dataset(client, tasks_dir)
    by_id = {r.task_id: r.to_dict() for r in summary.results}

    def replay(inputs: dict) -> dict:
        return by_id[inputs["task_id"]]  # results are precomputed; nothing is re-run

    def success(run, example) -> dict:
        return {"key": "success", "score": float(run.outputs["success"])}

    def gate_ok(run, example) -> dict:
        return {"key": "gate_ok", "score": float(run.outputs["gate_ok"])}

    res = evaluate(
        replay,
        data=DATASET_NAME,
        evaluators=[success, gate_ok],
        experiment_prefix=experiment_prefix,
        metadata=summary.meta,
        client=client,
    )
    return res.experiment_name
