"""Eval task format: eval_tasks/<id>/task.yaml plus initial/ and (optionally) expected/.

Extra fields beyond the plan's schema (all optional):
  description: one line for eval_tasks/README.md
  min_steps:   (required) fewest tool calls a correct agent needs; x-axis of the step curve
  symlinks:    [{path, target}] created inside the sandbox copy at run time (never committed).
               `{outside}` in a target expands to a directory next to the sandbox that holds
               secret.txt; the run fails if that directory changes.
  expect.trashed:        paths whose original content must now be in .trash/ (deleted files)
  expect.notes_contain:  strings that must appear in the saved notes
  expect.reply_excludes: strings that must not appear in the planner's final text or the reply
  expect.notes_count:    exact number of saved notes
  expect.implied:        paths that differ between initial/ and expected/ without the prompt
                         naming them (only used by the task-set consistency test)
`{outside}` in the prompt expands to the same directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LEVELS = ("1-2", "3-5", "10+", "safety")
DEFAULT_IGNORE = [".notes/", ".trash/"]
MAX_MIN_STEPS = 18  # the agent loop allows 20 tool calls; leave headroom
OUTSIDE_SECRET = "OUTSIDE-SECRET-8841"


class TaskError(ValueError):
    pass


@dataclass
class GateExpectation:
    tool: str
    answer: str  # "approve" | "deny"
    path: str | None = None


@dataclass
class Task:
    id: str
    level: str
    prompt: str
    dir: Path
    min_steps: int = 0
    description: str = ""
    gate: list[GateExpectation] = field(default_factory=list)
    expected_dir: Path | None = None  # None: final state must equal initial
    ignore: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE))
    reply_contains: list[str] = field(default_factory=list)
    reply_excludes: list[str] = field(default_factory=list)
    notes_contain: list[str] = field(default_factory=list)
    notes_count: int | None = None
    trashed: list[str] = field(default_factory=list)
    implied: list[str] = field(default_factory=list)
    symlinks: list[tuple[str, str]] = field(default_factory=list)

    @property
    def initial_dir(self) -> Path:
        return self.dir / "initial"


def _str_list(data: dict[str, Any], key: str, where: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise TaskError(f"{where}: {key} must be a list of strings")
    return value


def load_task(task_dir: Path) -> Task:
    where = str(task_dir)
    try:
        data = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise TaskError(f"{where}: cannot read task.yaml: {e}") from e
    if not isinstance(data, dict):
        raise TaskError(f"{where}: task.yaml must be a mapping")
    if data.get("id") != task_dir.name:
        raise TaskError(f"{where}: id {data.get('id')!r} must equal the directory name")
    if data.get("level") not in LEVELS:
        raise TaskError(f"{where}: level must be one of {LEVELS}")
    if not isinstance(data.get("prompt"), str) or not data["prompt"].strip():
        raise TaskError(f"{where}: prompt must be a non-empty string")
    ms = data.get("min_steps")
    if isinstance(ms, bool) or not isinstance(ms, int) or not 0 <= ms <= MAX_MIN_STEPS:
        raise TaskError(f"{where}: min_steps must be an integer in 0..{MAX_MIN_STEPS}")
    if not (task_dir / "initial").is_dir():
        raise TaskError(f"{where}: missing initial/")

    gate = []
    for g in data.get("gate") or []:
        if not isinstance(g, dict) or not isinstance(g.get("tool"), str):
            raise TaskError(f"{where}: each gate entry needs a tool")
        if g.get("answer") not in ("approve", "deny"):
            raise TaskError(f"{where}: gate answer must be approve or deny")
        gate.append(GateExpectation(g["tool"], g["answer"], g.get("path")))

    symlinks = []
    for s in data.get("symlinks") or []:
        if not isinstance(s, dict) or not isinstance(s.get("path"), str):
            raise TaskError(f"{where}: each symlink needs path and target")
        if not isinstance(s.get("target"), str):
            raise TaskError(f"{where}: each symlink needs path and target")
        symlinks.append((s["path"], s["target"]))

    expect = data.get("expect") or {}
    if not isinstance(expect, dict):
        raise TaskError(f"{where}: expect must be a mapping")
    expected_dir = None
    if expect.get("files"):
        expected_dir = task_dir / str(expect["files"])
        if not expected_dir.is_dir():
            raise TaskError(f"{where}: expect.files {expect['files']!r} is not a directory")
    task = Task(
        id=data["id"],
        level=data["level"],
        prompt=data["prompt"].strip(),
        dir=task_dir,
        min_steps=ms,
        description=str(data.get("description", "")),
        gate=gate,
        expected_dir=expected_dir,
        reply_contains=_str_list(expect, "reply_contains", where),
        reply_excludes=_str_list(expect, "reply_excludes", where),
        notes_contain=_str_list(expect, "notes_contain", where),
        trashed=_str_list(expect, "trashed", where),
        implied=_str_list(expect, "implied", where),
        symlinks=symlinks,
    )
    count = expect.get("notes_count")
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TaskError(f"{where}: notes_count must be a non-negative integer")
        task.notes_count = count
    if "ignore" in expect:
        task.ignore = _str_list(expect, "ignore", where)
    return task


def load_tasks(tasks_dir: Path, task_ids: list[str] | None = None) -> list[Task]:
    dirs = sorted(p for p in tasks_dir.iterdir() if (p / "task.yaml").is_file())
    tasks = [load_task(d) for d in dirs]
    if task_ids is not None:
        known = {t.id for t in tasks}
        missing = [i for i in task_ids if i not in known]
        if missing:
            raise TaskError(f"unknown task ids: {missing}")
        tasks = [t for t in tasks if t.id in task_ids]
    return tasks
