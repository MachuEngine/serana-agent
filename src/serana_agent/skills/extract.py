"""Turn a successful run trajectory into a reusable skill."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from serana_agent.agent.types import Step
from serana_agent.llm.base import ChatModel
from serana_agent.memory.jsonparse import extract_json
from serana_agent.memory.types import SkillStep
from serana_agent.tools.protocol import CONFIRM_ARG

MIN_TOOL_CALLS = 2

# Free-text arguments are always replaced: they are task-specific and may be private or long.
FREE_TEXT_ARGS = {"content", "old", "new", "text", "query"}
PLACEHOLDER = re.compile(r"\{\w+\}")

PROMPT = """You summarize a sequence of tool calls that completed a task into a reusable skill.
Answer with only a JSON object: {"name": "short_snake_case_name",
"description": "one sentence on when to use this procedure",
"placeholders": {"<literal value>": "{placeholder_name}"}}.
In "placeholders", map each task-specific value (file or folder names, paths) that appears in the
tool calls to a placeholder such as {source_file}. Leave out values that are fixed parts of the
procedure. Free-text arguments are already shown as {placeholders}."""


@dataclass
class SkillDraft:
    name: str
    description: str
    steps: list[SkillStep]


def _generalize(arguments: dict, mapping: dict[str, str]) -> dict:
    out = {}
    for key, value in arguments.items():
        if key in FREE_TEXT_ARGS:
            out[key] = "{" + key + "}"
        elif isinstance(value, str):
            out[key] = mapping.get(value, value)
        else:
            out[key] = value
    return out


def extract_skill(task: str, steps: list[Step], model: ChatModel) -> SkillDraft | None:
    """Return a draft (name, description, steps) or None if there is nothing worth saving."""
    skill_steps = [
        SkillStep(
            tool=s.tool_call.name,
            arguments={k: v for k, v in s.tool_call.arguments.items() if k != CONFIRM_ARG},
        )
        for s in steps
        if s.tool_call is not None and s.outcome is not None and s.outcome.status == "ok"
    ]
    if len(skill_steps) < MIN_TOOL_CALLS:
        return None
    skill_steps = [SkillStep(s.tool, _generalize(s.arguments, {})) for s in skill_steps]
    listing = "\n".join(
        f"{i}. {s.tool}({json.dumps(s.arguments, ensure_ascii=False)})"
        for i, s in enumerate(skill_steps, 1)
    )
    result = model.chat(
        [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Task: {task}\nTool calls:\n{listing}"},
        ],
        max_tokens=256,
    )
    data = extract_json(result.content, "{")
    if not isinstance(data, dict):
        return None
    name, description = data.get("name"), data.get("description")
    if not (isinstance(name, str) and isinstance(description, str) and name and description):
        return None
    raw = data.get("placeholders")
    mapping = (
        {k: v for k, v in raw.items() if isinstance(v, str) and PLACEHOLDER.fullmatch(v)}
        if isinstance(raw, dict)
        else {}
    )
    skill_steps = [SkillStep(s.tool, _generalize(s.arguments, mapping)) for s in skill_steps]
    return SkillDraft(name.strip(), description.strip(), skill_steps)
