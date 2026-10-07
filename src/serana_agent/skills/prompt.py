"""Render retrieved skills as text for the planner system prompt."""

from __future__ import annotations

import json

from serana_agent.memory.types import Skill


def format_skills(skills: list[Skill]) -> str:
    if not skills:
        return ""
    blocks = []
    for s in skills:
        steps = "\n".join(
            f"  {i}. {st.tool}({json.dumps(st.arguments, ensure_ascii=False)})"
            for i, st in enumerate(s.steps, 1)
        )
        blocks.append(f"- {s.name} (confidence {s.confidence:.1f}): {s.description}\n{steps}")
    return (
        "Procedures that worked before for similar requests. Adapt the {placeholders} "
        "to the current task; do not follow them if they do not fit.\n" + "\n".join(blocks)
    )
