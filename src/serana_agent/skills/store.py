"""Skill store kept in one JSON file; search ranks skills by description similarity."""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from serana_agent.memory.types import Skill, SkillStep

log = logging.getLogger(__name__)

MAX_FAILURES = 3
SUCCESS_BONUS = 0.1
FAILURE_PENALTY = 0.2


def _default_embedding() -> Any:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class JsonSkillStore:
    def __init__(
        self, path: str | Path, embedding_function: Any | None = None, read_only: bool = False
    ):
        self.path = Path(path)
        self.read_only = read_only
        self._ef = embedding_function
        self._skills: dict[str, Skill] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            for raw in json.loads(self.path.read_text(encoding="utf-8")):
                raw["steps"] = [SkillStep(**s) for s in raw["steps"]]
                skill = Skill(**raw)
                self._skills[skill.id] = skill
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            self._skills = {}
            backup = self.path.with_name(self.path.name + ".bak")
            log.warning("Could not load %s (%s); starting empty", self.path, e)
            if not self.read_only:
                self.path.replace(backup)

    def add(self, name: str, description: str, steps: list[SkillStep]) -> Skill:
        skill = Skill(id=uuid.uuid4().hex, name=name, description=description, steps=steps)
        if not self.read_only:
            self._skills[skill.id] = skill
            self._save()
        return skill

    def search(self, query: str, k: int = 3) -> list[Skill]:
        enabled = [s for s in self._skills.values() if s.enabled]
        if not enabled or k <= 0:
            return []
        if self._ef is None:
            self._ef = _default_embedding()
        vectors = self._ef([query] + [s.description for s in enabled])
        q = [float(x) for x in vectors[0]]
        scored = [
            (_cosine(q, [float(x) for x in v]), s)
            for v, s in zip(vectors[1:], enabled, strict=False)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [s for _, s in scored[:k]]

    def record_outcome(self, skill_id: str, success: bool) -> Skill:
        skill = self._skills[skill_id]
        if self.read_only:
            return skill
        skill.uses += 1
        if success:
            skill.confidence = min(1.0, round(skill.confidence + SUCCESS_BONUS, 10))
        else:
            skill.confidence = max(0.0, round(skill.confidence - FAILURE_PENALTY, 10))
            skill.failures += 1
            if skill.failures >= MAX_FAILURES:
                skill.enabled = False
        self._save()
        return skill

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(s) for s in self._skills.values()]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
