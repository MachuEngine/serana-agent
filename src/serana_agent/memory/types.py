"""Interfaces for long-term memory and the skill store, used by the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class MemoryItem:
    id: str
    text: str  # one user fact or preference, e.g. "Prefers summaries in bullet points"
    metadata: dict[str, Any] = field(default_factory=dict)  # source session, model, created_at


class MemoryStore(Protocol):
    def add(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryItem: ...
    def search(self, query: str, k: int = 5) -> list[MemoryItem]: ...
    def is_duplicate(self, text: str) -> bool: ...  # near-identical memory already stored
    def all(self) -> list[MemoryItem]: ...


@dataclass
class SkillStep:
    tool: str
    arguments: dict[str, Any]  # values may contain {placeholders} filled by the planner


@dataclass
class Skill:
    id: str
    name: str
    description: str  # when to use it; this text is what search matches against
    steps: list[SkillStep]
    confidence: float = 0.5  # 0..1, raised on successful reuse, lowered on failure
    failures: int = 0
    uses: int = 0
    enabled: bool = True


class SkillStore(Protocol):
    read_only: bool

    def add(self, name: str, description: str, steps: list[SkillStep]) -> Skill: ...
    def search(self, query: str, k: int = 3) -> list[Skill]: ...  # enabled skills only
    def record_outcome(self, skill_id: str, success: bool) -> Skill: ...
    def all(self) -> list[Skill]: ...
