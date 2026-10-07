"""Session-end reflection: extract durable user facts and preferences into memory."""

from __future__ import annotations

from typing import Any

from serana_agent.llm.base import ChatModel, Message
from serana_agent.memory.jsonparse import extract_json
from serana_agent.memory.types import MemoryItem, MemoryStore

PROMPT = """You review a conversation between a user and an assistant.
List durable facts about the user or their preferences that would help in future sessions
(name, habits, tools they use, how they like answers formatted).
Skip one-off task details and anything about the assistant itself.
Answer with only a JSON array of short strings, e.g. ["Prefers bullet-point summaries"].
Answer [] if there is nothing worth remembering."""


MAX_TRANSCRIPT_CHARS = 12000  # keep the prompt inside the local model context


def _transcript(messages: list[Message]) -> str:
    lines = [f"{m['role']}: {m.get('content', '')}" for m in messages if m.get("content")]
    # Later turns matter most, so drop from the front.
    return "\n".join(lines)[-MAX_TRANSCRIPT_CHARS:]


def reflect(
    messages: list[Message],
    model: ChatModel,
    store: MemoryStore,
    metadata: dict[str, Any] | None = None,
) -> list[MemoryItem]:
    """Add new facts from the session to `store` and return the items that were added."""
    transcript = _transcript([m for m in messages if m["role"] in ("user", "assistant")])
    if not transcript:
        return []
    result = model.chat(
        [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": transcript},
        ],
        max_tokens=512,
    )
    facts = extract_json(result.content, "[")
    if not isinstance(facts, list):
        return []
    added: list[MemoryItem] = []
    for fact in facts:
        if not isinstance(fact, str) or not fact.strip():
            continue
        fact = fact.strip()
        if store.is_duplicate(fact):
            continue
        added.append(store.add(fact, metadata))
    return added
