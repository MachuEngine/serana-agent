"""Long-term memory backed by a persistent Chroma collection."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb

from serana_agent.memory.types import MemoryItem

COLLECTION = "memories"
# Cosine distance below which a new fact is treated as already known.
DUPLICATE_DISTANCE = 0.1


def _scalar_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    # Chroma only accepts scalar metadata values.
    return {
        k: v if isinstance(v, str | int | float | bool) else str(v) for k, v in metadata.items()
    }


class ChromaMemoryStore:
    def __init__(self, path: str | Path, embedding_function: Any | None = None):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        kwargs: dict[str, Any] = {"metadata": {"hnsw:space": "cosine"}}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self._col = self._client.get_or_create_collection(COLLECTION, **kwargs)

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryItem:
        meta = {"created_at": datetime.now(UTC).isoformat(), **(metadata or {})}
        item = MemoryItem(id=uuid.uuid4().hex, text=text, metadata=meta)
        self._col.add(ids=[item.id], documents=[text], metadatas=[_scalar_metadata(meta)])
        return item

    def search(self, query: str, k: int = 5) -> list[MemoryItem]:
        return [item for item, _ in self._query(query, k)]

    def is_duplicate(self, text: str) -> bool:
        hits = self._query(text, 1)
        return bool(hits) and hits[0][1] < DUPLICATE_DISTANCE

    def all(self) -> list[MemoryItem]:
        res = self._col.get()
        items = [
            MemoryItem(id=i, text=d, metadata=dict(m or {}))
            for i, d, m in zip(
                res["ids"], res["documents"] or [], res["metadatas"] or [], strict=False
            )
        ]
        return sorted(items, key=lambda it: str(it.metadata.get("created_at", "")))

    def _query(self, query: str, k: int) -> list[tuple[MemoryItem, float]]:
        count = self._col.count()
        if count == 0 or k <= 0:
            return []
        res = self._col.query(query_texts=[query], n_results=min(k, count))
        return [
            (MemoryItem(id=i, text=d, metadata=dict(m or {})), dist)
            for i, d, m, dist in zip(
                res["ids"][0],
                (res["documents"] or [[]])[0],
                (res["metadatas"] or [[]])[0],
                (res["distances"] or [[]])[0],
                strict=False,
            )
        ]
