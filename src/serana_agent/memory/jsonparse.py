"""Pull a JSON value out of model text that may carry prose or code fences."""

from __future__ import annotations

import json
from typing import Any


def extract_json(text: str, opener: str) -> Any | None:
    """Return the first JSON value starting with `opener` ("[" or "{"), or None."""
    decoder = json.JSONDecoder()
    start = text.find(opener)
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
            return value
        except ValueError:
            start = text.find(opener, start + 1)
    return None
