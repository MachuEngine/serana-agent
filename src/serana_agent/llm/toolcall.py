"""Parse Qwen3 output: strip `<think>` blocks and extract `<tool_call>` JSON blocks."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from serana_agent.llm.base import ToolCall

_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _split_think(text: str) -> tuple[str, bool]:
    """Remove reasoning from `text` (which must not contain tool_call blocks).

    Returns (text, unclosed): `unclosed` means a <think> was cut off before </think>.
    """
    text = _THINK_CLOSED.sub("", text)
    if "</think>" in text:  # the template already opened <think>, only the closing tag remains
        text = text.split("</think>", 1)[1]
    if "<think>" in text:
        return text.split("<think>", 1)[0].strip(), True
    return text.strip(), False


def strip_think(text: str) -> str:
    return _split_think(text)[0]


def _drop_leading_think(text: str) -> tuple[str, bool]:
    """Cut the reasoning prefix. Only the first </think> counts when it comes before any
    <tool_call>, so a "</think>" inside tool-call arguments is left alone."""
    stripped = text.lstrip()
    end = stripped.find("</think>")
    first_call = stripped.find("<tool_call>")
    if stripped.startswith("<think>"):
        if end == -1:
            return "", True
        return stripped[end + len("</think>") :], False
    if end != -1 and (first_call == -1 or end < first_call):
        return stripped[end + len("</think>") :], False
    return text, False


def _parse_block(raw: str) -> ToolCall:
    try:
        data = json.loads(raw.strip(), strict=False)  # models emit raw newlines in strings
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in <tool_call>: {e.msg} (position {e.pos})") from e
    if not isinstance(data, dict):
        raise ValueError("<tool_call> JSON must be an object")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError('<tool_call> JSON is missing a string "name"')
    args: Any = data.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args, strict=False)
        except json.JSONDecodeError as e:
            raise ValueError(f'"arguments" for {name} is a string that is not JSON: {e.msg}') from e
    if not isinstance(args, dict):
        raise ValueError(f'"arguments" for {name} must be a JSON object')
    return ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=args)


def parse_output(text: str) -> tuple[str, list[ToolCall], str | None]:
    """Return (content, tool_calls, parse_error). Content excludes think and tool_call blocks."""
    text, unclosed = _drop_leading_think(text)
    if unclosed:
        return "", [], "output ended inside <think> (truncated by max_tokens?)"
    calls: list[ToolCall] = []
    errors: list[str] = []
    outside: list[str] = []
    pos = 0
    for m in _TOOL_BLOCK.finditer(text):
        outside.append(text[pos : m.start()])
        pos = m.end()
        try:
            calls.append(_parse_block(m.group(1)))
        except ValueError as e:
            errors.append(str(e))
    tail = text[pos:]
    if "<tool_call>" in tail:  # opened but never closed, usually truncated output
        errors.append("<tool_call> block was not closed")
        tail = tail.split("<tool_call>", 1)[0]
    outside.append(tail)
    # Think blocks are only stripped outside tool calls, where their text is not data.
    content, unclosed = _split_think("".join(outside))
    if unclosed:
        errors.append("output ended inside <think>")
    if not content and not calls and not errors:
        errors.append("empty output")
    return content, calls, "; ".join(errors) or None
