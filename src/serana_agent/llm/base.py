"""Model interface shared by the local MLX model and API providers.

Contract: every provider returns tool calls already parsed into ToolCall objects.
Parsing of Qwen3 `<tool_call>` text happens inside the local provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

Role = Literal["system", "user", "assistant", "tool"]


class Message(TypedDict, total=False):
    role: Role
    content: str
    tool_calls: list[dict[str, Any]]  # assistant turns: [{"id", "name", "arguments"}]
    tool_call_id: str  # tool turns
    name: str  # tool turns: tool name


@dataclass
class ToolSpec:
    """A tool as shown to the planner. `input_schema` is JSON Schema (MCP inputSchema)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Set when the model emitted something that looked like a tool call but did not parse.
    parse_error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens


class ChatModel(Protocol):
    name: str

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        think: bool = False,
        max_tokens: int = 1024,
    ) -> ChatResult: ...
