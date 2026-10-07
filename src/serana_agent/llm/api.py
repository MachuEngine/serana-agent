"""API providers behind the ChatModel contract. Keys come from the env var named in config."""

from __future__ import annotations

import json
import os
from typing import Any

from langsmith import traceable

from serana_agent.llm.base import ChatResult, Message, ToolCall, ToolSpec


def _api_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"environment variable {env_var} is not set")
    return key


# --- OpenAI ---------------------------------------------------------------------------


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "tool":
            out.append(
                {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m.get("content", "")}
            )
        elif m["role"] == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                            },
                        }
                        for c in m["tool_calls"]
                    ],
                }
            )
        else:
            out.append({"role": m["role"], "content": m.get("content", "")})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def from_openai_response(resp: Any) -> ChatResult:
    msg = resp.choices[0].message
    calls: list[ToolCall] = []
    errors: list[str] = []
    for c in msg.tool_calls or []:
        try:
            args = json.loads(c.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            errors.append(f"tool call {c.function.name}: invalid arguments ({e})")
            continue
        calls.append(ToolCall(id=c.id, name=c.function.name, arguments=args))
    if resp.choices[0].finish_reason == "length":
        errors.append("response truncated (finish_reason=length)")
    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    return ChatResult(
        content=msg.content or "",
        tool_calls=calls,
        parse_error="; ".join(errors) or None,
        usage=usage,
    )


class OpenAIModel:
    def __init__(
        self,
        name: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        client: Any = None,
        temperature: float = 0.0,
    ):
        self.temperature = temperature
        self.name = name
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=_api_key(self.api_key_env), base_url=self.base_url)
        return self._client

    @traceable(run_type="llm", name="openai_chat")
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        think: bool = False,  # no portable switch for OpenAI models; ignored
        max_tokens: int = 1024,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "max_completion_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        return from_openai_response(self.client.chat.completions.create(**kwargs))


# --- Anthropic ------------------------------------------------------------------------


def to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Return (system, messages). Consecutive tool results merge into one user turn."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_parts.append(m.get("content", ""))
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m.get("content", ""),
            }
            last = out[-1] if out else None
            if last and last["role"] == "user" and isinstance(last["content"], list):
                last["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if not m.get("content") and not m.get("tool_calls"):
                continue  # Anthropic rejects empty content
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls", []):
                blocks.append(
                    {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": m.get("content", "")})
    return "\n\n".join(system_parts), out


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def from_anthropic_response(resp: Any) -> ChatResult:
    text: list[str] = []
    calls: list[ToolCall] = []
    for block in resp.content:
        if block.type == "text":
            text.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    usage = {}
    if getattr(resp, "usage", None):
        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        }
    truncated = getattr(resp, "stop_reason", None) == "max_tokens"
    return ChatResult(
        content="".join(text),
        tool_calls=calls,
        parse_error="response truncated (stop_reason=max_tokens)" if truncated else None,
        usage=usage,
    )


class AnthropicModel:
    def __init__(
        self,
        name: str,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        client: Any = None,
        temperature: float = 0.0,
    ):
        self.temperature = temperature
        self.name = name
        self.model = model
        self.api_key_env = api_key_env
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=_api_key(self.api_key_env))
        return self._client

    @traceable(run_type="llm", name="anthropic_chat")
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        think: bool = False,  # extended thinking is not enabled; ignored
        max_tokens: int = 1024,
    ) -> ChatResult:
        system, msgs = to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        return from_anthropic_response(self.client.messages.create(**kwargs))
