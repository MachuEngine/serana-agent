"""In-process MLX model: one 4bit base, LoRA layers toggled by scale per call.

The planner runs with the adapter off (scale 0) and the persona with it on. If switching
turns out not to match a merged model, `merged_persona_path` loads a separate merged model
for the persona role instead (design section 4 fallback, costs a second copy in memory).

The adapter state lives on the shared model, so generation is serialized with a lock and the
previous scale is restored afterwards. Planner and persona calls must not run concurrently
(the lock makes them wait, not interleave).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langsmith import traceable

from serana_agent.llm.api import to_openai_tools
from serana_agent.llm.base import ChatResult, Message, ToolSpec
from serana_agent.llm.toolcall import parse_output


@dataclass
class Generation:
    text: str
    token_ids: list[int]
    prompt_tokens: int


def to_template_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Message -> the dict shape Qwen3's chat template expects (tool_calls[].function)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
        if m.get("tool_calls"):
            d["tool_calls"] = [
                {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in m["tool_calls"]
            ]
        out.append(d)
    return out


class LocalModel:
    def __init__(
        self,
        base_path: str,
        adapter_path: str | None = None,
        *,
        merged_persona_path: str | None = None,
        temperature: float = 0.0,
    ):
        from mlx_lm import load
        from mlx_lm.tuner.lora import LoRALinear

        self.temperature = temperature
        self.model, self.tokenizer = load(base_path, adapter_path=adapter_path)
        self._lora = [m for _, m in self.model.named_modules() if isinstance(m, LoRALinear)]
        self.adapter_scale = self._lora[0].scale if self._lora else 0.0
        # Template shipped with the adapter; None means "use the tokenizer's own (Qwen3)".
        self.persona_template: str | None = None
        if adapter_path and (Path(adapter_path) / "chat_template.jinja").exists():
            self.persona_template = (Path(adapter_path) / "chat_template.jinja").read_text()
        self._lock = threading.Lock()
        self.merged = None
        if merged_persona_path:
            self.merged = load(merged_persona_path)

    @contextmanager
    def _adapter(self, on: bool) -> Iterator[None]:
        if on and not self._lora:
            raise ValueError("adapter requested but no LoRA layers are loaded (adapter_path=None)")
        previous = [layer.scale for layer in self._lora]
        for layer in self._lora:
            layer.scale = self.adapter_scale if on else 0.0
        try:
            yield
        finally:
            for layer, scale in zip(self._lora, previous, strict=True):
                layer.scale = scale

    def _uses_merged(self, persona: bool, adapter: bool) -> bool:
        return bool(self.merged) and persona and adapter

    def render(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        persona: bool = False,
        adapter: bool = True,
        think: bool = False,
    ) -> str:
        tokenizer = self.merged[1] if self._uses_merged(persona, adapter) else self.tokenizer
        template = self.persona_template if persona else None
        return tokenizer.apply_chat_template(
            to_template_messages(messages),
            tools=to_openai_tools(tools) if tools else None,
            chat_template=template,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=think,
        )

    def generate(
        self, prompt: str, max_tokens: int, *, adapter: bool, use_merged: bool = False
    ) -> Generation:
        from mlx_lm.sample_utils import make_sampler

        kwargs: dict[str, Any] = {}
        if self.temperature > 0:
            kwargs["sampler"] = make_sampler(temp=self.temperature)
        with self._lock:
            if use_merged and self.merged:
                model, tokenizer = self.merged
                scope = nullcontext()
            else:
                model, tokenizer = self.model, self.tokenizer
                scope = self._adapter(adapter)
            with scope:
                return self._run(model, tokenizer, prompt, max_tokens, kwargs)

    @staticmethod
    def _run(model, tokenizer, prompt: str, max_tokens: int, kwargs: dict[str, Any]) -> Generation:
        from mlx_lm import stream_generate

        text, ids, prompt_tokens = "", [], 0
        for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, **kwargs):
            text += r.text
            ids.append(r.token)
            prompt_tokens = r.prompt_tokens
        return Generation(text, ids, prompt_tokens)

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        persona: bool,
        adapter: bool,
        think: bool,
        max_tokens: int,
    ) -> ChatResult:
        prompt = self.render(messages, tools, persona=persona, adapter=adapter, think=think)
        gen = self.generate(
            prompt, max_tokens, adapter=adapter, use_merged=self._uses_merged(persona, adapter)
        )
        content, calls, error = parse_output(gen.text)
        return ChatResult(
            content=content,
            tool_calls=calls,
            parse_error=error,
            usage={"prompt_tokens": gen.prompt_tokens, "completion_tokens": len(gen.token_ids)},
        )

    def planner(self, *, adapter: bool = False) -> LocalRole:
        return LocalRole(self, "local-planner", persona=False, adapter=adapter)

    def persona(self, *, adapter: bool = True) -> LocalRole:
        return LocalRole(self, "local-persona", persona=True, adapter=adapter)


class LocalRole:
    """ChatModel view of LocalModel. Persona never thinks and never gets tools."""

    def __init__(self, lm: LocalModel, name: str, *, persona: bool, adapter: bool):
        self.lm = lm
        self.name = name
        self.persona = persona
        self.adapter = adapter

    @traceable(run_type="llm", name="local_chat")
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        think: bool = False,
        max_tokens: int = 1024,
    ) -> ChatResult:
        return self.lm.chat(
            messages,
            None if self.persona else tools,
            persona=self.persona,
            adapter=self.adapter,
            think=False if self.persona else think,
            max_tokens=max_tokens,
        )
