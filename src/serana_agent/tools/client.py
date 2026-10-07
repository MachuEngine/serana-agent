from __future__ import annotations

import copy
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Literal

import anyio
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from serana_agent.llm.base import ToolSpec
from serana_agent.tools.launch import server_params
from serana_agent.tools.protocol import CONFIRM_ARG, ToolOutcome


class ToolClient:
    def __init__(self, root: Path, mode: Literal["host", "docker"] = "host", timeout: float = 30.0):
        self._timeout = timeout
        self._params = server_params(root, mode)
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def __aenter__(self) -> ToolClient:
        try:
            read, write = await self._stack.enter_async_context(stdio_client(self._params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()
        self._session = None

    async def list_tools(self) -> list[ToolSpec]:
        assert self._session is not None, "use ToolClient as an async context manager"
        result = await self._session.list_tools()
        specs = []
        for t in result.tools:
            schema = copy.deepcopy(t.input_schema)
            schema.get("properties", {}).pop(CONFIRM_ARG, None)
            if CONFIRM_ARG in schema.get("required", []):
                schema["required"].remove(CONFIRM_ARG)
            specs.append(ToolSpec(t.name, t.description or "", schema))
        return specs

    async def call(
        self, name: str, arguments: dict | None = None, confirmed: bool = False
    ) -> ToolOutcome:
        assert self._session is not None, "use ToolClient as an async context manager"
        args = dict(arguments or {})
        args.pop(CONFIRM_ARG, None)  # only the caller's `confirmed` flag may approve
        if confirmed:
            args[CONFIRM_ARG] = True
        try:
            with anyio.fail_after(self._timeout):
                result = await self._session.call_tool(name, args)
        except TimeoutError:
            return ToolOutcome("error", error=f"tool call timed out after {self._timeout}s")
        except Exception as e:
            return ToolOutcome("error", error=f"tool call failed: {e}")
        text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        if result.is_error:
            return ToolOutcome("error", error=text or "tool returned an error")
        try:
            return ToolOutcome.from_json(text)
        except (ValueError, TypeError, KeyError):
            return ToolOutcome("error", error=f"tool returned a non-JSON response: {text[:200]}")
