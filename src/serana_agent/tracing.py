"""LangSmith tracing switch. Off means `traced` functions run untouched, with no network use.

The design (section 9) allows tracing by default only for `serana eval` (synthetic data);
chat and `run` need an explicit opt-in because traces carry conversations and memory excerpts.
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from langsmith import traceable, tracing_context

API_KEY_ENV = "LANGSMITH_API_KEY"
DEFAULT_PROJECT = "serana-agent"

_enabled = False


def tracing_enabled() -> bool:
    return _enabled


def set_tracing(enabled: bool, project: str = DEFAULT_PROJECT) -> bool:
    """Turn tracing on or off. Turning on without an API key fails and returns False."""
    global _enabled
    if enabled and not os.environ.get(API_KEY_ENV):
        enabled = False
    if enabled:
        os.environ.setdefault("LANGSMITH_PROJECT", project)
    _enabled = enabled
    return _enabled


def tracing_scope() -> AbstractContextManager[None]:
    """Pin tracing to the switch for everything inside, including code outside `traced`
    functions, so LANGSMITH_TRACING in the user's shell cannot turn it on silently."""
    return tracing_context(enabled=_enabled)


def traced(name: str, run_type: str = "chain") -> Callable[[Callable], Callable]:
    """Decorator: record the call in LangSmith while enabled.

    The switch is read per call and applied through `tracing_context`, so it also governs
    `@traceable` code called inside (the model adapters) and survives langsmith's env caching.
    """

    def decorate(fn: Callable) -> Callable:
        inner = traceable(name=name, run_type=run_type)(fn)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _enabled:
                    with tracing_context(enabled=False):
                        return await fn(*args, **kwargs)
                with tracing_context(enabled=True):
                    return await inner(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _enabled:
                with tracing_context(enabled=False):
                    return fn(*args, **kwargs)
            with tracing_context(enabled=True):
                return inner(*args, **kwargs)

        return wrapper

    return decorate
