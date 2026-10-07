"""Scripted fake ChatModel shared by the agent and CLI tests."""

from serana_agent.llm.base import ChatResult, ToolCall


class FakeModel:
    """Returns scripted ChatResults in order and records every call's messages."""

    def __init__(self, name, script):
        self.name = name
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None, *, think=False, max_tokens=1024):
        self.calls.append({"messages": messages, "tools": tools, "think": think})
        return self.script.pop(0) if self.script else ChatResult("done")


def call(name, **arguments):
    return ChatResult("", [ToolCall(f"c-{name}", name, arguments)])


def final(text="all done"):
    return ChatResult(text)
