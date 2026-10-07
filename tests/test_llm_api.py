import json
from types import SimpleNamespace as NS

import pytest

from serana_agent.llm.api import (
    AnthropicModel,
    OpenAIModel,
    from_anthropic_response,
    from_openai_response,
    to_anthropic_messages,
    to_anthropic_tools,
    to_openai_messages,
    to_openai_tools,
)
from serana_agent.llm.base import ToolSpec

TOOLS = [
    ToolSpec(
        "read_file", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}}}
    )
]
CONVERSATION = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "읽어줘"},
    {
        "role": "assistant",
        "content": "읽을게",
        "tool_calls": [
            {"id": "c1", "name": "read_file", "arguments": {"path": "a.txt"}},
            {"id": "c2", "name": "read_file", "arguments": {"path": "b.txt"}},
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "A"},
    {"role": "tool", "tool_call_id": "c2", "name": "read_file", "content": "B"},
    {"role": "user", "content": "고마워"},
]


def test_openai_messages():
    out = to_openai_messages(CONVERSATION)
    assert out[0] == {"role": "system", "content": "sys"}
    assistant = out[2]
    assert assistant["content"] == "읽을게"
    fn = assistant["tool_calls"][0]
    assert fn["id"] == "c1" and fn["type"] == "function"
    assert json.loads(fn["function"]["arguments"]) == {"path": "a.txt"}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "A"}
    assert out[5] == {"role": "user", "content": "고마워"}


def test_openai_assistant_without_text_has_null_content():
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "x", "name": "t", "arguments": {}}],
    }
    assert to_openai_messages([msg])[0]["content"] is None


def test_openai_tools():
    assert to_openai_tools(TOOLS) == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": TOOLS[0].input_schema,
            },
        }
    ]


def _openai_resp(content, tool_calls=None, finish_reason="stop"):
    return NS(
        choices=[
            NS(message=NS(content=content, tool_calls=tool_calls), finish_reason=finish_reason)
        ],
        usage=NS(prompt_tokens=7, completion_tokens=3),
    )


def test_openai_response_with_tool_call():
    call = NS(id="c9", function=NS(name="read_file", arguments='{"path": "a"}'))
    res = from_openai_response(_openai_resp(None, [call]))
    assert res.content == "" and res.parse_error is None
    assert (res.tool_calls[0].id, res.tool_calls[0].arguments) == ("c9", {"path": "a"})
    assert res.usage == {"prompt_tokens": 7, "completion_tokens": 3}


def test_openai_response_bad_arguments_sets_parse_error():
    call = NS(id="c9", function=NS(name="read_file", arguments="{nope"))
    res = from_openai_response(_openai_resp("x", [call]))
    assert res.tool_calls == [] and "read_file" in res.parse_error


def test_anthropic_messages_system_and_tool_results_merged():
    system, msgs = to_anthropic_messages(CONVERSATION)
    assert system == "sys"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "user"]
    assert msgs[1]["content"] == [
        {"type": "text", "text": "읽을게"},
        {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a.txt"}},
        {"type": "tool_use", "id": "c2", "name": "read_file", "input": {"path": "b.txt"}},
    ]
    assert msgs[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": "A"},
        {"type": "tool_result", "tool_use_id": "c2", "content": "B"},
    ]
    assert msgs[3] == {"role": "user", "content": "고마워"}


def test_anthropic_tools():
    assert to_anthropic_tools(TOOLS) == [
        {"name": "read_file", "description": "Read a file", "input_schema": TOOLS[0].input_schema}
    ]


def test_anthropic_response():
    resp = NS(
        content=[
            NS(type="text", text="읽을게"),
            NS(type="tool_use", id="tu1", name="read_file", input={"path": "a"}),
        ],
        usage=NS(input_tokens=11, output_tokens=5),
    )
    res = from_anthropic_response(resp)
    assert res.content == "읽을게"
    assert (res.tool_calls[0].id, res.tool_calls[0].arguments) == ("tu1", {"path": "a"})
    assert res.usage == {"prompt_tokens": 11, "completion_tokens": 5}


class _FakeOpenAI:
    def __init__(self):
        self.kwargs = None
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return _openai_resp("안녕")


def test_openai_model_chat_sends_converted_request():
    fake = _FakeOpenAI()
    model = OpenAIModel("gpt", "some-model", client=fake)
    res = model.chat(CONVERSATION[:2], TOOLS, max_tokens=50)
    assert res.content == "안녕"
    assert fake.kwargs["model"] == "some-model"
    assert fake.kwargs["max_completion_tokens"] == 50
    assert fake.kwargs["tools"][0]["function"]["name"] == "read_file"


def test_anthropic_model_chat_sends_converted_request():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return NS(content=[NS(type="text", text="ok")], usage=NS(input_tokens=1, output_tokens=1))

    model = AnthropicModel("sonnet", "m", client=NS(messages=NS(create=create)))
    model.chat(CONVERSATION[:2], TOOLS, max_tokens=64)
    assert seen["system"] == "sys" and seen["max_tokens"] == 64
    assert seen["messages"] == [{"role": "user", "content": "읽어줘"}]
    assert seen["tools"][0]["name"] == "read_file"


def test_missing_api_key_raises_without_network(monkeypatch):
    monkeypatch.delenv("MY_KEY_VAR", raising=False)
    model = OpenAIModel("gpt", "m", api_key_env="MY_KEY_VAR")
    with pytest.raises(RuntimeError, match="MY_KEY_VAR"):
        _ = model.client


def test_openai_truncation_reported():
    res = from_openai_response(_openai_resp("partial", finish_reason="length"))
    assert "truncated" in res.parse_error


def test_anthropic_truncation_reported():
    resp = NS(content=[NS(type="text", text="par")], stop_reason="max_tokens", usage=None)
    assert "truncated" in from_anthropic_response(resp).parse_error


def test_anthropic_skips_empty_assistant_message():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "b"},
    ]
    _, out = to_anthropic_messages(msgs)
    assert [m["role"] for m in out] == ["user", "user"]


def test_temperature_defaults_to_zero():
    fake = _FakeOpenAI()
    OpenAIModel("gpt", "m", client=fake).chat([{"role": "user", "content": "x"}])
    assert fake.kwargs["temperature"] == 0.0
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return NS(content=[], usage=None)

    AnthropicModel("a", "m", client=NS(messages=NS(create=create))).chat(
        [{"role": "user", "content": "x"}]
    )
    assert seen["temperature"] == 0.0
