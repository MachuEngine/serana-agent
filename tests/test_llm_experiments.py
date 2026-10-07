import importlib.util
import json
from pathlib import Path

from serana_agent.llm.base import ChatResult, ToolCall

exp1_path = Path(__file__).parents[1] / "experiments" / "exp1_toolcall_smoke.py"
spec = importlib.util.spec_from_file_location("exp1", exp1_path)
exp1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp1)

TOOLS = exp1.load_tools()


def test_data_files_are_consistent():
    prompts = exp1.load_prompts()
    names = {t.name for t in TOOLS}
    assert len(prompts) == 20
    assert all(p["expected_tool"] in names for p in prompts)
    assert len({p["id"] for p in prompts}) == 20


def _score(item, calls, parse_error=None):
    return exp1.score(item, ChatResult("", calls, parse_error), TOOLS)


ITEM = {"expected_tool": "read_file", "expected_args": {"path": "src/main.py"}}


def test_score_perfect_call_with_path_normalization():
    s = _score(ITEM, [ToolCall("1", "read_file", {"path": "./src/main.py"})])
    assert s == {"format_ok": True, "tool_ok": True, "args_valid": True, "args_match": True}


def test_score_wrong_tool_and_missing_call():
    s = _score(ITEM, [ToolCall("1", "list_dir", {"path": "."})])
    assert s["format_ok"] and not s["tool_ok"] and not s["args_match"]
    assert not any(_score(ITEM, []).values())


def test_score_invalid_args_and_parse_error():
    s = _score(ITEM, [ToolCall("1", "read_file", {"file": "x"})])
    assert s["tool_ok"] and not s["args_valid"] and not s["args_match"]
    assert not _score(ITEM, [], "broken")["format_ok"]


def test_args_match_contains_for_free_text():
    assert exp1.args_match({"content": "hello"}, {"content": "say hello world"})
    assert not exp1.args_match({"old": "a"}, {"old": "ab"})


def test_summarize():
    rows = [
        {
            "condition": "c",
            "format_ok": True,
            "tool_ok": True,
            "args_valid": True,
            "args_match": False,
        },
        {
            "condition": "c",
            "format_ok": False,
            "tool_ok": False,
            "args_valid": False,
            "args_match": False,
        },
    ]
    assert exp1.summarize(rows)["c"]["format_ok"] == 0.5


async def test_tools_json_matches_live_server_schemas():
    """Fails when the server's tools change; fix with experiments/gen_tools_json.py."""
    spec = importlib.util.spec_from_file_location(
        "gen_tools_json", exp1_path.with_name("gen_tools_json.py")
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    live = await gen.fetch_tools()
    stored = json.loads(gen.TOOLS_JSON.read_text())
    assert stored == live
