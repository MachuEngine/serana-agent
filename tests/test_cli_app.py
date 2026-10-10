import io
import sys
import types
from pathlib import Path

import pytest
from rich.console import Console
from test_agent_fakes import FakeModel, call, final
from typer.testing import CliRunner

from serana_agent import tracing
from serana_agent.agent.gate import ScriptedApprover
from serana_agent.cli import app as app_mod
from serana_agent.cli.session import Session
from serana_agent.config import Config
from serana_agent.llm.base import ChatResult
from serana_agent.memory.types import MemoryItem, Skill, SkillStep
from serana_agent.tools.client import ToolClient

runner = CliRunner()


class FakeRegistry:
    models = {
        "serana": {"provider": "local"},
        "sonnet": {"provider": "anthropic"},
        "broken": {"provider": "openai"},
    }

    def __init__(self, planner_script=(), persona_script=()):
        self.planner = FakeModel("local-planner", planner_script)
        self.persona = FakeModel("local-persona", persona_script or [ChatResult("Serana says hi")])
        self.api = FakeModel("sonnet", [])
        self.pairs = []

    def names(self):
        return list(self.models)

    def pair(self, name, *, full=False):
        self.pairs.append((name, full))
        if name == "broken":
            raise RuntimeError("OPENAI_API_KEY is not set")
        if name == "serana":
            return self.planner, self.persona
        return (self.api, self.api) if full else (self.api, self.persona)


class StubMemory:
    def __init__(self, items=()):
        self.items = [MemoryItem(str(i), t) for i, t in enumerate(items)]

    def add(self, text, metadata=None):
        item = MemoryItem(str(len(self.items)), text, metadata or {})
        self.items.append(item)
        return item

    def search(self, query, k=5):
        return [m for m in self.items if query.lower() in m.text.lower()][:k]

    def is_duplicate(self, text):
        return any(m.text == text for m in self.items)

    def all(self):
        return list(self.items)


class StubSkills:
    read_only = False

    def __init__(self, skills=()):
        self.skills = list(skills)

    def add(self, name, description, steps):
        skill = Skill("new", name, description, steps)
        self.skills.append(skill)
        return skill

    def search(self, query, k=3):
        return []

    def record_outcome(self, skill_id, success):
        pass

    def all(self):
        return list(self.skills)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERANA_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    tracing._enabled = False
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("hello")
    memory, skills = StubMemory(), StubSkills()
    registry = FakeRegistry()
    monkeypatch.setattr(app_mod, "build_registry", lambda cfg: registry)
    monkeypatch.setattr(app_mod, "open_memory", lambda cfg: memory)
    monkeypatch.setattr(app_mod, "open_skills", lambda cfg, read_only=False: skills)
    yield types.SimpleNamespace(root=root, memory=memory, skills=skills, registry=registry)
    tracing._enabled = False


def base_args(env):
    return ["--root", str(env.root), "--sandbox-mode", "host"]


def test_run_command(env):
    env.registry.planner.script = [call("read_file", path="a.txt"), final("it says hello")]
    result = runner.invoke(app_mod.app, ["run", "read a.txt", *base_args(env)])
    assert result.exit_code == 0, result.output
    assert "Serana says hi" in result.output
    summary = env.registry.persona.calls[0]["messages"][-1]["content"]
    assert "read_file" in summary
    assert not tracing.tracing_enabled()


def test_run_exit_code_when_stopped(env):
    env.registry.planner.script = [call("nope"), call("nope")]
    result = runner.invoke(app_mod.app, ["run", "x", *base_args(env)])
    assert result.exit_code == 1 and "invalid_tool_call" in result.output


def test_run_offers_skill_after_two_tool_calls(env):
    env.registry.planner.script = [
        call("list_dir", path="."),
        call("read_file", path="a.txt"),
        final(),
        ChatResult("Listed and read."),  # reporter
        ChatResult('{"name": "peek", "description": "look around", "placeholders": {}}'),
    ]
    result = runner.invoke(app_mod.app, ["run", "look", *base_args(env)], input="y\n")
    assert result.exit_code == 0, result.output
    assert [s.name for s in env.skills.skills] == ["peek"]
    assert [s.tool for s in env.skills.skills[0].steps] == ["list_dir", "read_file"]


def test_run_skill_declined_by_default(env):
    env.registry.planner.script = [
        call("list_dir", path="."),
        call("read_file", path="a.txt"),
        final(),
    ]
    result = runner.invoke(app_mod.app, ["run", "look", *base_args(env)], input="\n")
    assert result.exit_code == 0 and env.skills.skills == []


def test_run_trace_flag_asks_first(env, monkeypatch):
    env.registry.planner.script = [final()]
    result = runner.invoke(app_mod.app, ["run", "x", "--trace", *base_args(env)], input="n\n")
    assert "LangSmith" in result.output and not tracing.tracing_enabled()
    result = runner.invoke(app_mod.app, ["run", "x", "--trace", *base_args(env)], input="y\n")
    assert "LANGSMITH_API_KEY" in result.output and not tracing.tracing_enabled()


def test_repl_slash_commands_via_cli(env):
    env.memory.add("Likes tea")
    result = runner.invoke(
        app_mod.app, base_args(env), input="/model\n/memory\n/skills\n/bogus\n/exit\n"
    )
    assert result.exit_code == 0, result.output
    assert "현재 모델: serana (local)" in result.output
    assert "Likes tea" in result.output
    assert "저장된 스킬이 없습니다" in result.output
    assert "알 수 없는 명령: /bogus" in result.output


def test_repl_reflects_on_exit(env):
    env.registry.planner.script = [final("ok"), ChatResult('["Prefers short answers"]')]
    result = runner.invoke(app_mod.app, base_args(env), input="hello\n/exit\n")
    assert result.exit_code == 0, result.output
    assert [m.text for m in env.memory.items] == ["Prefers short answers"]
    assert "서라나 [local]" in result.output


def fake_eval_modules(monkeypatch, seen):
    async def fake_run_eval(
        tasks_dir, agent_factory, *, out_dir, sandbox_mode="host", skills=None,
        task_ids=None, meta=None,
    ):  # fmt: skip
        seen.update(
            tasks_dir=tasks_dir, out_dir=out_dir, mode=sandbox_mode, skills=skills,
            task_ids=task_ids, factory=agent_factory, meta=meta,
        )  # fmt: skip
        return "SUMMARY"

    def print_table(summary, console=None):
        seen["table"] = summary
        console.print("LEVEL-TABLE")

    def sync(tasks_dir, summary):
        seen["synced"] = summary
        return "exp-1"

    for name, attrs in {
        "runner": {"run_eval": fake_run_eval},
        "report": {"print_table": print_table},
        "langsmith_sync": {"sync_to_langsmith": sync},
    }.items():
        module = types.ModuleType(f"serana_agent.eval.{name}")
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, f"serana_agent.eval.{name}", module)


def test_eval_command_calls_run_eval(env, monkeypatch, tmp_path):
    seen = {}
    fake_eval_modules(monkeypatch, seen)
    out = tmp_path / "out"
    result = runner.invoke(
        app_mod.app,
        ["--sandbox-mode", "host", "eval", "tasks", "--model", "sonnet", "--out", str(out),
         "--task-id", "t1", "--task-id", "t2"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert seen["tasks_dir"] == Path("tasks") and seen["out_dir"] == out
    assert seen["mode"] == "host" and seen["task_ids"] == ["t1", "t2"]
    assert seen["skills"] is None  # skills are off unless --skills-store is given
    assert seen["meta"] == {
        "planner_model": "sonnet", "persona_model": "local-persona", "step_limit": 20,
        "sandbox_mode": "host", "skills": False,
    }  # fmt: skip
    assert env.registry.pairs == [("sonnet", False)]
    assert "LANGSMITH_API_KEY" in result.output  # tracing wanted by default, but no key
    assert "LEVEL-TABLE" in result.output and "synced" not in seen

    agent = seen["factory"]("tools", "approver", None)
    assert agent.planner is env.registry.api and agent.memory is None and agent.skills is None
    assert agent.approver == "approver"


def test_eval_skills_store_and_langsmith_sync(env, monkeypatch, tmp_path):
    seen, opened = {}, {}
    fake_eval_modules(monkeypatch, seen)
    monkeypatch.setenv("LANGSMITH_API_KEY", "dummy")

    def open_skills(cfg, read_only=False, path=None):
        opened.update(read_only=read_only, path=path)
        return env.skills

    monkeypatch.setattr(app_mod, "open_skills", open_skills)
    result = runner.invoke(
        app_mod.app,
        ["eval", "tasks", "--sandbox-mode", "host", "--skills-store", "s.json",
         "--out", str(tmp_path / "o")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert opened == {"read_only": True, "path": Path("s.json")}
    assert seen["skills"] is env.skills and seen["meta"]["skills"] is True
    assert seen["synced"] == "SUMMARY" and "exp-1" in result.output


def test_options_before_subcommand_are_honored(env):
    env.registry.planner.script = [call("read_file", path="a.txt"), final()]
    result = runner.invoke(
        app_mod.app, ["--root", str(env.root), "--sandbox-mode", "host", "run", "read it"]
    )
    assert result.exit_code == 0, result.output
    assert "result: hello" in env.registry.persona.calls[0]["messages"][-1]["content"]


def test_trace_flag_before_subcommand_asks_for_confirmation(env):
    env.registry.planner.script = [final()]
    result = runner.invoke(
        app_mod.app, ["--trace", "--root", str(env.root), "--sandbox-mode", "host", "run", "x"],
        input="n\n",
    )  # fmt: skip
    assert result.exit_code == 0 and "LangSmith" in result.output


@pytest.mark.parametrize("args", [["run", "hi"], [], ["eval", "tasks"]])
def test_docker_preflight_failure_exits_cleanly(env, monkeypatch, args):
    monkeypatch.setattr(app_mod, "docker_problem", lambda: "Docker 데몬에 연결하지 못했습니다.")
    result = runner.invoke(
        app_mod.app, [*args, "--root", str(env.root)] if args != ["eval", "tasks"] else args
    )
    assert result.exit_code == 1
    assert "Docker 데몬" in result.output and "--sandbox-mode host" in result.output
    assert "Traceback" not in result.output and env.registry.pairs == []


def test_docker_preflight_skipped_in_host_mode(env, monkeypatch):
    monkeypatch.setattr(app_mod, "docker_problem", lambda: pytest.fail("should not be called"))
    env.registry.planner.script = [final()]
    result = runner.invoke(app_mod.app, ["run", "x", *base_args(env)])
    assert result.exit_code == 0


def test_docker_problem_checks(monkeypatch):
    results = {}

    def fake_run(cmd, **kw):
        outcome = results[tuple(cmd[:2])]
        if isinstance(outcome, Exception):
            raise outcome
        return types.SimpleNamespace(returncode=outcome)

    monkeypatch.setattr(app_mod.subprocess, "run", fake_run)
    results[("docker", "info")] = 1
    results[("docker", "image")] = 0
    assert "Docker Desktop" in app_mod.docker_problem()
    results[("docker", "info")] = FileNotFoundError()
    assert "Docker Desktop" in app_mod.docker_problem()
    results[("docker", "info")] = 0
    results[("docker", "image")] = 1
    assert "docker build -f docker/Dockerfile -t serana-mcp:latest" in app_mod.docker_problem()
    results[("docker", "image")] = 0
    assert app_mod.docker_problem() is None


def test_env_langsmith_tracing_cannot_turn_tracing_on_outside_agent_run(env, monkeypatch):
    from langsmith.utils import tracing_is_enabled

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    states = []
    for model in (env.registry.planner, env.registry.persona):
        original = model.chat

        def chat(*a, _orig=original, **k):
            states.append(tracing_is_enabled())
            return _orig(*a, **k)

        model.chat = chat
    env.registry.planner.script = [
        call("list_dir", path="."),
        call("read_file", path="a.txt"),
        final(),
        ChatResult("Listed and read."),  # reporter
        ChatResult('{"name": "peek", "description": "look", "placeholders": {}}'),  # skill summary
    ]
    result = runner.invoke(app_mod.app, ["run", "look", *base_args(env)], input="y\n")
    assert result.exit_code == 0, result.output
    assert len(states) == 6 and not any(states)  # planner x3, reporter, persona, skill extraction
    assert not tracing.tracing_enabled()


# -- slash commands on a Session --


def make_session(env, confirm_answers=()):
    answers = list(confirm_answers)
    prompts = []

    def confirm(prompt):
        prompts.append(prompt)
        return answers.pop(0)

    out = io.StringIO()
    console = Console(file=out, width=120)
    session = Session(
        Config(home=Path("/nonexistent")), env.registry, None, None, confirm, console,
        memory=env.memory, skills=env.skills,
    )  # fmt: skip
    return session, out, prompts


async def test_model_command_and_privacy_gate(env):
    session, out, prompts = make_session(env, [False, True, True, True])
    await session.handle("/model list")
    assert "- sonnet (anthropic)" in out.getvalue()

    await session.handle("/model sonnet")  # first API switch: gate, denied
    assert len(prompts) == 1 and session.label == "local"
    assert "툴 결과" in out.getvalue() and "장기 기억" in out.getvalue()

    await session.handle("/model sonnet")  # gate again, accepted
    assert session.label == "sonnet" and session.agent.planner is env.registry.api
    assert session.agent.persona is env.registry.persona

    await session.handle("/model serana")
    await session.handle("/model sonnet")  # same provider, not full: no new prompt
    assert len(prompts) == 2
    await session.handle("/model sonnet --full")  # persona input leaves the machine too: ask
    assert len(prompts) == 3 and session.label == "sonnet/full"
    await session.handle("/model sonnet --full")
    assert len(prompts) == 3
    await session.handle("/model broken")  # different provider: ask (fake refuses to build it)
    assert len(prompts) == 4
    assert session.agent.persona is env.registry.api
    assert session.label == "sonnet/full"
    await session.handle("/model")
    assert "현재 모델: sonnet (anthropic) --full" in out.getvalue()


async def test_model_command_errors(env):
    session, out, prompts = make_session(env, [True])
    await session.handle("/model nope")
    assert "사용법" in out.getvalue() and prompts == []
    await session.handle("/model serana --full")
    assert session.label == "local"  # registry would refuse; here the fake accepts, label stays
    await session.handle("/model broken")
    assert "OPENAI_API_KEY is not set" in out.getvalue() and session.label == "local"


async def test_memory_and_skills_commands(env):
    env.memory.add("Likes tea")
    env.memory.add("Uses vim")
    env.skills.skills.append(Skill("s", "tidy", "tidy a folder", [SkillStep("list_dir", {})]))
    session, out, _ = make_session(env)
    await session.handle("/memory vim")
    assert "Uses vim" in out.getvalue() and "Likes tea" not in out.getvalue()
    await session.handle("/skills")
    assert "tidy" in out.getvalue()
    assert await session.handle("/exit") is False
    assert await session.handle("   ") is True


async def test_trace_command_shows_last_run(env):
    env.registry.planner.script = [call("read_file", path="a.txt"), final()]
    out = io.StringIO()
    async with ToolClient(env.root, "host") as tools:
        session = Session(
            Config(home=Path("/nonexistent")), env.registry, tools, ScriptedApprover([]),
            lambda p: False,
            Console(file=out, width=200), memory=env.memory, skills=env.skills,
        )  # fmt: skip
        await session.handle("/trace")
        assert "아직 실행한 작업이 없습니다" in out.getvalue()
        await session.handle("read a.txt")
        await session.handle("/trace")
    text = out.getvalue()
    assert "read_file {'path': 'a.txt'} -> ok" in text and "종료: final" in text
    assert "planner=local-planner" in text


async def test_history_keeps_tool_record_and_skips_failure_placeholder(env):
    env.registry.planner.script = [
        call("read_file", path="a.txt"),
        final(),
        ChatResult("Read a.txt: one short line."),  # reporter
        call("read_file", path="a.txt"),
        final(),
        ChatResult("Read a.txt again."),  # reporter
        call("read_file", path="a.txt"),
        final(),
        ChatResult("Read a.txt a third time."),  # reporter
    ]
    out = io.StringIO()
    async with ToolClient(env.root, "host") as tools:
        session = Session(
            Config(home=Path("/nonexistent")), env.registry, tools, ScriptedApprover([]),
            lambda p: False, Console(file=out, width=200),
        )  # fmt: skip
        await session.handle("read a.txt")
        assert session.history[1]["content"].endswith("[tools run this turn] read_file(a.txt) ok")
        assert "hello" not in session.history[1]["content"]
        assert "[planner notes] Read a.txt: one short line." in session.history[1]["content"]

        class Boom(FakeModel):
            def chat(self, *a, **k):
                raise RuntimeError("persona down")

        session.agent.persona = Boom("p", [])
        await session.handle("list")
        await session.handle("again")
    contents = [m["content"] for m in session.history]
    assert not any("응답을 만들지 못했습니다" in c for c in contents)
    # failed persona turns store no reply, only the record lines
    assert [m["role"] for m in session.history] == ["user", "assistant"] * 3
    assert session.history[3]["content"].startswith("[planner notes]")
    planner_msgs = env.registry.planner.calls[3]["messages"]
    assert "[tools run this turn] read_file(a.txt) ok" in str(planner_msgs)


async def test_end_reflects_on_history_without_record_lines(env, monkeypatch):
    env.registry.planner.script = [
        call("read_file", path="a.txt"),
        final("planner claim"),
        ChatResult("Read a.txt: one short line."),  # reporter
    ]
    seen = []

    def fake_reflect(history, model, memory, meta=None):
        seen.append(history)
        return []

    from serana_agent.cli import session as session_mod

    monkeypatch.setattr(session_mod, "reflect", fake_reflect)
    async with ToolClient(env.root, "host") as tools:
        session = Session(
            Config(home=Path("/nonexistent")), env.registry, tools, ScriptedApprover([]),
            lambda p: False, Console(file=io.StringIO(), width=200), memory=env.memory,
        )  # fmt: skip
        await session.handle("read a.txt")
        stored = session.history[1]["content"]
        assert "[planner notes] Read a.txt: one short line." in stored
        assert "planner claim" not in stored
        session.history.append({"role": "user", "content": "x"})
        session.history.append({"role": "assistant", "content": "[tools run this turn] a(b) ok"})
        await session.end()
    assert [(m["role"], m["content"]) for m in seen[0]] == [
        ("user", "read a.txt"),
        ("assistant", "Serana says hi"),
        ("user", "x"),
    ]
