"""`serana` command line: chat REPL, `run`, `eval`."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from serana_agent.agent.audit import AuditLog
from serana_agent.agent.gate import TerminalApprover
from serana_agent.agent.loop import Agent
from serana_agent.cli.session import Session, new_audit
from serana_agent.config import Config, load_config
from serana_agent.llm.registry import ModelRegistry
from serana_agent.memory.types import MemoryStore, SkillStore
from serana_agent.tools.client import ToolClient
from serana_agent.tools.launch import DOCKER_IMAGE
from serana_agent.tracing import API_KEY_ENV, set_tracing, tracing_scope

app = typer.Typer(add_completion=False, help="Serana: 로컬 컴패니언 에이전트")

ConfigOpt = Annotated[Path | None, typer.Option("--config", help="serana.toml 경로")]
RootOpt = Annotated[Path | None, typer.Option("--root", help="샌드박스 루트 폴더")]
ModeOpt = Annotated[
    str | None, typer.Option("--sandbox-mode", help="host 또는 docker (기본: 설정값)")
]
TraceOpt = Annotated[bool, typer.Option("--trace", help="LangSmith 추적 켜기 (확인 후)")]


@dataclass
class GlobalOpts:
    """Options given before the subcommand (`serana --root X run ...`); subcommands may override."""

    config: Path | None = None
    root: Path | None = None
    sandbox_mode: str | None = None
    trace: bool = False


def _opts(ctx: typer.Context) -> GlobalOpts:
    return ctx.obj if isinstance(ctx.obj, GlobalOpts) else GlobalOpts()


# -- construction helpers (tests replace these) --


def build_registry(cfg: Config) -> ModelRegistry:
    return ModelRegistry(cfg.models, cfg.models_dir)


def open_memory(cfg: Config) -> MemoryStore:
    from serana_agent.memory.chroma_store import ChromaMemoryStore

    return ChromaMemoryStore(cfg.chroma_path)


def open_skills(cfg: Config, read_only: bool = False, path: Path | None = None) -> SkillStore:
    from serana_agent.skills.store import JsonSkillStore

    return JsonSkillStore(path or cfg.skills_path, read_only=read_only)


def ask_yes_no(console: Console, prompt: str) -> bool:
    try:
        return Confirm.ask(prompt, default=False, console=console)
    except EOFError:
        return False


def _mode(cfg: Config, mode: str | None) -> Literal["host", "docker"]:
    mode = mode or cfg.sandbox_mode
    if mode not in ("host", "docker"):
        raise typer.BadParameter("host 또는 docker만 쓸 수 있습니다", param_hint="--sandbox-mode")
    return mode  # type: ignore[return-value]


def docker_problem() -> str | None:
    """Why the docker sandbox cannot start, or None when it can."""
    checks = [
        (["docker", "info"], "Docker 데몬에 연결하지 못했습니다. Docker Desktop을 실행하세요."),
        (
            ["docker", "image", "inspect", DOCKER_IMAGE],
            f"{DOCKER_IMAGE} 이미지가 없습니다. 프로젝트 루트에서 "
            f"`docker build -f docker/Dockerfile -t {DOCKER_IMAGE} .` 로 빌드하세요.",
        ),
    ]
    for cmd, message in checks:
        try:
            ok = subprocess.run(cmd, capture_output=True, timeout=20).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if not ok:
            return message
    return None


def _preflight(mode: str, console: Console) -> None:
    if mode != "docker":
        return
    problem = docker_problem()
    if problem:
        console.print(
            f"[red]{problem}[/red]\nDocker 없이 쓰려면 --sandbox-mode host 를 지정하세요."
        )
        raise typer.Exit(1)


def _maybe_enable_tracing(cfg: Config, console: Console, requested: bool) -> None:
    """Chat and run send nothing to LangSmith unless the user opts in here."""
    if not requested:
        return
    console.print(
        "[yellow]추적을 켜면 대화, 파일 내용, 장기 기억 발췌가 "
        "LangSmith 클라우드로 전송됩니다.[/yellow]"
    )
    if not ask_yes_no(console, "추적을 켤까요?"):
        return
    if not set_tracing(True, cfg.langsmith_project):
        console.print(f"[red]{API_KEY_ENV}가 없어 추적을 켜지 못했습니다.[/red]")


def _make_session(cfg: Config, tools: ToolClient, console: Console, read_only_skills=False):
    return Session(
        cfg,
        build_registry(cfg),
        tools,
        TerminalApprover(console),
        lambda prompt: ask_yes_no(console, prompt),
        console,
        memory=open_memory(cfg),
        skills=open_skills(cfg, read_only=read_only_skills),
        audit=new_audit(cfg),
    )


def _workspace(cfg: Config, root: Path | None) -> Path:
    path = (root or cfg.workspace).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


# -- chat --


def _banner(console: Console) -> None:
    import pyfiglet

    logo = pyfiglet.figlet_format("Serana", font="slant")
    console.print(Panel(logo.rstrip(), subtitle="/help 로 명령 보기", border_style="cyan"))


def _read_line(label: str, history_file: Path) -> str:
    if not sys.stdin.isatty():
        return input(f"서라나 [{label}] > ")
    from prompt_toolkit import prompt
    from prompt_toolkit.history import FileHistory

    history_file.parent.mkdir(parents=True, exist_ok=True)
    return prompt(f"서라나 [{label}] > ", history=FileHistory(str(history_file)))


async def _repl(cfg: Config, root: Path, mode: Literal["host", "docker"], console: Console):
    _banner(console)
    async with ToolClient(root, mode) as tools:
        session = _make_session(cfg, tools, console)
        try:
            while True:
                try:
                    line = await asyncio.to_thread(_read_line, session.label, cfg.home / "history")
                except (EOFError, KeyboardInterrupt):
                    break
                if not await session.handle(line):
                    break
        finally:
            await session.end()


@app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    config: ConfigOpt = None,
    root: RootOpt = None,
    sandbox_mode: ModeOpt = None,
    trace: TraceOpt = False,
):
    """인자 없이 실행하면 대화 모드를 시작한다."""
    ctx.obj = GlobalOpts(config, root, sandbox_mode, trace)
    if ctx.invoked_subcommand is not None:
        return
    cfg = load_config(config)
    console = Console()
    mode = _mode(cfg, sandbox_mode)
    _preflight(mode, console)
    _maybe_enable_tracing(cfg, console, trace)
    with tracing_scope():
        asyncio.run(_repl(cfg, _workspace(cfg, root), mode, console))


# -- run --


async def _run_once(cfg, task, root, mode, console) -> int:
    async with ToolClient(root, mode) as tools:
        session = _make_session(cfg, tools, console)
        result = await session.run_task(task)
    return 0 if result.stop_reason == "final" else 1


@app.command()
def run(
    ctx: typer.Context,
    task: Annotated[str, typer.Argument(help="실행할 작업")],
    config: ConfigOpt = None,
    root: RootOpt = None,
    sandbox_mode: ModeOpt = None,
    trace: TraceOpt = False,
):
    """작업 하나를 실행하고 종료한다."""
    g = _opts(ctx)
    cfg = load_config(config or g.config)
    console = Console()
    mode = _mode(cfg, sandbox_mode or g.sandbox_mode)
    _preflight(mode, console)
    _maybe_enable_tracing(cfg, console, trace or g.trace)
    workspace = _workspace(cfg, root or g.root)
    with tracing_scope():
        code = asyncio.run(_run_once(cfg, task, workspace, mode, console))
    raise typer.Exit(code)


# -- eval --


@app.command("eval")
def eval_command(
    ctx: typer.Context,
    tasks_dir: Annotated[Path, typer.Argument(help="작업셋 디렉터리 (eval_tasks/)")],
    model: Annotated[str | None, typer.Option("--model", help="플래너 모델 이름")] = None,
    full: Annotated[
        bool, typer.Option("--full", help="API 모델이 플래너와 응답 모두 담당")
    ] = False,
    skills_store: Annotated[
        Path | None, typer.Option("--skills-store", help="스킬 저장소(읽기 전용). 기본은 스킬 끔")
    ] = None,
    sandbox_mode: ModeOpt = None,
    out: Annotated[Path | None, typer.Option("--out", help="결과 디렉터리")] = None,
    task_id: Annotated[list[str] | None, typer.Option("--task-id", help="일부 작업만 실행")] = None,
    trace: Annotated[bool, typer.Option("--trace/--no-trace", help="LangSmith 추적")] = True,
    config: ConfigOpt = None,
):
    """평가 작업셋을 배치 실행한다. 추적은 기본으로 켜진다(합성 데이터만 쓰므로)."""
    from serana_agent.eval.runner import run_eval  # WP-E; imported late so other commands work

    g = _opts(ctx)
    cfg = load_config(config or g.config)
    console = Console()
    mode = _mode(cfg, sandbox_mode or g.sandbox_mode)
    _preflight(mode, console)
    traced_run = trace and set_tracing(True, cfg.langsmith_project)
    if trace and not traced_run:
        console.print(f"[yellow]{API_KEY_ENV}가 없어 LangSmith 추적 없이 실행합니다.[/yellow]")
    registry = build_registry(cfg)
    planner, persona = registry.pair(model or "serana", full=full)
    # Skills are off unless a store is named; memory stays off because eval traces leave the
    # machine and personal memory must not end up in them.
    skills = open_skills(cfg, read_only=True, path=skills_store) if skills_store else None
    out_dir = out or Path("runs/eval") / time.strftime("%Y%m%d-%H%M%S")
    audit = AuditLog(out_dir / "audit.jsonl")

    def factory(tools: ToolClient, approver, skill_store: SkillStore | None) -> Agent:
        return Agent(
            planner,
            persona,
            tools,
            approver,
            skills=skill_store,
            audit=audit,
            step_limit=cfg.step_limit,
            skill_k=cfg.skill_k,
            planner_think=cfg.planner_think,
            planner_max_tokens=cfg.planner_max_tokens,
        )

    meta = {
        "planner_model": planner.name,
        "persona_model": persona.name,
        "step_limit": cfg.step_limit,
        "sandbox_mode": mode,
        "skills": skills is not None,
    }
    with tracing_scope():
        summary = asyncio.run(
            run_eval(
                tasks_dir,
                factory,
                out_dir=out_dir,
                sandbox_mode=mode,
                skills=skills,
                task_ids=task_id or None,
                meta=meta,
            )
        )
    console.print(f"결과: {out_dir}")
    if summary is None:
        return
    from serana_agent.eval.report import print_table

    print_table(summary, console)
    if traced_run:
        from serana_agent.eval.langsmith_sync import sync_to_langsmith

        try:
            name = sync_to_langsmith(tasks_dir, summary)
        except Exception as e:  # the local results are already saved
            console.print(f"[yellow]LangSmith 동기화 실패: {e}[/yellow]")
        else:
            if name:
                console.print(f"LangSmith 실험: {name}")


def main() -> None:
    app()
