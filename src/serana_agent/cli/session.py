"""Interactive session state: model routing, slash commands, history, and session-end hooks."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from serana_agent.agent.audit import AuditLog
from serana_agent.agent.loop import Agent, turn_record
from serana_agent.agent.types import Approver, RunResult
from serana_agent.config import Config
from serana_agent.llm.base import ChatModel, Message
from serana_agent.llm.registry import ModelRegistry
from serana_agent.memory.reflect import reflect
from serana_agent.memory.types import MemoryStore, SkillStore
from serana_agent.skills.extract import extract_skill
from serana_agent.tools.client import ToolClient

LOCAL = "serana"
MAX_HISTORY = 20
HELP = """/memory [검색어]  저장된 기억 조회
/skills           스킬 목록
/trace            직전 작업의 툴 호출 내역
/model            현재 모델 | /model list | /model serana | /model <이름> [--full]
/exit             종료"""


class Session:
    def __init__(
        self,
        cfg: Config,
        registry: ModelRegistry,
        tools: ToolClient,
        approver: Approver,
        confirm: Callable[[str], bool],
        console: Console,
        *,
        memory: MemoryStore | None = None,
        skills: SkillStore | None = None,
        audit: AuditLog | None = None,
    ):
        self.cfg = cfg
        self.registry = registry
        self.tools = tools
        self.approver = approver
        self.confirm = confirm
        self.console = console
        self.memory = memory
        self.skills = skills
        self.audit = audit or AuditLog()
        self.history: list[Message] = []
        self.last_run_id = ""
        self.models_used: set[str] = set()
        self.confirmed_providers: set[str] = set()
        self.full_confirmed = False
        self.model_name = LOCAL
        self.full = False
        self._local_planner: ChatModel | None = None
        self.agent = self._make_agent(*registry.pair(LOCAL))

    @property
    def label(self) -> str:
        if self.model_name == LOCAL:
            return "local"
        return self.model_name + ("/full" if self.full else "")

    def _make_agent(self, planner: ChatModel, persona: ChatModel) -> Agent:
        cfg = self.cfg
        return Agent(
            planner,
            persona,
            self.tools,
            self.approver,
            memory=self.memory,
            skills=self.skills,
            audit=self.audit,
            step_limit=cfg.step_limit,
            memory_k=cfg.memory_k,
            skill_k=cfg.skill_k,
            planner_think=cfg.planner_think,
            planner_max_tokens=cfg.planner_max_tokens,
        )

    def local_planner(self) -> ChatModel:
        """Reflection and skill summaries stay local so memory never leaves the machine."""
        if self._local_planner is None:
            self._local_planner = self.registry.pair(LOCAL)[0]
        return self._local_planner

    # -- one task --

    async def run_task(self, task: str) -> RunResult:
        result = await self.agent.run(task, self.history)
        self.last_run_id = self.agent.last_run_id
        self.models_used.add(self.model_name)
        # The planner sees which tools ran and how they ended, not the raw outputs.
        reply = "" if self.agent.last_persona_failed else result.reply
        turn = "\n".join(p for p in (reply, turn_record(result)) if p)
        self.history.append({"role": "user", "content": task})
        if turn:
            self.history.append({"role": "assistant", "content": turn})
        del self.history[:-MAX_HISTORY]
        self.console.print(Panel(Markdown(result.reply), title="서라나", border_style="cyan"))
        if result.stop_reason != "final":
            self.console.print(f"[yellow]작업이 중단되었습니다: {result.stop_reason}[/yellow]")
        await self.offer_skill(result)
        return result

    async def offer_skill(self, result: RunResult) -> None:
        ok = [s for s in result.steps if s.tool_call and s.outcome and s.outcome.status == "ok"]
        if (
            not self.skills
            or self.skills.read_only
            or result.stop_reason != "final"
            or len(ok) < 2
            or not self.confirm("이 절차를 스킬로 저장할까요?")
        ):
            return
        draft = await asyncio.to_thread(
            extract_skill, result.task, result.steps, self.local_planner()
        )
        if draft is None:
            self.console.print("저장할 만한 절차를 만들지 못했습니다.")
            return
        self.skills.add(draft.name, draft.description, draft.steps)
        self.console.print(f"스킬 저장: {draft.name}")

    async def end(self) -> None:
        """Session-end reflection with the local model."""
        if not (self.memory and self.history):
            return
        meta = {"session_models": ",".join(sorted(self.models_used))}
        try:
            added = await asyncio.to_thread(
                reflect, self.history, self.local_planner(), self.memory, meta
            )
        except Exception as e:  # reflection is best effort; do not lose the exit
            self.console.print(f"[yellow]회고를 건너뜁니다: {e}[/yellow]")
            return
        for item in added:
            self.console.print(f"기억 추가: {item.text}")

    # -- slash commands --

    async def handle(self, line: str) -> bool:
        """Process one input line. Returns False when the user asked to exit."""
        line = line.strip()
        if not line:
            return True
        if not line.startswith("/"):
            await self.run_task(line)
            return True
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        if cmd in ("/exit", "/quit"):
            return False
        handlers = {
            "/memory": self._cmd_memory,
            "/skills": self._cmd_skills,
            "/trace": self._cmd_trace,
            "/model": self._cmd_model,
            "/help": lambda _: self.console.print(HELP),
        }
        handler = handlers.get(cmd)
        if handler is None:
            self.console.print(f"알 수 없는 명령: {cmd}  (/help)")
        else:
            handler(rest)
        return True

    def _cmd_memory(self, query: str) -> None:
        if not self.memory:
            self.console.print("기억 저장소가 없습니다.")
            return
        items = self.memory.search(query, self.cfg.memory_k) if query else self.memory.all()
        if not items:
            self.console.print("저장된 기억이 없습니다.")
        for item in items:
            self.console.print(f"- {item.text}", markup=False)

    def _cmd_skills(self, _: str) -> None:
        skills = self.skills.all() if self.skills else []
        if not skills:
            self.console.print("저장된 스킬이 없습니다.")
            return
        table = Table("이름", "설명", "신뢰도", "사용", "실패", "활성")
        for s in skills:
            table.add_row(
                s.name, s.description, f"{s.confidence:.1f}", str(s.uses), str(s.failures),
                "y" if s.enabled else "n",
            )  # fmt: skip
        self.console.print(table)

    def _cmd_trace(self, _: str) -> None:
        events = self.audit.for_run(self.last_run_id) if self.last_run_id else []
        if not events:
            self.console.print("아직 실행한 작업이 없습니다.")
            return
        for e in events:
            kind = e["event"]
            if kind == "run_start":
                text = f"모델: planner={e['planner']} persona={e['persona']}"
            elif kind == "tool_call":
                extra = " (승인 후 재호출)" if e.get("confirmed") else ""
                text = f"{e['tool']} {e['arguments']} -> {e['status']}{extra}"
                if e.get("error"):
                    text += f": {e['error']}"
            elif kind == "gate":
                text = f"게이트 {e['tool']}: {'승인' if e['approved'] else '거부'}"
            elif kind == "run_end":
                text = f"종료: {e['stop_reason']} ({e['steps']} 스텝)"
            else:
                text = f"{kind}: {e.get('error', '')}"
            self.console.print(text, markup=False)

    def _cmd_model(self, arg: str) -> None:
        parts = arg.split()
        names = self.registry.names()
        if not parts:
            provider = self.registry.models[self.model_name].get("provider")
            self.console.print(f"현재 모델: {self.model_name} ({provider}){' --full' * self.full}")
            return
        if parts[0] == "list":
            for n in names:
                self.console.print(f"- {n} ({self.registry.models[n].get('provider')})")
            return
        name, flags = parts[0], parts[1:]
        full = "--full" in flags
        if name not in names or set(flags) - {"--full"}:
            self.console.print(
                f"사용법: /model [list|<이름> [--full]]. 등록된 모델: {', '.join(names)}"
            )
            return
        provider = self.registry.models[name].get("provider")
        if provider != "local" and (
            provider not in self.confirmed_providers or (full and not self.full_confirmed)
        ):
            if not self._privacy_gate(name, full):
                self.console.print("모델을 바꾸지 않았습니다.")
                return
            self.confirmed_providers.add(provider)
            self.full_confirmed = self.full_confirmed or full
        try:
            pair = self.registry.pair(name, full=full)
        except Exception as e:
            self.console.print(f"[red]모델을 전환하지 못했습니다: {e}[/red]")
            return
        self.agent = self._make_agent(*pair)
        self.model_name, self.full = name, full
        self.console.print(f"모델 전환: {self.label}")

    def _privacy_gate(self, name: str, full: bool) -> bool:
        provider = self.registry.models[name].get("provider")
        role = "플래너와 최종 응답" if full else "플래너"
        self.console.print(
            f"[yellow]{name}({provider})이(가) {role}를 맡으면 "
            "다음 정보가 외부 API로 전송됩니다.[/yellow]\n"
            "- 대화 내용\n- 툴 결과(읽은 파일 내용 포함)\n- 프롬프트에 주입되는 장기 기억 발췌"
        )
        return self.confirm("외부 API 사용을 허용할까요?")


def new_audit(cfg: Config) -> AuditLog:
    return AuditLog(cfg.sessions_dir() / time.strftime("%Y%m%d-%H%M%S") / "audit.jsonl")
