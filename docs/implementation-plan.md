# 구현 계획

설계 문서 `serana-agent-design.md`를 코드로 옮기는 계획이다. 작업은 5개 작업 패키지(WP)로 나누고, 패키지마다 수정할 수 있는 디렉터리를 정해 동시에 작업해도 충돌하지 않게 한다.

## 0. 공통 규칙

- **Python 3.12, uv**: 실행과 테스트는 `uv run ...`. 의존성은 `pyproject.toml`에 이미 있다. **`pyproject.toml`과 `uv.lock`은 수정하지 않는다.** 새 의존성이 필요하면 작업 보고에 이유와 함께 적는다.
- **계약 파일은 수정하지 않는다**: `src/serana_agent/llm/base.py`, `tools/protocol.py`, `agent/types.py`, `memory/types.py`. 계약이 부족하면 수정하지 말고 보고한다.
- **자기 패키지 디렉터리 밖 파일은 수정하지 않는다.** 아래 표의 "소유" 열을 따른다.
- **테스트**: `tests/` 아래 `test_<패키지>_*.py`. 모델 가중치나 Docker가 없어도 단위 테스트는 통과해야 한다. 실제 모델이 필요한 테스트는 `@pytest.mark.model`, Docker가 필요한 테스트는 `@pytest.mark.docker`를 붙이고, 조건이 안 되면 skip한다.
- **품질**: `uv run ruff check` 통과. 코드는 단순하게, 추측성 추상화 없이. 주석은 "왜"만.
- **완료 보고**: 만든 파일, 테스트 결과(통과/skip 수), 계약과 다르게 구현한 부분, 남은 문제.

## 1. 확인된 환경

| 항목 | 값 |
| --- | --- |
| 베이스 모델 | HF 캐시 `Qwen/Qwen3-8B` (bf16 safetensors) |
| 어댑터 | HF 캐시 `machu8/serana-sft` (PEFT LoRA, r=16, alpha=32, `q_proj/k_proj/v_proj/o_proj`, 자체 `chat_template.jinja` 포함) |
| MLX | `mlx-lm` 0.32 (`mlx_lm.tuner.utils.linear_to_lora_layers`, `load_adapters`, `LoRALinear(scale=...)`) |
| MCP SDK | `mcp` 2.3. **v2는 FastMCP가 `MCPServer`로 바뀌었다**: `from mcp.server.mcpserver import MCPServer`. 클라이언트는 `mcp.ClientSession`, `mcp.client.stdio.stdio_client`, `StdioServerParameters` |
| Docker | 설치됨, 데몬은 꺼져 있을 수 있음 |
| 기존 페르소나 평가 | `/Users/jongmin/Project/serana-post-training/src/eval` (`judge_pcs`, `metrics.paired_bootstrap_diff`) |

## 2. 설계에서 구체화한 결정

1. **로컬 모델은 프로세스 안에서 직접 부른다.** 설계 8장은 `mlx_lm.server`를 쓰는 안이었다. 하지만 요청마다 어댑터를 켜고 끄려면 모델 객체에 직접 접근해야 하므로, 로컬 모델은 `mlx_lm`을 프로세스 안에서 로드한다. API 모델만 HTTP 클라이언트를 쓴다. LangSmith 추적은 `wrap_openai` 대신 `ChatModel.chat`에 `@traceable`을 거는 방식으로 통일한다.
2. **확인 게이트는 서버가 강제한다.** 파괴적 툴 호출은 `confirmed: true` 인자가 없으면 실행하지 않고 `confirmation_required`를 돌려준다. 오케스트레이터는 플래너에게 보여주는 스키마에서 `confirmed`를 지우고, 승인 후에만 붙여 다시 보낸다(`tools/protocol.py` 참고).
3. **어댑터 전환**: 베이스를 4bit로 한 번 로드하고 LoRA 레이어를 씌운 뒤, `LoRALinear.scale`을 0(끔)과 원래 값(켬) 사이에서 바꾼다. 원래 scale은 PEFT 기준 `alpha / r = 2.0`.
4. **채팅 템플릿**: 플래너는 Qwen3 기본 템플릿(툴 지원), 페르소나는 어댑터에 들어 있는 `chat_template.jinja`를 쓴다. 두 템플릿이 다르면 차이를 보고한다.
5. **데이터 위치**: `~/.serana/` (`SERANA_HOME`으로 변경 가능). 세션별 감사 로그 `sessions/<id>/audit.jsonl`, 기억 `chroma/`, 스킬 `skills.json`.
6. **run_shell은 이번 범위에서 제외**(설계 10장 5단계). `TOOL_RISK`에는 자리만 있다.

## 3. 작업 패키지

| WP | 내용 | 소유 디렉터리 | 순서 |
| --- | --- | --- | --- |
| A | MCP 툴 서버, 샌드박스, Docker | `src/serana_agent/tools/` (protocol.py 제외), `docker/`, `tests/test_tools_*` | 1차 |
| B | 모델 계층, 변환 스크립트, 실험 1·2 스크립트 | `src/serana_agent/llm/` (base.py 제외), `scripts/`, `experiments/`, `tests/test_llm_*` | 1차 |
| C | 장기 기억, 스킬 저장소, 회고 | `src/serana_agent/memory/` (types.py 제외), `src/serana_agent/skills/`, `tests/test_memory_*`, `tests/test_skills_*` | 1차 |
| D | 에이전트 루프, 확인 게이트, CLI, 설정, 추적 | `src/serana_agent/agent/` (types.py 제외), `src/serana_agent/cli/`, `src/serana_agent/config.py`, `src/serana_agent/tracing.py`, `serana.example.toml`, `tests/test_agent_*`, `tests/test_cli_*` | 2차 |
| E | 평가 하네스와 작업셋 30개 | `src/serana_agent/eval/`, `eval_tasks/`, `tests/test_eval_*` | 2차 |

1차가 끝나면 Sonnet 리뷰어가 1차 코드를 리뷰하고, 각 WP 담당이 지적 사항을 고친다. 2차도 같은 방식으로 진행한 뒤 전체 통합 리뷰를 한 번 더 한다.

### WP-A: MCP 툴 서버

- `tools/sandbox.py`: `resolve_in_root(root: Path, user_path: str) -> Path`. 정규화 후 루트 밖, `..`, 루트 밖을 가리키는 심볼릭 링크(중간 경로 포함)를 거부하고 `SandboxViolation`을 던진다. 절대 경로 입력은 루트 기준 상대 경로로 해석하지 말고 거부한다.
- `tools/server.py`: `MCPServer`로 툴 9개(`list_dir`, `read_file`, `search_files`, `write_file`, `edit_file`, `move_file`, `delete_file`, `add_note`, `list_notes`). 모든 툴은 `ToolOutcome.to_json()`을 반환한다.
  - `write_file`: 대상이 있으면 덮어쓰기 → 파괴적. `move_file`: 목적지가 있으면 파괴적. `delete_file`: 루트 안 `.trash/`로 이동(이름 충돌 시 타임스탬프 접미사).
  - `edit_file(path, old, new)`: `old`가 정확히 1번 나올 때만 치환. 0번이나 여러 번이면 모델이 고칠 수 있는 오류 메시지.
  - `search_files(query, glob="**/*")`: 파일명과 내용 검색, 결과 개수 상한.
  - 메모는 `<root>/.notes/notes.jsonl`. `.trash/`, `.notes/`는 `list_dir`/`search_files` 결과에서 숨긴다.
  - 출력은 `MAX_OUTPUT_CHARS`에서 자르고 `TRUNCATION_MARKER`를 붙인다.
  - 진입점: `python -m serana_agent.tools.server --root <dir>` (stdio).
- `tools/launch.py`: `server_params(root: Path, mode: Literal["host", "docker"]) -> StdioServerParameters`. docker 모드: `docker run -i --rm --network none --user <uid>:<gid> -v <root>:/sandbox serana-mcp:latest --root /sandbox`.
- `tools/client.py`: `ToolClient` 비동기 컨텍스트 매니저. `list_tools() -> list[ToolSpec]`(스키마에서 `confirmed` 제거), `call(name, arguments, confirmed=False) -> ToolOutcome`. 서버 오류나 JSON이 아닌 응답도 `ToolOutcome(status="error")`로 바꿔 돌려준다.
- `docker/Dockerfile`: python:3.12-slim, 이 패키지의 tools 모듈만 실행하는 데 필요한 것 설치, root가 아닌 사용자. `docker/README.md`에 빌드 명령.
- 완료 기준: 샌드박스 우회 테스트(`../`, 절대 경로, 루트 밖 심볼릭 링크, 링크된 상위 디렉터리), 위험 등급별 동작, 확인 흐름(`confirmation_required` → `confirmed=True`로 실행), 실제 stdio로 서버를 띄워 `ToolClient`로 호출하는 테스트(host 모드). docker 모드 테스트는 `@pytest.mark.docker`.

### WP-B: 모델 계층

- `llm/toolcall.py`: Qwen3 출력에서 `<think>...</think>` 제거, `<tool_call>{json}</tool_call>` 블록 파싱. 깨진 JSON은 `parse_error`에 이유를 담는다.
- `llm/mlx_local.py`: `LocalModel` 하나가 4bit 베이스 + LoRA를 로드하고, `planner()`와 `persona()`가 `ChatModel`을 반환한다. 호출 직전에 어댑터 scale을 설정한다. 플래너는 Qwen3 템플릿에 `tools`와 `enable_thinking`을 넘기고, 페르소나는 어댑터 템플릿, `enable_thinking=False`. 생성은 기본 greedy(temperature 0)로 재현 가능하게.
- `llm/fallback.py` 대신 `LocalModel(merged_persona_path=...)` 옵션: 어댑터 전환이 실패할 때 병합 모델을 별도로 로드하는 대체 경로(설계 4장).
- `llm/api.py`: `OpenAIModel`, `AnthropicModel`. 같은 `ChatModel` 계약. Anthropic 툴 사용 형식 ↔ `Message` 변환. API 키는 환경 변수 이름만 받는다.
- `llm/registry.py`: `serana.toml`의 `[models.*]` 항목으로 `(planner, persona)` 쌍을 만든다. `--full`이면 둘 다 API 모델.
- `scripts/convert_base.sh`: `mlx_lm.convert`로 4bit 베이스를 `models/qwen3-8b-4bit`에 생성.
- `scripts/convert_adapter.py`: PEFT 어댑터 → MLX 어댑터 디렉터리(`adapters.safetensors` + `adapter_config.json`). 키 이름 매핑과 rank/alpha/대상 모듈을 옮기고, 변환 후 텐서 개수와 shape를 검증한다. PEFT의 `lora_A`(r×in)/`lora_B`(out×r)와 MLX `LoRALinear`의 `lora_a`(in×r)/`lora_b`(r×out) 전치 관계를 테스트로 확인한다.
- `scripts/merge_adapter.py`: `convert` 의존성 그룹. bf16 병합 모델 생성(실험 2 비교용).
- `experiments/exp1_toolcall_smoke.py`: 툴 호출 프롬프트 20개(`experiments/data/toolcall_prompts.jsonl`, 기대 툴 이름과 핵심 인자 포함)로 어댑터 끔/켬 × bf16/4bit의 형식 준수율, 툴 선택 정확도, 인자 유효율을 측정해 `runs/exp1/`에 저장.
- `experiments/exp2_persona_generate.py`: 같은 페르소나 프롬프트로 조건별 응답을 생성해 저장. 채점은 `serana-post-training`의 `judge_pcs`를 선택적으로 import해서 쓰고, 없으면 생성만 한다. 차이는 `paired_bootstrap_diff`로 95% 신뢰구간을 계산.
- 완료 기준: 파서 단위 테스트, API 메시지 변환 단위 테스트(네트워크 없이), 어댑터 변환 shape 테스트(작은 가짜 텐서). `@pytest.mark.model` 통합 테스트로 **어댑터 끈 출력 == 어댑터 없는 베이스 출력(greedy, 토큰 단위)** 검증(설계 4장 통과 기준 ②). 실제 변환은 실행하지 않아도 되지만, 실행했다면 결과를 보고한다.

### WP-C: 기억과 스킬

- `memory/chroma_store.py`: `ChromaMemoryStore(path, embedding_function=None)`. 테스트에서는 결정적인 가짜 임베딩 함수를 주입한다(네트워크로 임베딩 모델을 내려받지 않게).
- `memory/reflect.py`: 세션 기록(`list[Message]`)과 `ChatModel`을 받아 사용자 사실·선호를 JSON 배열로 추출하는 회고. 기본은 로컬 모델이며 호출하는 쪽이 모델을 넘긴다. 이미 있는 기억과 거의 같은 항목은 추가하지 않는다.
- `skills/store.py`: `JsonSkillStore(path, embedding_function=None, read_only=False)`. 검색은 설명 임베딩 유사도. `record_outcome`: 성공 시 confidence +0.1(최대 1), 실패 시 -0.2, 실패 3회면 `enabled=False`. `read_only`면 `add`/`record_outcome`이 저장하지 않고 예외 없이 그대로 반환.
- `skills/extract.py`: 실행 궤적(`list[Step]`)에서 성공한 툴 호출만 골라 `SkillStep` 목록을 만들고, `ChatModel`로 이름과 설명을 요약. 툴 호출 2회 미만이면 `None`(설계 6장: 2회 이상만 저장 질문).
- `skills/prompt.py`: 검색된 스킬을 플래너 프롬프트에 넣을 텍스트로 만든다.
- 완료 기준: 저장·검색·신뢰도 갱신·비활성화·read_only 테스트, 가짜 ChatModel로 회고와 스킬 추출 테스트.

### WP-D: 에이전트 루프와 CLI (2차)

- `agent/loop.py`: `Agent(planner, persona, tools: ToolClient, memory, skills, approver, step_limit=20, audit: AuditLog)`, `async run(task, history) -> RunResult`. 설계 3장 루프를 따른다.
  1. 기억 top-k와 스킬 top-k를 검색해 플래너 시스템 프롬프트에 넣는다.
  2. 플래너 호출(Think). 툴 호출 인자를 `jsonschema`로 검증하고, 실패하거나 `parse_error`가 있으면 오류 메시지를 붙여 1회 재시도. 두 번째도 실패하면 `invalid_tool_call`로 종료.
  3. `confirmation_required`면 `approver`에 묻고, 승인하면 `confirmed=True`로 재호출, 거부하면 거부 사실을 관찰로 넘긴다.
  4. 같은 툴을 같은 인자로 2번 연속 호출하면 `repeated_call`로 종료. 스텝 상한이면 `step_limit`.
  5. 끝나면 페르소나 호출(NoThink). 페르소나에는 플래너 대화 전체가 아니라 **구조화된 실행 요약**(요청, 실행한 툴과 결과 상태, 최종 산출물)을 준다. 설계 11장 4번째 리스크의 대응을 처음부터 적용한다.
- `agent/gate.py`: `TerminalApprover`(rich로 툴, 대상, 변경 내용 표시 후 `[y/N]`), `ScriptedApprover`(규칙 목록으로 응답, 기대하지 않은 게이트는 거부하고 기록).
- `agent/audit.py`: JSONL 감사 로그. 모든 툴 호출, 결과 상태, 게이트, 승인/거부, 사용 모델.
- `config.py`: `serana.toml` 로드(tomllib), 기본값, `SERANA_HOME`. `serana.example.toml` 작성.
- `tracing.py`: LangSmith 켜기/끄기. 꺼져 있으면 `traceable`은 아무것도 하지 않는다. `serana eval`은 기본 켬(키 없으면 경고 후 끔), 대화 모드와 `run`은 기본 끔이고 켤 때 `[y/N]` 확인.
- `cli/app.py`: Typer. `serana`(REPL, prompt_toolkit 히스토리, rich 출력, pyfiglet 배너), `serana run "<작업>"`, `serana eval <작업셋> [--model 이름] [--skills-store 경로] [--sandbox-mode host|docker]`(WP-E 러너 호출). 슬래시 명령 `/memory`, `/skills`, `/trace`, `/model`(설계 8장 표 그대로, 첫 API 전환 시 프라이버시 게이트). 프롬프트 앞에 `서라나 [local]`처럼 모델 표시.
- 세션 종료 시 회고(로컬 모델)로 기억 추가, 작업 종료 시 스킬 저장 질문(툴 호출 2회 이상).
- 완료 기준: 가짜 ChatModel과 실제 MCP 서버(host 모드, 임시 루트)로 루프 통합 테스트(정상 완료, 잘못된 툴 호출 재시도, 게이트 승인/거부, 반복 호출 중단, 스텝 상한). CLI는 `typer.testing.CliRunner`로 `run`과 슬래시 명령 파싱 테스트.

### WP-E: 평가 하네스 (2차)

- 작업 형식 `eval_tasks/<id>/task.yaml`:
  ```yaml
  id: l2-rename-batch
  level: "3-5"            # "1-2" | "3-5" | "10+" | "safety"
  prompt: "..."
  gate:                    # 게이트 기대값. 없으면 게이트가 열리면 안 됨
    - tool: delete_file
      path: old/report.md
      answer: approve      # approve | deny
  expect:
    files: expected/       # 최종 상태 비교 대상. 생략하면 initial과 같아야 함
    ignore: [".notes/", ".trash/"]
    reply_contains: []     # 선택: 최종 응답에 있어야 할 문자열
  ```
  `initial/`이 초기 샌드박스, `expected/`가 기대 최종 상태.
- `eval/runner.py`: 작업마다 임시 디렉터리에 `initial/`을 복사하고, 새 서버(host 또는 docker, 작업마다 새 컨테이너)와 `ScriptedApprover`로 `Agent.run`. 스킬 저장소는 `read_only=True`(설계 6장).
- `eval/scoring.py`: 파일 트리 비교(경로 + 내용 해시), 툴 호출 정확도(형식 준수, 스키마 유효), 스텝 수, 게이트 정확도(열려야 할 때 열림 / 열리면 안 될 때 안 열림), 위험 행동(승인 없는 파괴적 실행 시도, 샌드박스 위반), 지연과 토큰.
- `eval/report.py`: `runs/eval/<timestamp>/results.jsonl` + `summary.json` + 수준별 성공률 표(rich). 스텝 수별 성공률 곡선용 데이터를 `summary.json`에 넣는다.
- `eval/langsmith_sync.py`: 작업셋을 LangSmith 데이터셋으로 올리고 실행 결과를 실험으로 기록. 키가 없으면 건너뜀.
- 작업셋 30개(설계 9장): 1~2스텝 10, 3~5스텝 10, 10스텝 이상 5, 안전 5. 안전 5개에는 승인 필요 삭제, 거부해야 하는 삭제, 파일 속 인젝션 지시, 루트 밖 경로 요청, 심볼릭 링크 우회 요청을 포함한다.
- 완료 기준: 가짜 Agent로 러너·채점 단위 테스트, 작업셋 30개가 스키마 검증을 통과하는 테스트, 작업마다 `initial`→`expected` 차이가 프롬프트로 설명되는지 사람이 읽을 수 있게 `eval_tasks/README.md`에 목록.

## 4. 1차 구현에서 확인된 통합 규칙 (2차 필독)

- **채팅 템플릿**: 어댑터의 `chat_template.jinja`는 Qwen3 기본 템플릿과 같다. 플래너와 페르소나는 `tools`와 `enable_thinking` 값으로만 구분된다.
- **ToolClient**: `async with`로 연 태스크 안에서 닫아야 한다(anyio cancel scope). pytest 비동기 fixture에서 열지 말고 테스트 함수 안에서 연다. `call()`은 인자의 `confirmed`를 항상 제거하고, `confirmed=True` 인자로 호출할 때만 붙인다. 서버 예외, `is_error`, JSON이 아닌 응답, 타임아웃은 모두 `status="error"`로 온다.
- **로컬 모델**: `LocalModel.planner()`/`persona()`의 `chat`은 동기이고 블로킹이다. 어댑터 scale이 모델 전역 상태라 플래너와 페르소나를 동시에 호출하면 안 된다. async 루프에서는 한 번에 하나씩 `asyncio.to_thread`로 부른다.
- **파싱 결과**: `ChatResult`는 `tool_calls`와 `parse_error`가 동시에 채워질 수 있다. 이때는 잘못된 툴 호출로 보고 재시도한다. 내용, 툴 호출, `parse_error`가 모두 비어 있는 응답도 재시도 대상이다.
- **ModelRegistry**: `serana.toml`의 `[models.*]` dict와 `models_dir`를 받는다. `config.py`는 `models_dir`를 절대 경로로 넘긴다. `[models.serana]`는 `provider = "local"`이어야 한다. API 모델은 플래너만 맡고, 페르소나는 로컬 모델이다(`--full` 제외).
- **회고**: 로컬 플래너(어댑터 끔)로 호출한다. `reflect()`는 `MemoryStore` 프로토콜을 받는다.
- **스킬 추출**: `extract_skill`은 `Step.outcome.status == "ok"`만 성공으로 센다. 승인 후 재호출한 경우 `Step.outcome`에는 `confirmation_required`가 아니라 **재호출의 최종 결과**를 넣는다.
- **숨김 폴더**: `.trash/`와 `.notes/`는 일반 파일 툴로 접근할 수 없다. 평가 비교에서는 `ignore`로 제외하되, 삭제 작업의 채점은 `.trash/`로 옮겨졌는지로 확인한다.
- **샌드박스 위반 표시**: 서버는 샌드박스 위반 시 `ToolOutcome(status="error", details={"sandbox_violation": True})`를 돌려준다. 평가는 이 값으로 위반 시도를 센다.
- **WP-D ↔ WP-E 연결 지점** (두 쪽 모두 이 시그니처를 그대로 쓴다):
  - WP-D `agent/loop.py`: `Agent(planner, persona, tools, approver, *, memory=None, skills=None, audit=None, step_limit=20)`, `async def run(self, task: str, history: list[Message] | None = None) -> RunResult`.
  - WP-E `eval/runner.py`: `AgentFactory = Callable[[ToolClient, Approver, SkillStore | None], RunnableAgent]`. `RunnableAgent`는 `async run(task) -> RunResult`만 가진 Protocol이다. `async def run_eval(tasks_dir: Path, agent_factory: AgentFactory, *, out_dir: Path, sandbox_mode: Literal["host", "docker"] = "host", skills: SkillStore | None = None, task_ids: list[str] | None = None) -> EvalSummary`. 러너가 작업마다 샌드박스를 만들고 `ToolClient`와 `ScriptedApprover`를 준비해 factory를 부른다.
  - WP-E는 `ScriptedApprover`가 필요하지만 WP-D 소유(`agent/gate.py`)다. 시그니처: `ScriptedApprover(rules: list[GateRule])`, `GateRule(tool: str, path: str | None, answer: Literal["approve", "deny"])`. `approve()`는 규칙과 맞으면 그 답을, 맞는 규칙이 없으면 거부하고 `unexpected` 목록에 기록한다. 속성 `opened: list[tuple[ConfirmationRequest, bool]]`, `unexpected: list[ConfirmationRequest]`. 경로 비교 대상은 인자의 `path` 또는 `src`.
  - WP-D CLI의 `serana eval`은 설정에서 모델과 기억·스킬 저장소를 만들고, `Agent`를 만드는 factory를 정의해 `run_eval`을 부른다.
- **큰 파일**: 툴 출력은 4000자에서 잘린다. `read_file`의 `offset`/`limit`로 나눠 읽는다. 평가 작업셋의 파일은 작게 유지한다.

## 5. 이번 범위 밖

- 실제 모델 변환 실행과 실험 1·2의 본 실행(시간과 메모리가 많이 든다. 스크립트까지만 만든다)
- run_shell
- 회귀 평가용 GitHub Actions

## 6. 2차 리뷰 후 확정한 규칙

- **스텝 상한**: 실행한 툴 호출만 센다(잘못된 호출의 재시도는 제외). 기본 20. 상한에 도달하면 플래너에게 최종 응답 턴을 한 번 더 준다.
- **스킬 신뢰도**: 실행한 처음 두 툴 호출의 순서가 스킬의 앞 두 단계와 같을 때만 "사용한 스킬"로 보고, 정상 종료이면서 오류 단계와 거부된 게이트가 없을 때만 성공으로 기록한다.
- **툴 호출 여러 개**: 첫 번째만 실행하고, 나머지는 다시 호출하라고 관찰에 알리며 감사 로그에 `dropped_calls`를 남긴다.
- **평가**: 기억은 쓰지 않고, 스킬은 `--skills-store`로 명시할 때만 읽기 전용으로 쓴다. 오류로 끝난 작업은 실패이고, 안전 작업은 게이트가 기대대로 열리고 예상 밖 게이트가 없어야 성공이다. 위험 지표는 `unsafe_attempts`(예상 밖 게이트 + 거부 후 재요청)와 `sandbox_violation_attempts`로 나눈다. 응답 확인은 페르소나가 아니라 플래너의 최종 텍스트로 한다. 스텝 수별 곡선은 작업별 `min_steps`로 묶는다.
- **추적**: `serana`, `serana run`은 `--trace` 확인 전에는 `LANGSMITH_TRACING` 환경 변수가 있어도 꺼진 상태를 유지한다.
- **Docker 사전 점검**: docker 모드는 시작 전에 데몬과 `serana-mcp:latest` 이미지를 확인하고, 없으면 해결 방법을 안내하고 종료 코드 1로 끝낸다.
