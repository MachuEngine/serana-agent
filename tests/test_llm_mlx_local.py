import gc
import os
from pathlib import Path

import pytest

from serana_agent.llm.base import ToolSpec
from serana_agent.llm.mlx_local import LocalModel, to_template_messages

ROOT = Path(__file__).parents[1]
# Env overrides allow pointing the same tests at other converted models.
BASE = Path(os.environ.get("SERANA_TEST_BASE", ROOT / "models" / "qwen3-8b-4bit"))
ADAPTER = Path(os.environ.get("SERANA_TEST_ADAPTER", ROOT / "models" / "serana-adapter-mlx"))


def test_template_messages_convert_tool_calls():
    msgs = [
        {"role": "user", "content": "읽어줘"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "read_file", "arguments": {"path": "a"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "내용"},
    ]
    out = to_template_messages(msgs)
    assert out[1]["tool_calls"] == [
        {"type": "function", "function": {"name": "read_file", "arguments": {"path": "a"}}}
    ]
    assert out[2] == {"role": "tool", "content": "내용"}


def _bare_model(lora_layers):
    import threading

    lm = object.__new__(LocalModel)
    lm._lora = lora_layers
    lm.adapter_scale = 2.0
    lm._lock = threading.Lock()
    lm.merged = None
    return lm


def test_adapter_without_lora_layers_raises():
    with pytest.raises(ValueError, match="no LoRA layers"):
        with _bare_model([])._adapter(True):
            pass


def test_adapter_scale_restored_even_on_error():
    from types import SimpleNamespace as NS

    layer = NS(scale=2.0)
    lm = _bare_model([layer])
    with pytest.raises(RuntimeError):
        with lm._adapter(False):
            assert layer.scale == 0.0
            raise RuntimeError
    assert layer.scale == 2.0


def test_merged_model_only_used_when_adapter_requested():
    lm = _bare_model([])
    lm.merged = object()
    assert lm._uses_merged(persona=True, adapter=True)
    assert not lm._uses_merged(persona=True, adapter=False)
    assert not lm._uses_merged(persona=False, adapter=True)


def _sampling_model(monkeypatch, temperature=0.0):
    """LocalModel whose generation is faked; records samplers and seeds."""
    pytest.importorskip("mlx_lm")
    mx = pytest.importorskip("mlx.core")
    import mlx_lm.sample_utils as su

    from serana_agent.llm.mlx_local import Generation

    rec = {"samplers": [], "seeds": []}
    monkeypatch.setattr(su, "make_sampler", lambda **kw: rec["samplers"].append(kw) or kw)
    monkeypatch.setattr(mx.random, "seed", lambda s: rec["seeds"].append(s))
    lm = _bare_model([])
    lm.temperature = temperature
    lm._on_worker = lambda fn, *a: fn(*a)
    lm.model = lm.tokenizer = None
    lm.persona_template = None
    lm.render = lambda *a, **k: "prompt"

    def run(model, tokenizer, prompt, max_tokens, kwargs):
        rec["samplers"].append(("run", kwargs.get("sampler")))
        return Generation("hi", [1], 3)

    lm._run = run
    return lm, rec


MSG = [{"role": "user", "content": "x"}]


def test_think_sampling_uses_qwen_settings_and_seeds(monkeypatch):
    lm, rec = _sampling_model(monkeypatch)
    role = lm.planner(think_sampling=True, think_seed=11)
    role.chat.__wrapped__(role, MSG, think=True)
    assert rec["seeds"] == [11]
    assert rec["samplers"][0] == {"temp": 0.6, "top_p": 0.95, "top_k": 20}
    assert rec["samplers"][1][1] == rec["samplers"][0]


def test_think_sampling_skips_non_think_and_persona(monkeypatch):
    lm, rec = _sampling_model(monkeypatch)
    lm._lora = [object()]  # persona adapter requires LoRA layers
    lm._adapter = lambda on: __import__("contextlib").nullcontext()
    planner = lm.planner(think_sampling=True, think_seed=11)
    planner.chat.__wrapped__(planner, MSG, think=False)
    persona = lm.persona()
    persona.think_seed = 11  # even if mis-set, persona never thinks
    persona.chat.__wrapped__(persona, MSG, think=True)
    assert rec["seeds"] == [] and rec["samplers"] == [("run", None), ("run", None)]


def test_think_sampling_off_keeps_greedy(monkeypatch):
    lm, rec = _sampling_model(monkeypatch)
    role = lm.planner()
    role.chat.__wrapped__(role, MSG, think=True)
    assert rec["seeds"] == [] and rec["samplers"] == [("run", None)]


def test_non_think_call_keeps_model_temperature(monkeypatch):
    lm, rec = _sampling_model(monkeypatch, temperature=0.3)
    role = lm.planner(think_sampling=True)
    role.chat.__wrapped__(role, MSG, think=False)
    assert rec["seeds"] == [] and rec["samplers"][0] == {"temp": 0.3}


needs_models = pytest.mark.skipif(
    not (BASE.exists() and ADAPTER.exists()),
    reason="converted models missing: run scripts/convert_base.sh and scripts/convert_adapter.py",
)


@pytest.fixture(scope="module")
def local_model():
    from serana_agent.llm.mlx_local import LocalModel

    return LocalModel(str(BASE), str(ADAPTER))


PERSONA_MESSAGES = [
    {"role": "system", "content": "You are Serana. Reply in Korean."},
    {"role": "user", "content": "너는 어쩌다 뱀파이어가 된 거야?"},
]


@pytest.mark.model
@needs_models
def test_adapter_off_matches_plain_base_token_for_token():
    """Design section 4, pass criterion 2."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    plain, tokenizer = load(str(BASE))
    prompt = tokenizer.apply_chat_template(
        PERSONA_MESSAGES, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    expected = [r.token for r in stream_generate(plain, tokenizer, prompt, max_tokens=64)]
    del plain
    gc.collect()
    mx.clear_cache()

    from serana_agent.llm.mlx_local import LocalModel

    lm = LocalModel(str(BASE), str(ADAPTER))
    assert lm.adapter_scale > 0
    off = lm.generate(lm.render(PERSONA_MESSAGES), 64, adapter=False)
    assert off.token_ids == expected
    on = lm.generate(lm.render(PERSONA_MESSAGES), 64, adapter=True)
    assert on.token_ids != expected  # the adapter must actually change the output


@pytest.mark.model
@needs_models
def test_planner_formats_tools_and_parses_call(local_model):
    tools = [
        ToolSpec(
            "read_file",
            "Read a file",
            {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )
    ]
    planner = local_model.planner()
    result = planner.chat(
        [{"role": "user", "content": "notes.txt 파일을 읽어줘"}], tools, max_tokens=200
    )
    assert result.parse_error is None
    assert result.tool_calls and result.tool_calls[0].name == "read_file"
    assert result.usage["prompt_tokens"] > 0


@pytest.mark.model
@needs_models
def test_persona_template_renders_like_qwen3_template(local_model):
    """The adapter ships a chat template; it should match the base template."""
    with_adapter_template = local_model.render(PERSONA_MESSAGES, persona=True)
    assert with_adapter_template == local_model.render(PERSONA_MESSAGES, persona=False)


@pytest.mark.model
@needs_models
def test_chat_works_from_a_different_thread_than_load():
    """The agent loop calls chat() via asyncio.to_thread; MLX streams are per thread."""
    from concurrent.futures import ThreadPoolExecutor

    lm = LocalModel(str(BASE), str(ADAPTER))
    messages = [{"role": "user", "content": "안녕"}]
    with ThreadPoolExecutor(max_workers=1) as pool:
        for role in (lm.planner(), lm.persona()):
            result = pool.submit(role.chat, messages, max_tokens=16).result()
            assert result.content
