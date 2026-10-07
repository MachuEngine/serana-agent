import pytest
from langsmith.utils import tracing_is_enabled

from serana_agent import tracing
from serana_agent.config import load_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERANA_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    tracing._enabled = False
    yield
    tracing._enabled = False


def test_defaults_without_file(tmp_path):
    cfg = load_config()
    assert cfg.home == tmp_path / "home"
    assert cfg.models["serana"]["provider"] == "local"
    assert cfg.workspace == tmp_path / "home" / "workspace"
    assert cfg.models_dir.is_absolute() and cfg.sandbox_mode == "docker"


def test_file_overrides_and_relative_paths(tmp_path):
    conf = tmp_path / "conf" / "serana.toml"
    conf.parent.mkdir()
    conf.write_text(
        'models_dir = "weights"\n'
        '[models.serana]\nprovider = "local"\nbase = "b"\nadapter = "a"\n'
        '[models.sonnet]\nprovider = "anthropic"\nmodel = "m"\n'
        "[agent]\nstep_limit = 5\nplanner_think = false\n"
        '[sandbox]\nroot = "ws"\nmode = "host"\n'
        '[tracing]\nproject = "p"\n'
    )
    cfg = load_config(conf)
    assert cfg.models_dir == (conf.parent / "weights").resolve()
    assert set(cfg.models) == {"serana", "sonnet"}
    assert cfg.step_limit == 5 and cfg.planner_think is False
    assert cfg.workspace == (conf.parent / "ws").resolve() and cfg.sandbox_mode == "host"
    assert cfg.langsmith_project == "p"


def test_local_serana_toml_is_found_and_bad_mode_rejected(tmp_path):
    (tmp_path / "serana.toml").write_text('[sandbox]\nmode = "vm"\n')
    with pytest.raises(ValueError):
        load_config()
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")


def test_example_config_loads():
    from pathlib import Path

    cfg = load_config(Path(__file__).parent.parent / "serana.example.toml")
    assert {"serana", "sonnet"} <= set(cfg.models)


def test_set_tracing_needs_api_key(monkeypatch):
    assert tracing.set_tracing(True) is False and not tracing.tracing_enabled()
    monkeypatch.setenv("LANGSMITH_API_KEY", "dummy")
    assert tracing.set_tracing(True) is True and tracing.tracing_enabled()
    assert tracing.set_tracing(False) is False


async def test_traced_is_a_noop_when_disabled():
    seen = []

    @tracing.traced("t")
    async def fn(x):
        seen.append(tracing_is_enabled())
        return x + 1

    @tracing.traced("s")
    def sync_fn(x):
        seen.append(tracing_is_enabled())
        return x * 2

    assert await fn(1) == 2 and sync_fn(2) == 4
    assert seen == [False, False]
