"""serana.toml loading. Defaults work without a file; API keys are never read from it."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DEFAULT_MODELS: dict[str, dict[str, Any]] = {
    "serana": {"provider": "local", "base": "qwen3-8b-4bit", "adapter": "serana-adapter-mlx"},
}


@dataclass
class Config:
    home: Path
    models: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    models_dir: Path = Path("models")
    sandbox_root: Path | None = None  # None: <home>/workspace
    sandbox_mode: Literal["host", "docker"] = "docker"
    step_limit: int = 20
    memory_k: int = 5
    skill_k: int = 3
    planner_think: bool = True
    planner_max_tokens: int = 2048
    langsmith_project: str = "serana-agent"

    @property
    def workspace(self) -> Path:
        return self.sandbox_root or self.home / "workspace"

    @property
    def chroma_path(self) -> Path:
        return self.home / "chroma"

    @property
    def skills_path(self) -> Path:
        return self.home / "skills.json"

    def sessions_dir(self) -> Path:
        return self.home / "sessions"


def serana_home() -> Path:
    return Path(os.environ.get("SERANA_HOME", "~/.serana")).expanduser()


def _path(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: Path | None = None) -> Config:
    """Read `path`, else ./serana.toml, else <home>/serana.toml, else defaults.

    Relative paths in the file resolve against the file's directory.
    """
    home = serana_home()
    candidates = [path] if path else [Path("serana.toml"), home / "serana.toml"]
    file = next((p for p in candidates if p is not None and p.exists()), None)
    if path and file is None:
        raise FileNotFoundError(f"config file not found: {path}")
    cfg = Config(home=home)
    if file is None:
        cfg.models_dir = cfg.models_dir.resolve()
        return cfg
    base = file.resolve().parent
    data = tomllib.loads(file.read_text(encoding="utf-8"))
    if "models" in data:
        cfg.models = data["models"]
    cfg.models_dir = _path(data.get("models_dir", "models"), base)
    agent = data.get("agent", {})
    cfg.step_limit = int(agent.get("step_limit", cfg.step_limit))
    cfg.memory_k = int(agent.get("memory_k", cfg.memory_k))
    cfg.skill_k = int(agent.get("skill_k", cfg.skill_k))
    cfg.planner_think = bool(agent.get("planner_think", cfg.planner_think))
    cfg.planner_max_tokens = int(agent.get("planner_max_tokens", cfg.planner_max_tokens))
    sandbox = data.get("sandbox", {})
    if "root" in sandbox:
        cfg.sandbox_root = _path(sandbox["root"], base)
    if sandbox.get("mode", cfg.sandbox_mode) not in ("host", "docker"):
        raise ValueError("sandbox.mode must be 'host' or 'docker'")
    cfg.sandbox_mode = sandbox.get("mode", cfg.sandbox_mode)
    cfg.langsmith_project = data.get("tracing", {}).get("project", cfg.langsmith_project)
    return cfg
