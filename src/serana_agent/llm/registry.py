"""Build (planner, persona) pairs from the `[models.*]` tables of serana.toml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from serana_agent.llm.api import AnthropicModel, OpenAIModel
from serana_agent.llm.base import ChatModel
from serana_agent.llm.mlx_local import LocalModel


class ModelRegistry:
    def __init__(self, models: dict[str, dict[str, Any]], models_dir: Path = Path("models")):
        self.models = models
        self.models_dir = models_dir
        self._local: dict[str, LocalModel] = {}

    def names(self) -> list[str]:
        return list(self.models)

    def _local_name(self) -> str:
        for name, cfg in self.models.items():
            if cfg.get("provider") == "local":
                return name
        raise KeyError('no provider = "local" entry in [models.*]')

    def _local_model(self, name: str) -> LocalModel:
        if name not in self._local:
            cfg = self.models[name]
            merged = cfg.get("merged_persona")
            self._local[name] = LocalModel(
                str(self.models_dir / cfg["base"]),
                str(self.models_dir / cfg["adapter"]) if cfg.get("adapter") else None,
                merged_persona_path=str(self.models_dir / merged) if merged else None,
            )
        return self._local[name]

    def _api(self, name: str) -> ChatModel:
        cfg = self.models[name]
        provider = cfg.get("provider")
        if provider == "anthropic":
            return AnthropicModel(name, cfg["model"], cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
        if provider == "openai":
            return OpenAIModel(
                name,
                cfg["model"],
                cfg.get("api_key_env", "OPENAI_API_KEY"),
                base_url=cfg.get("base_url"),
            )
        raise ValueError(f"unknown provider {provider!r} for model {name!r}")

    def pair(self, name: str, *, full: bool = False) -> tuple[ChatModel, ChatModel]:
        """(planner, persona). Local: base without adapter + adapter. API: planner only,
        persona stays local unless `full`, where the API model plays both roles."""
        cfg = self.models[name]
        if cfg.get("provider") == "local":
            if full:
                raise ValueError("--full needs an API model, not a local one")
            lm = self._local_model(name)
            return lm.planner(), lm.persona()
        api = self._api(name)
        if full:
            return api, api
        return api, self._local_model(self._local_name()).persona()
