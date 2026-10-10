import pytest

from serana_agent.llm import registry
from serana_agent.llm.api import AnthropicModel, OpenAIModel

MODELS = {
    "serana": {
        "provider": "local",
        "base": "qwen3-8b-4bit",
        "adapter": "serana-adapter-mlx",
    },
    "sonnet": {"provider": "anthropic", "model": "claude-x", "api_key_env": "A_KEY"},
    "gpt": {"provider": "openai", "model": "gpt-x", "api_key_env": "O_KEY"},
}


class FakeLocal:
    created = []

    def __init__(self, base, adapter, merged_persona_path=None):
        self.args = (base, adapter, merged_persona_path)
        FakeLocal.created.append(self)

    def planner(self, **kwargs):
        self.planner_kwargs = kwargs
        return ("planner", self)

    def persona(self):
        return ("persona", self)


@pytest.fixture(autouse=True)
def fake_local(monkeypatch):
    FakeLocal.created.clear()
    monkeypatch.setattr(registry, "LocalModel", FakeLocal)


def test_local_pair_uses_models_dir_and_caches():
    reg = registry.ModelRegistry(MODELS)
    (p_role, p_lm), (s_role, s_lm) = reg.pair("serana")
    assert (p_role, s_role) == ("planner", "persona") and p_lm is s_lm
    assert p_lm.args == ("models/qwen3-8b-4bit", "models/serana-adapter-mlx", None)
    reg.pair("serana")
    assert len(FakeLocal.created) == 1


def test_api_pair_keeps_local_persona():
    planner, persona = registry.ModelRegistry(MODELS).pair("sonnet")
    assert isinstance(planner, AnthropicModel) and planner.api_key_env == "A_KEY"
    assert persona[0] == "persona"


def test_full_uses_api_for_both_and_loads_no_local_model():
    planner, persona = registry.ModelRegistry(MODELS).pair("gpt", full=True)
    assert isinstance(planner, OpenAIModel) and planner is persona
    assert FakeLocal.created == []


def test_full_rejects_local_and_unknown_provider():
    reg = registry.ModelRegistry({**MODELS, "x": {"provider": "nope"}})
    with pytest.raises(ValueError):
        reg.pair("serana", full=True)
    with pytest.raises(ValueError, match="unknown provider"):
        reg.pair("x")


def test_planner_think_sampling_passed_to_local_planner():
    reg = registry.ModelRegistry(MODELS, planner_think_sampling=True, planner_think_seed=5)
    (_, lm), _ = reg.pair("serana")
    assert lm.planner_kwargs == {"think_sampling": True, "think_seed": 5}


def test_planner_think_sampling_off_by_default():
    (_, lm), _ = registry.ModelRegistry(MODELS).pair("serana")
    assert lm.planner_kwargs == {}
