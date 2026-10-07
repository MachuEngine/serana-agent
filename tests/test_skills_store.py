import json

from test_memory_fakes import FakeEmbedding

from serana_agent.agent.types import Step
from serana_agent.llm.base import ChatResult, ToolCall
from serana_agent.memory.types import SkillStep
from serana_agent.skills.extract import extract_skill
from serana_agent.skills.prompt import format_skills
from serana_agent.skills.store import JsonSkillStore
from serana_agent.tools.protocol import ToolOutcome

STEPS = [SkillStep("list_dir", {"path": "{dir}"}), SkillStep("read_file", {"path": "{file}"})]


def make(tmp_path, **kw):
    return JsonSkillStore(tmp_path / "skills.json", embedding_function=FakeEmbedding(), **kw)


def test_add_and_reload(tmp_path):
    skill = make(tmp_path).add("inspect", "inspect a folder", STEPS)
    loaded = make(tmp_path).all()
    assert [s.id for s in loaded] == [skill.id]
    assert loaded[0].steps == STEPS


def test_search_ranks_by_description(tmp_path):
    store = make(tmp_path)
    store.add("rename", "rename many files in a folder", STEPS)
    store.add("notes", "write a note about meeting", STEPS)
    assert store.search("rename files in folder", k=1)[0].name == "rename"
    assert len(store.search("anything", k=5)) == 2


def test_search_empty(tmp_path):
    assert make(tmp_path).search("x") == []


def test_confidence_updates(tmp_path):
    store = make(tmp_path)
    s = store.add("a", "desc", STEPS)
    assert store.record_outcome(s.id, True).confidence == 0.6
    for _ in range(10):
        store.record_outcome(s.id, True)
    assert store.all()[0].confidence == 1.0
    store2 = make(tmp_path)
    assert store2.all()[0].confidence == 1.0
    assert store2.all()[0].uses == 11


def test_failure_lowers_and_disables_after_three(tmp_path):
    store = make(tmp_path)
    s = store.add("a", "rename files", STEPS)
    assert store.record_outcome(s.id, False).confidence == 0.3
    store.record_outcome(s.id, False)
    assert store.all()[0].enabled
    out = store.record_outcome(s.id, False)
    assert not out.enabled and out.failures == 3 and out.confidence == 0.0
    assert store.search("rename files") == []
    assert len(make(tmp_path).all()) == 1  # still stored, only disabled


def test_read_only_does_not_persist(tmp_path):
    rw = make(tmp_path)
    s = rw.add("a", "desc", STEPS)
    ro = make(tmp_path, read_only=True)
    added = ro.add("b", "other", STEPS)
    assert added.name == "b"
    assert len(ro.all()) == 1
    assert ro.record_outcome(s.id, False).confidence == 0.5
    reloaded = make(tmp_path).all()
    assert len(reloaded) == 1 and reloaded[0].confidence == 0.5 and reloaded[0].failures == 0


def test_read_only_missing_file_creates_nothing(tmp_path):
    make(tmp_path, read_only=True).add("b", "other", STEPS)
    assert not (tmp_path / "skills.json").exists()


class FakeModel:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, tools=None, *, think=False, max_tokens=1024):
        return ChatResult(content=self.reply)


def step(i, name="read_file", status="ok", args=None):
    return Step(
        index=i,
        tool_call=ToolCall(id=str(i), name=name, arguments=args or {"path": "a"}),
        outcome=ToolOutcome(status=status),
    )


REPLY = 'Sure:\n```json\n{"name": "read_two", "description": "Read files"}\n```'


def test_extract_builds_skill_from_successful_calls():
    steps = [
        step(0),
        step(1, status="error"),
        step(2, "delete_file", args={"path": "a", "confirmed": True}),
        Step(index=3, tool_call=None, outcome=None, planner_text="done"),
    ]
    draft = extract_skill("task", steps, FakeModel(REPLY))
    assert draft.name == "read_two" and draft.description == "Read files"
    assert [s.tool for s in draft.steps] == ["read_file", "delete_file"]
    assert draft.steps[1].arguments == {"path": "a"}


def test_extract_replaces_free_text_and_never_shows_it_to_model():
    seen = []

    class Spy(FakeModel):
        def chat(self, messages, *a, **kw):
            seen.append(messages[-1]["content"])
            return super().chat(messages)

    steps = [
        step(0, "write_file", args={"path": "notes/a.md", "content": "SECRET body"}),
        step(1, "edit_file", args={"path": "notes/a.md", "old": "x", "new": "y"}),
    ]
    draft = extract_skill("t", steps, Spy(REPLY))
    assert "SECRET" not in seen[0]
    assert draft.steps[0].arguments == {"path": "notes/a.md", "content": "{content}"}
    assert draft.steps[1].arguments == {"path": "notes/a.md", "old": "{old}", "new": "{new}"}


def test_extract_generalizes_paths_from_model_mapping():
    reply = (
        '{"name": "n", "description": "d", '
        '"placeholders": {"notes/a.md": "{note_file}", "x": "bad", "y": 3}}'
    )
    steps = [step(0, args={"path": "notes/a.md"}), step(1, args={"path": "other.md"})]
    draft = extract_skill("t", steps, FakeModel(reply))
    assert draft.steps[0].arguments == {"path": "{note_file}"}
    assert draft.steps[1].arguments == {"path": "other.md"}


def test_extract_bad_placeholder_mapping_keeps_literal_paths():
    reply = '{"name": "n", "description": "d", "placeholders": ["a"]}'
    draft = extract_skill("t", [step(0), step(1)], FakeModel(reply))
    assert draft.steps[0].arguments == {"path": "a"}


def test_extract_none_with_fewer_than_two_successes():
    assert extract_skill("t", [step(0), step(1, status="error")], FakeModel(REPLY)) is None
    assert extract_skill("t", [], FakeModel(REPLY)) is None


def test_extract_none_on_bad_model_output():
    steps = [step(0), step(1)]
    for reply in ["no json", '{"name": "x"}', '{"name": 1, "description": 2}', "{oops"]:
        assert extract_skill("t", steps, FakeModel(reply)) is None


def test_format_skills(tmp_path):
    assert format_skills([]) == ""
    s = make(tmp_path).add("inspect", "inspect a folder", STEPS)
    text = format_skills([s])
    assert "inspect" in text and "inspect a folder" in text
    assert json.dumps({"path": "{dir}"}) in text and "1. list_dir" in text


def test_corrupted_file_is_backed_up_and_store_starts_empty(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text("{not json")
    store = make(tmp_path)
    assert store.all() == []
    assert (tmp_path / "skills.json.bak").read_text() == "{not json"
    assert not path.exists()
    store.add("a", "desc", STEPS)
    assert len(make(tmp_path).all()) == 1


def test_unknown_keys_are_backed_up(tmp_path):
    (tmp_path / "skills.json").write_text(
        json.dumps([{"id": "1", "name": "n", "description": "d", "steps": [], "extra": 1}])
    )
    assert make(tmp_path).all() == []
    assert (tmp_path / "skills.json.bak").exists()
