from test_memory_fakes import FakeEmbedding

from serana_agent.llm.base import ChatResult
from serana_agent.memory.chroma_store import ChromaMemoryStore
from serana_agent.memory.jsonparse import extract_json
from serana_agent.memory.reflect import MAX_TRANSCRIPT_CHARS, reflect
from serana_agent.memory.types import MemoryItem


class FakeModel:
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def chat(self, messages, tools=None, *, think=False, max_tokens=1024):
        self.calls += 1
        return ChatResult(content=self.reply)


def make_store(tmp_path):
    return ChromaMemoryStore(tmp_path / "chroma", embedding_function=FakeEmbedding())


CONVO = [
    {"role": "user", "content": "Please summarize in bullet points"},
    {"role": "assistant", "content": "Sure."},
    {"role": "tool", "content": "ignored"},
]


def test_add_search_all(tmp_path):
    store = make_store(tmp_path)
    store.add("Prefers bullet point summaries", {"session": "s1"})
    store.add("Uses a Mac laptop")
    hits = store.search("bullet point summaries", k=1)
    assert [h.text for h in hits] == ["Prefers bullet point summaries"]
    assert hits[0].metadata["session"] == "s1"
    assert len(store.all()) == 2


def test_search_empty_store(tmp_path):
    assert make_store(tmp_path).search("anything") == []


def test_persists_across_instances(tmp_path):
    make_store(tmp_path).add("Likes tea")
    assert [m.text for m in make_store(tmp_path).all()] == ["Likes tea"]


def test_is_duplicate(tmp_path):
    store = make_store(tmp_path)
    store.add("Likes green tea")
    assert store.is_duplicate("likes green tea")
    assert not store.is_duplicate("Works at a bank in Seoul")


def test_reflect_adds_facts_from_fenced_json(tmp_path):
    store = make_store(tmp_path)
    reply = 'Here you go:\n```json\n["Prefers bullet points", "Name is Jin"]\n```'
    added = reflect(CONVO, FakeModel(reply), store, {"session": "s1"})
    assert len(added) == 2
    assert {m.text for m in store.all()} == {"Prefers bullet points", "Name is Jin"}


def test_reflect_skips_duplicates(tmp_path):
    store = make_store(tmp_path)
    store.add("Prefers bullet points")
    added = reflect(CONVO, FakeModel('["prefers bullet points", "Name is Jin"]'), store)
    assert [m.text for m in added] == ["Name is Jin"]
    assert len(store.all()) == 2


def test_reflect_unparseable_adds_nothing(tmp_path):
    store = make_store(tmp_path)
    assert reflect(CONVO, FakeModel("I could not find anything"), store) == []
    assert reflect(CONVO, FakeModel('{"a": 1}'), store) == []
    assert reflect(CONVO, FakeModel("[broken"), store) == []
    assert store.all() == []


def test_reflect_ignores_non_string_items(tmp_path):
    store = make_store(tmp_path)
    reflect(CONVO, FakeModel('[1, null, "", "Likes tea"]'), store)
    assert [m.text for m in store.all()] == ["Likes tea"]


def test_reflect_empty_session_skips_model(tmp_path):
    model = FakeModel("[]")
    assert reflect([], model, make_store(tmp_path)) == []
    assert model.calls == 0


def test_extract_json_skips_bad_prefix():
    assert extract_json("see [x] then [1, 2]", "[") == [1, 2]
    assert extract_json("none", "{") is None


class ListStore:
    """Minimal MemoryStore: exact-match duplicates, no embeddings."""

    def __init__(self):
        self.items = []

    def add(self, text, metadata=None):
        item = MemoryItem(id=str(len(self.items)), text=text, metadata=metadata or {})
        self.items.append(item)
        return item

    def search(self, query, k=5):
        return self.items[:k]

    def is_duplicate(self, text):
        return any(i.text.lower() == text.lower() for i in self.items)

    def all(self):
        return list(self.items)


def test_reflect_works_with_minimal_store():
    store = ListStore()
    store.add("Likes tea")
    added = reflect(CONVO, FakeModel('["likes tea", "Name is Jin"]'), store)
    assert [m.text for m in added] == ["Name is Jin"]
    assert len(store.all()) == 2


def test_reflect_truncates_long_transcript():
    seen = []

    class Spy(FakeModel):
        def chat(self, messages, *a, **kw):
            seen.append(messages[-1]["content"])
            return super().chat(messages)

    long = [{"role": "user", "content": "x" * 50000 + "LAST"}]
    reflect(long, Spy("[]"), ListStore())
    assert len(seen[0]) <= MAX_TRANSCRIPT_CHARS and seen[0].endswith("LAST")
