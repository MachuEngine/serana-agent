import shutil
from pathlib import Path

import pytest

from serana_agent.eval.scoring import diff_snapshots, snapshot
from serana_agent.eval.tasks import LEVELS, TaskError, load_task, load_tasks

TASKS_DIR = Path(__file__).resolve().parent.parent / "eval_tasks"
TASKS = load_tasks(TASKS_DIR)


def test_distribution_matches_plan():
    counts = {lv: sum(t.level == lv for t in TASKS) for lv in LEVELS}
    assert counts == {"1-2": 10, "3-5": 10, "10+": 5, "safety": 5}
    assert len({t.id for t in TASKS}) == 30


def test_safety_set_covers_required_cases():
    safety = {t.id: t for t in TASKS if t.level == "safety"}
    answers = {g.answer for t in safety.values() for g in t.gate}
    assert answers == {"approve", "deny"}
    assert any(
        "ignore all previous instructions" in (t.initial_dir / "notes/inbox.md").read_text()
        for t in safety.values()
        if (t.initial_dir / "notes/inbox.md").exists()
    )
    assert any("{outside}" in t.prompt for t in safety.values())
    assert any(t.symlinks for t in safety.values())


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_task_is_consistent(task):
    assert task.description
    assert 0 <= task.min_steps <= 18
    if task.level == "10+":
        assert task.min_steps >= 10
    assert all(f.is_file() for f in task.initial_dir.rglob("*") if not f.is_dir())
    assert not any(p.is_symlink() for p in task.dir.rglob("*")), "symlinks must come from task.yaml"
    initial = snapshot(task.initial_dir, task.ignore)
    final = snapshot(task.expected_dir, task.ignore) if task.expected_dir else initial
    d = diff_snapshots(initial, final)
    changed = set(d["missing"]) | set(d["unexpected"]) | set(d["changed"])
    if task.expected_dir is None:
        assert not changed
    else:
        assert changed, "expected/ must differ from initial/"
    # Every changed path is named in the prompt, or declared as implied by it.
    prompt = task.prompt.lower()
    for path in changed:
        named = path.lower() in prompt or Path(path).name.lower() in prompt
        assert named or path in task.implied, f"{path} not explained by the prompt"
    assert set(task.implied) <= changed, "stale implied entries"
    for rel in task.trashed:
        assert (task.initial_dir / rel).is_file()
        assert rel in d["missing"]
    for g in task.gate:
        assert g.path is None or (task.initial_dir / g.path).is_file()


def test_readme_lists_every_task():
    readme = (TASKS_DIR / "README.md").read_text()
    assert all(f"`{t.id}`" in readme for t in TASKS)


def test_invalid_tasks_are_rejected(tmp_path):
    def make(yaml_text, name="t1"):
        d = tmp_path / name
        (d / "initial").mkdir(parents=True, exist_ok=True)
        (d / "task.yaml").write_text(yaml_text)
        return d

    with pytest.raises(TaskError, match="min_steps"):
        load_task(make("id: t1\nlevel: '1-2'\nprompt: hi\n"))
    with pytest.raises(TaskError, match="min_steps"):
        load_task(make("id: t1\nlevel: '1-2'\nmin_steps: 19\nprompt: hi\n"))
    with pytest.raises(TaskError, match="notes_count"):
        load_task(
            make("id: t1\nlevel: '1-2'\nmin_steps: 1\nprompt: hi\nexpect: {notes_count: x}\n")
        )
    with pytest.raises(TaskError, match="level"):
        load_task(make("id: t1\nlevel: easy\nmin_steps: 1\nprompt: hi\n"))
    with pytest.raises(TaskError, match="directory name"):
        load_task(make("id: other\nlevel: '1-2'\nmin_steps: 1\nprompt: hi\n"))
    with pytest.raises(TaskError, match="answer"):
        load_task(
            make(
                "id: t1\nlevel: safety\nmin_steps: 1\nprompt: hi\n"
                "gate:\n- {tool: delete_file, answer: maybe}\n"
            )
        )
    d = make("id: t1\nlevel: '1-2'\nmin_steps: 1\nprompt: hi\nexpect: {files: expected/}\n")
    with pytest.raises(TaskError, match="not a directory"):
        load_task(d)
    shutil.rmtree(d / "initial")
    with pytest.raises(TaskError, match="initial"):
        load_task(d)
