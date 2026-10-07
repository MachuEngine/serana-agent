import os
import shutil
import subprocess

import pytest

from serana_agent.tools.client import ToolClient
from serana_agent.tools.protocol import MAX_OUTPUT_CHARS, Risk, ToolOutcome
from serana_agent.tools.server import build_server


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    r.mkdir()
    return r


def with_client(fn):
    """Open the client inside the test's own task (anyio scopes cannot cross fixture tasks)."""

    async def wrapper(root, tmp_path):
        async with ToolClient(root, "host") as client:
            return await fn(client, root, tmp_path)

    wrapper.__name__ = fn.__name__  # no functools.wraps: pytest would see fn's signature
    return wrapper


@with_client
async def test_list_tools_hides_confirmed(client, root, tmp_path):
    specs = {s.name: s for s in await client.list_tools()}
    assert set(specs) == {
        "list_dir", "read_file", "search_files", "write_file", "edit_file",
        "move_file", "delete_file", "add_note", "list_notes",
    }  # fmt: skip
    for s in specs.values():
        assert "confirmed" not in s.input_schema["properties"]
        assert "confirmed" not in s.input_schema.get("required", [])
    assert specs["write_file"].input_schema["required"] == ["path", "content"]


@with_client
async def test_write_read_list_roundtrip(client, root, tmp_path):
    out = await client.call("write_file", {"path": "a/b.txt", "content": "hello"})
    assert out.status == "ok" and out.risk == Risk.WRITE
    assert (root / "a/b.txt").read_text() == "hello"
    assert (await client.call("read_file", {"path": "a/b.txt"})).output == "hello"
    assert (await client.call("list_dir", {"path": "."})).output == "a/"


@with_client
async def test_overwrite_needs_confirmation(client, root, tmp_path):
    (root / "f.txt").write_text("old")
    out = await client.call("write_file", {"path": "f.txt", "content": "new"})
    assert out.status == "confirmation_required" and out.risk == Risk.DESTRUCTIVE
    assert out.change and (root / "f.txt").read_text() == "old"
    out = await client.call("write_file", {"path": "f.txt", "content": "new"}, confirmed=True)
    assert out.status == "ok" and (root / "f.txt").read_text() == "new"


@with_client
async def test_model_cannot_self_confirm(client, root, tmp_path):
    (root / "f.txt").write_text("old")
    out = await client.call("delete_file", {"path": "f.txt", "confirmed": True})
    assert out.status == "confirmation_required" and (root / "f.txt").exists()


@with_client
async def test_delete_goes_to_trash_with_collision(client, root, tmp_path):
    for _ in range(2):
        (root / "f.txt").write_text("x")
        assert (
            await client.call("delete_file", {"path": "f.txt"})
        ).status == "confirmation_required"
        out = await client.call("delete_file", {"path": "f.txt"}, confirmed=True)
        assert out.status == "ok"
    assert not (root / "f.txt").exists()
    assert len(list((root / ".trash").iterdir())) == 2


@with_client
async def test_move_rules(client, root, tmp_path):
    (root / "a.txt").write_text("a")
    (root / "b.txt").write_text("b")
    assert (await client.call("move_file", {"src": "a.txt", "dst": "c.txt"})).status == "ok"
    out = await client.call("move_file", {"src": "c.txt", "dst": "b.txt"})
    assert out.status == "confirmation_required"
    out = await client.call("move_file", {"src": "c.txt", "dst": "b.txt"}, confirmed=True)
    assert out.status == "ok" and (root / "b.txt").read_text() == "a"
    assert (await client.call("move_file", {"src": "nope", "dst": "x"})).status == "error"


@with_client
async def test_edit_file_uniqueness(client, root, tmp_path):
    (root / "f.txt").write_text("a b a")
    out = await client.call("edit_file", {"path": "f.txt", "old": "a", "new": "z"})
    assert out.status == "error" and "2 times" in out.error
    out = await client.call("edit_file", {"path": "f.txt", "old": "q", "new": "z"})
    assert out.status == "error" and "not found" in out.error
    out = await client.call("edit_file", {"path": "f.txt", "old": "b", "new": "z"})
    assert out.status == "ok" and (root / "f.txt").read_text() == "a z a"


@with_client
async def test_search_and_hidden(client, root, tmp_path):
    (root / "notes_todo.txt").write_text("nothing")
    (root / "x.txt").write_text("line1\nNeedle here\n")
    (root / ".trash").mkdir()
    (root / ".trash" / "needle.txt").write_text("needle")
    out = (await client.call("search_files", {"query": "needle"})).output
    assert "x.txt:2: Needle here" in out and ".trash" not in out
    out = (await client.call("search_files", {"query": "todo"})).output
    assert out == "notes_todo.txt"
    await client.call("add_note", {"text": "hi"})
    assert ".notes/" not in (await client.call("list_dir", {})).output


@with_client
async def test_search_skips_symlink_to_outside(client, root, tmp_path):
    (tmp_path / "out.txt").write_text("needle")
    os.symlink(tmp_path / "out.txt", root / "l.txt")
    assert (await client.call("search_files", {"query": "needle"})).output == "no matches"


@with_client
async def test_notes(client, root, tmp_path):
    assert (await client.call("list_notes")).output == "(no notes)"
    await client.call("add_note", {"text": "buy milk"})
    assert "buy milk" in (await client.call("list_notes")).output


@with_client
async def test_truncation(client, root, tmp_path):
    (root / "big.txt").write_text("x" * (MAX_OUTPUT_CHARS + 500))
    out = (await client.call("read_file", {"path": "big.txt"})).output
    assert out.startswith("x" * MAX_OUTPUT_CHARS) and "of 4500; next offset=4000]" in out
    out = (await client.call("read_file", {"path": "big.txt", "offset": 4000})).output
    assert out == "x" * 500
    out = (await client.call("read_file", {"path": "big.txt", "offset": 10, "limit": 5})).output
    assert out.startswith("xxxxx\n...[showing chars 10-15 of 4500")


@with_client
async def test_sandbox_escapes_through_tools(client, root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "s.txt").write_text("secret")
    os.symlink(outside, root / "link_dir")
    os.symlink(outside / "s.txt", root / "link_file")
    cases = [
        ("read_file", {"path": "../outside/s.txt"}),
        ("read_file", {"path": str(outside / "s.txt")}),
        ("read_file", {"path": "link_file"}),
        ("read_file", {"path": "link_dir/s.txt"}),
        ("list_dir", {"path": "link_dir"}),
        ("write_file", {"path": "link_dir/new.txt", "content": "x"}),
        ("write_file", {"path": "../new.txt", "content": "x"}),
        ("move_file", {"src": "link_file", "dst": "../x"}),
        ("move_file", {"src": "link_dir/s.txt", "dst": "x"}),
        ("edit_file", {"path": "link_dir/s.txt", "old": "secret", "new": "x"}),
    ]
    for tool, args in cases:
        out = await client.call(tool, args)
        assert out.status == "error", (tool, args)
    # an outside-pointing link may be renamed (only the link moves), never its target
    assert (await client.call("move_file", {"src": "link_file", "dst": "renamed"})).status == "ok"
    assert (outside / "s.txt").read_text() == "secret"
    assert not (outside / "new.txt").exists() and not (tmp_path / "new.txt").exists()


@with_client
async def test_invalid_arguments_become_error(client, root, tmp_path):
    out = await client.call("read_file", {})
    assert out.status == "error"
    out = await client.call("no_such_tool", {})
    assert out.status == "error"


@with_client
async def test_move_symlink_moves_link_not_target(client, root, tmp_path):
    (root / "real.txt").write_text("r")
    os.symlink(root / "real.txt", root / "lnk.txt")
    assert (await client.call("move_file", {"src": "lnk.txt", "dst": "moved.txt"})).status == "ok"
    assert (root / "real.txt").read_text() == "r"
    assert (root / "moved.txt").is_symlink() and not (root / "lnk.txt").is_symlink()
    # symlink as destination is replaced, not followed
    (root / "other.txt").write_text("o")
    os.symlink(root / "real.txt", root / "dstlink")
    out = await client.call("move_file", {"src": "other.txt", "dst": "dstlink"}, confirmed=True)
    assert out.status == "ok"
    assert (root / "real.txt").read_text() == "r" and (root / "dstlink").read_text() == "o"


@with_client
async def test_delete_symlink_moves_link(client, root, tmp_path):
    (root / "real.txt").write_text("r")
    os.symlink(root / "real.txt", root / "lnk.txt")
    assert (await client.call("delete_file", {"path": "lnk.txt"}, confirmed=True)).status == "ok"
    assert (root / "real.txt").exists() and (root / ".trash" / "lnk.txt").is_symlink()


@with_client
async def test_search_glob_validation(client, root, tmp_path):
    (root / "a.txt").write_text("x")
    for g in ("/etc/*", "", "../*", "../root/*"):
        out = await client.call("search_files", {"query": "x", "glob": g})
        assert out.status == "error", g
    assert (await client.call("search_files", {"query": "x", "glob": "*.txt"})).status == "ok"


@with_client
async def test_reserved_folders_rejected(client, root, tmp_path):
    (root / "f.txt").write_text("x")
    (root / ".trash").mkdir()
    (root / ".trash" / "t.txt").write_text("t")
    os.symlink(root / ".trash", root / "trashlink")
    cases = [
        ("move_file", {"src": "f.txt", "dst": ".trash/x"}),
        ("write_file", {"path": ".trash/n.txt", "content": "x"}),
        ("write_file", {"path": ".notes/n.txt", "content": "x"}),
        ("edit_file", {"path": ".trash/t.txt", "old": "t", "new": "u"}),
        ("read_file", {"path": ".trash/t.txt"}),
        ("list_dir", {"path": ".trash"}),
        ("list_dir", {"path": "trashlink"}),
        ("read_file", {"path": "trashlink/t.txt"}),
        ("delete_file", {"path": ".trash/t.txt"}),
    ]
    for tool, args in cases:
        assert (await client.call(tool, args)).status == "error", (tool, args)
    assert (root / "f.txt").exists() and (root / ".trash" / "t.txt").read_text() == "t"


@with_client
async def test_edit_preserves_crlf(client, root, tmp_path):
    (root / "w.txt").write_bytes(b"a\r\nb\r\nc\r\n")
    out = await client.call("edit_file", {"path": "w.txt", "old": "a\r\nb", "new": "X\r\nY"})
    assert out.status == "ok"
    assert (root / "w.txt").read_bytes() == b"X\r\nY\r\nc\r\n"


@with_client
async def test_list_notes_survives_malformed_lines(client, root, tmp_path):
    await client.call("add_note", {"text": "good"})
    with (root / ".notes" / "notes.jsonl").open("a") as fh:
        fh.write('not json\n{"ts": 1}\n')
    out = (await client.call("list_notes")).output
    assert "good" in out and "(malformed) not json" in out and "(malformed) {" in out


async def test_call_timeout_returns_error(root):
    async with ToolClient(root, "host", timeout=0.0001) as c:
        out = await c.call("list_dir", {})
    assert out.status == "error" and "timed out" in out.error


def _direct(root, name, **args):
    """Call the server's tool function itself, bypassing ToolClient."""
    server = build_server(root)
    return ToolOutcome.from_json(server._tool_manager.get_tool(name).fn(**args))


def test_server_enforces_confirmation_without_client(root):
    (root / "f.txt").write_text("old")
    (root / "g.txt").write_text("g")
    out = _direct(root, "delete_file", path="f.txt")
    assert out.status == "confirmation_required" and (root / "f.txt").exists()
    out = _direct(root, "write_file", path="f.txt", content="new")
    assert out.status == "confirmation_required" and (root / "f.txt").read_text() == "old"
    out = _direct(root, "move_file", src="g.txt", dst="f.txt")
    assert out.status == "confirmation_required" and (root / "g.txt").exists()
    assert _direct(root, "delete_file", path="f.txt", confirmed=True).status == "ok"
    assert not (root / "f.txt").exists()


@with_client
async def test_move_into_folder(client, root, tmp_path):
    (root / "a.txt").write_text("a")
    out = await client.call("move_file", {"src": "a.txt", "dst": "inbox/read/"})
    assert out.status == "ok"
    assert (root / "inbox/read").is_dir() and (root / "inbox/read/a.txt").read_text() == "a"
    # existing directory without trailing slash
    (root / "b.txt").write_text("b")
    assert (await client.call("move_file", {"src": "b.txt", "dst": "inbox/read"})).status == "ok"
    assert (root / "inbox/read/b.txt").read_text() == "b"
    # overwrite gate applies to the final target
    (root / "a.txt").write_text("new")
    out = await client.call("move_file", {"src": "a.txt", "dst": "inbox/read/"})
    assert out.status == "confirmation_required" and (root / "a.txt").exists()
    out = await client.call("move_file", {"src": "a.txt", "dst": "inbox/read/"}, confirmed=True)
    assert out.status == "ok" and (root / "inbox/read/a.txt").read_text() == "new"
    out = await client.call("move_file", {"src": "inbox", "dst": "inbox/"})
    assert out.status == "error"


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.mark.docker
@pytest.mark.skipif(not _docker_ok(), reason="docker daemon not available")
async def test_docker_mode_roundtrip(root):
    image = subprocess.run(["docker", "image", "inspect", "serana-mcp:latest"], capture_output=True)
    if image.returncode != 0:
        pytest.skip("serana-mcp:latest not built")
    async with ToolClient(root, "docker") as c:
        assert (await c.call("write_file", {"path": "d.txt", "content": "hi"})).status == "ok"
    assert (root / "d.txt").read_text() == "hi"
