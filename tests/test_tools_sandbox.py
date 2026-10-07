import os

import pytest

from serana_agent.tools.sandbox import SandboxViolation, resolve_in_root


@pytest.fixture
def layout(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (root / "ok.txt").write_text("ok")
    os.symlink(outside / "secret.txt", root / "link_file")
    os.symlink(outside, root / "link_dir")
    os.symlink(root / "ok.txt", root / "inner_link")
    return root, outside


def test_inside_paths_allowed(layout):
    root, _ = layout
    assert resolve_in_root(root, "ok.txt") == (root / "ok.txt").resolve()
    assert resolve_in_root(root, "new/dir/file.txt").is_relative_to(root.resolve())
    assert resolve_in_root(root, ".") == root.resolve()
    assert resolve_in_root(root, "inner_link") == (root / "ok.txt").resolve()


@pytest.mark.parametrize("bad", ["../outside/secret.txt", "a/../../x", "..", "a/../b"])
def test_dotdot_rejected(layout, bad):
    with pytest.raises(SandboxViolation):
        resolve_in_root(layout[0], bad)


def test_absolute_rejected_even_inside_root(layout):
    root, outside = layout
    for p in (str(outside / "secret.txt"), str(root / "ok.txt"), "/etc/passwd", "~/x"):
        with pytest.raises(SandboxViolation):
            resolve_in_root(root, p)


def test_symlink_to_outside_file_rejected(layout):
    with pytest.raises(SandboxViolation):
        resolve_in_root(layout[0], "link_file")


def test_symlinked_parent_dir_rejected(layout):
    root, _ = layout
    for p in ("link_dir/secret.txt", "link_dir/new.txt", "link_dir/sub/new.txt"):
        with pytest.raises(SandboxViolation):
            resolve_in_root(root, p)


def test_dangling_symlink_outside_rejected(layout):
    root, outside = layout
    os.symlink(outside / "missing.txt", root / "dangling")
    with pytest.raises(SandboxViolation):
        resolve_in_root(root, "dangling")


def test_null_byte_rejected(layout):
    with pytest.raises(SandboxViolation):
        resolve_in_root(layout[0], "a\x00b")
