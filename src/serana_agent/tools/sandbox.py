from __future__ import annotations

from pathlib import Path


class SandboxViolation(Exception):
    """The requested path is outside the sandbox root."""


def resolve_in_root(root: Path, user_path: str) -> Path:
    """Resolve a root-relative path, rejecting anything that ends up outside `root`."""
    root = root.resolve()
    if "\x00" in user_path:
        raise SandboxViolation("path contains a null byte")
    # Absolute input is rejected, not re-rooted: silently remapping would hide model mistakes.
    if Path(user_path).is_absolute() or user_path.startswith(("~", "\\")):
        raise SandboxViolation(f"absolute path is not allowed: {user_path!r}; use a relative path")
    if ".." in Path(user_path).parts:
        raise SandboxViolation(f"'..' is not allowed in paths: {user_path!r}")
    # resolve() follows symlinks in every component, including ones that do not exist yet
    # past the last real parent, so linked parents and dangling links are covered.
    resolved = (root / user_path).resolve()
    if not resolved.is_relative_to(root):
        raise SandboxViolation(f"path resolves outside the sandbox: {user_path!r}")
    return resolved


def resolve_link(root: Path, user_path: str) -> Path:
    """Like resolve_in_root, but the last component is not followed.

    Used for move/delete so a symlink is operated on itself, not on its target.
    """
    p = Path(user_path)
    if p.name in ("", ".", ".."):
        raise SandboxViolation(f"invalid path: {user_path!r}")
    return resolve_in_root(root, str(p.parent)) / p.name
