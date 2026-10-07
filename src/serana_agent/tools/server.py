"""MCP tool server. Run: python -m serana_agent.tools.server --root <dir> (stdio)."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from serana_agent.tools.protocol import (
    MAX_OUTPUT_CHARS,
    TOOL_RISK,
    TRUNCATION_MARKER,
    Risk,
    ToolOutcome,
)
from serana_agent.tools.sandbox import SandboxViolation, resolve_in_root, resolve_link

HIDDEN = {".trash", ".notes"}
MAX_SEARCH_RESULTS = 50
MAX_GLOB_VISITS = 10_000
MAX_SEARCH_FILE_BYTES = 1_000_000


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    marker = TRUNCATION_MARKER.format(omitted=len(text) - MAX_OUTPUT_CHARS)
    return text[:MAX_OUTPUT_CHARS] + marker


def _ok(tool: str, output: str = "", **details) -> str:
    return ToolOutcome("ok", _truncate(output), risk=TOOL_RISK[tool], details=details).to_json()


def _error(tool: str, message: str, **details) -> str:
    return ToolOutcome("error", error=message, risk=TOOL_RISK[tool], details=details).to_json()


def _confirm(tool: str, change: str) -> str:
    return ToolOutcome("confirmation_required", risk=Risk.DESTRUCTIVE, change=change).to_json()


def build_server(root: Path) -> MCPServer:
    root = root.resolve()
    server = MCPServer("serana-tools")

    def rel(p: Path) -> str:
        return p.relative_to(root).as_posix() or "."

    def guarded(tool: str, fn: Callable[[], str]) -> str:
        """Run a tool body, turning expected failures into messages the model can act on."""
        try:
            return fn()
        except SandboxViolation as e:
            # Flagged so the eval harness can count escape attempts without parsing messages.
            return _error(tool, str(e), sandbox_violation=True)
        except (OSError, UnicodeError) as e:
            return _error(tool, f"{type(e).__name__}: {e}")
        except (ValueError, NotImplementedError) as e:
            return _error(tool, f"invalid argument: {e}")

    def reserved(p: Path) -> bool:
        """True if p, or where it resolves to, is inside .trash/ or .notes/."""
        for q in (p, p.resolve()):
            try:
                parts = q.relative_to(root).parts
            except ValueError:
                continue
            if parts and parts[0].lower() in HIDDEN:
                return True
        return False

    def visible(p: Path) -> bool:
        return not reserved(p)

    def resolve_visible(path: str) -> Path:
        p = resolve_in_root(root, path)
        if reserved(p):
            raise SandboxViolation(f"{path!r} is in a reserved folder (.trash/.notes)")
        return p

    def link_visible(path: str) -> Path:
        """Path of the entry itself (symlinks not followed), outside reserved folders."""
        p = resolve_link(root, path)
        if reserved(p) or p.parent == root and p.name.lower() in HIDDEN:
            raise SandboxViolation(f"{path!r} is in a reserved folder (.trash/.notes)")
        return p

    def exists(p: Path) -> bool:
        return p.exists() or p.is_symlink()

    @server.tool()
    def list_dir(path: str = ".") -> str:
        """List files and folders in a directory (folders end with '/')."""

        def run() -> str:
            d = resolve_visible(path)
            if not d.is_dir():
                return _error("list_dir", f"not a directory: {path}")
            names = sorted(c.name + ("/" if c.is_dir() else "") for c in d.iterdir() if visible(c))
            return _ok("list_dir", "\n".join(names) or "(empty)")

        return guarded("list_dir", run)

    @server.tool()
    def read_file(path: str, offset: int = 0, limit: int = MAX_OUTPUT_CHARS) -> str:
        """Read a text file. Returns at most `limit` characters starting at character `offset`;
        if the file is longer, the output says the total size and the next offset to use."""

        def run() -> str:
            f = resolve_visible(path)
            if not f.is_file():
                return _error("read_file", f"file not found: {path}")
            if offset < 0 or limit < 1:
                return _error("read_file", "offset must be >= 0 and limit >= 1")
            with f.open(encoding="utf-8", errors="replace", newline="") as fh:
                text = fh.read()
            total = len(text)
            end = min(total, offset + min(limit, MAX_OUTPUT_CHARS))
            out = text[offset:end]
            if end < total:
                out += f"\n...[showing chars {offset}-{end} of {total}; next offset={end}]"
            return ToolOutcome(
                "ok", out, risk=TOOL_RISK["read_file"], details={"total_chars": total}
            ).to_json()

        return guarded("read_file", run)

    @server.tool()
    def search_files(query: str, glob: str = "**/*") -> str:
        """Search file names and file contents (case-insensitive) for a text query.
        `glob` is relative to the root (no absolute paths or '..')."""

        def run() -> str:
            if not glob or Path(glob).is_absolute() or ".." in Path(glob).parts:
                return _error("search_files", f"invalid glob {glob!r}: use a relative pattern")
            q = query.lower()
            hits: list[str] = []
            for f in sorted(itertools.islice(root.glob(glob), MAX_GLOB_VISITS)):
                if len(hits) >= MAX_SEARCH_RESULTS:
                    break
                if not f.is_file() or not f.resolve().is_relative_to(root) or not visible(f):
                    continue
                name = rel(f)
                if q in name.lower():
                    hits.append(name)
                    continue
                if f.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                text = f.read_text(encoding="utf-8", errors="replace")
                for n, line in enumerate(text.splitlines(), 1):
                    if q in line.lower():
                        hits.append(f"{name}:{n}: {line.strip()[:200]}")
                        break
            return _ok("search_files", "\n".join(hits) or "no matches", count=len(hits))

        return guarded("search_files", run)

    @server.tool()
    def write_file(path: str, content: str, confirmed: bool = False) -> str:
        """Create a file, or overwrite it if it exists (overwriting needs user confirmation)."""

        def run() -> str:
            f = resolve_visible(path)
            if f == root or f.is_dir():
                return _error("write_file", f"path is a directory: {path}")
            if f.exists() and not confirmed:
                return _confirm("write_file", f"Overwrite existing file {path}")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")
            return _ok("write_file", f"wrote {len(content)} chars to {path}")

        return guarded("write_file", run)

    @server.tool()
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace `old` with `new` in a file. `old` must appear exactly once."""

        def run() -> str:
            f = resolve_visible(path)
            if not f.is_file():
                return _error("edit_file", f"file not found: {path}")
            if not old:
                return _error("edit_file", "`old` must not be empty")
            # newline="" keeps CRLF as-is so `old` can match and untouched lines stay unchanged.
            with f.open(encoding="utf-8", newline="") as fh:
                text = fh.read()
            n = text.count(old)
            if n == 0:
                return _error("edit_file", f"`old` text not found in {path}; read the file first")
            if n > 1:
                return _error(
                    "edit_file",
                    f"`old` appears {n} times in {path}; include more surrounding text",
                )
            with f.open("w", encoding="utf-8", newline="") as fh:
                fh.write(text.replace(old, new, 1))
            return _ok("edit_file", f"edited {path}")

        return guarded("edit_file", run)

    @server.tool()
    def move_file(src: str, dst: str, confirmed: bool = False) -> str:
        """Move or rename a file or folder. If `dst` ends with '/' or is an existing folder, the
        item is moved into it. Replacing an existing file needs confirmation."""

        def run() -> str:
            s = link_visible(src)
            d = link_visible(dst)
            if not exists(s):
                return _error("move_file", f"source not found: {src}")
            shown = dst
            if dst.endswith("/") or d.is_dir():
                # "into folder" semantics; overwrite checks below use the final target
                target = f"{dst.rstrip('/')}/{s.name}"
                d = link_visible(target)
                shown = target
            if d.is_relative_to(s):
                return _error("move_file", f"invalid destination: {shown}")
            if exists(d):
                if (d.is_dir() and not d.is_symlink()) or (s.is_dir() and not s.is_symlink()):
                    return _error("move_file", f"destination already exists: {shown}")
                if not confirmed:
                    return _confirm(
                        "move_file", f"Move {src} to {shown}, replacing existing {shown}"
                    )
            d.parent.mkdir(parents=True, exist_ok=True)
            os.replace(s, d)  # renames the link itself; shutil.move would follow a dir symlink
            return _ok("move_file", f"moved {src} to {shown}")

        return guarded("move_file", run)

    @server.tool()
    def delete_file(path: str, confirmed: bool = False) -> str:
        """Delete a file by moving it to .trash/ (always needs user confirmation)."""

        def run() -> str:
            f = link_visible(path)
            if not f.is_symlink() and not f.is_file():
                return _error("delete_file", f"file not found: {path}")
            if not confirmed:
                return _confirm("delete_file", f"Move {path} to .trash/")
            trash = root / ".trash"
            trash.mkdir(exist_ok=True)
            target = trash / f.name
            if exists(target):
                target = trash / f"{f.name}.{time.strftime('%Y%m%d%H%M%S')}-{time.monotonic_ns()}"
            os.replace(f, target)
            return _ok("delete_file", f"moved {path} to {rel(target)}")

        return guarded("delete_file", run)

    @server.tool()
    def add_note(text: str) -> str:
        """Save a short note."""

        def run() -> str:
            notes = root / ".notes" / "notes.jsonl"
            notes.parent.mkdir(exist_ok=True)
            entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "text": text}
            with notes.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return _ok("add_note", "note saved")

        return guarded("add_note", run)

    @server.tool()
    def list_notes() -> str:
        """List saved notes."""

        def run() -> str:
            notes = root / ".notes" / "notes.jsonl"
            if not notes.is_file():
                return _ok("list_notes", "(no notes)")
            lines = []
            for raw in notes.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    e = json.loads(raw)
                    lines.append(f"[{e['ts']}] {e['text']}")
                except (ValueError, KeyError, TypeError):
                    lines.append(f"(malformed) {raw}")
            return _ok("list_notes", "\n".join(lines) or "(no notes)")

        return guarded("list_notes", run)

    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"root is not a directory: {args.root}")
    build_server(args.root).run("stdio")


if __name__ == "__main__":
    main()
