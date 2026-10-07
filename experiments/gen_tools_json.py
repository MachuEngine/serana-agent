"""Regenerate experiments/data/tools.json from the live MCP tool server.

Usage: uv run python experiments/gen_tools_json.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from serana_agent.tools.client import ToolClient

TOOLS_JSON = Path(__file__).parent / "data" / "tools.json"


async def fetch_tools() -> list[dict]:
    with tempfile.TemporaryDirectory() as root:
        async with ToolClient(Path(root)) as client:
            return [asdict(t) for t in await client.list_tools()]


def main() -> None:
    tools = asyncio.run(fetch_tools())
    TOOLS_JSON.write_text(json.dumps(tools, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(tools)} tools to {TOOLS_JSON}")


if __name__ == "__main__":
    main()
