from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from mcp.client.stdio import StdioServerParameters

DOCKER_IMAGE = "serana-mcp:latest"


def server_params(root: Path, mode: Literal["host", "docker"]) -> StdioServerParameters:
    root = root.resolve()
    if mode == "host":
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "serana_agent.tools.server", "--root", str(root)],
        )
    return StdioServerParameters(
        command="docker",
        args=[
            "run", "-i", "--rm", "--network", "none",
            "--read-only", "--tmpfs", "/tmp", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--memory", "512m", "--pids-limit", "128",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{root}:/sandbox",
            DOCKER_IMAGE, "--root", "/sandbox",
        ],
    )  # fmt: skip
