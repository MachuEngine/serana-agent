import sys

from serana_agent.tools.launch import server_params


def test_host_params(tmp_path):
    p = server_params(tmp_path, "host")
    assert p.command == sys.executable
    assert p.args[-2:] == ["--root", str(tmp_path.resolve())]


def test_docker_params(tmp_path):
    p = server_params(tmp_path, "docker")
    assert p.command == "docker"
    assert "--network" in p.args and p.args[p.args.index("--network") + 1] == "none"
    assert f"{tmp_path.resolve()}:/sandbox" in p.args
    assert p.args[-3:] == ["serana-mcp:latest", "--root", "/sandbox"]


def test_docker_hardening(tmp_path):
    a = server_params(tmp_path, "docker").args
    for flag in ("--read-only", "--cap-drop", "--security-opt", "--memory", "--pids-limit"):
        assert flag in a
    assert a[a.index("--cap-drop") + 1] == "ALL"
