# serana-mcp 이미지

MCP 툴 서버를 컨테이너에서 실행하기 위한 이미지다. 프로젝트 루트에서 빌드한다.

```
docker build -f docker/Dockerfile -t serana-mcp:latest .
```

실행은 `serana_agent.tools.launch.server_params(root, "docker")`가 만든 명령을 쓴다.
`--network none`, 호스트 uid/gid, 샌드박스 폴더만 `/sandbox`로 마운트한다.

컨테이너는 `--read-only`(쓰기 가능한 곳은 `/tmp` tmpfs와 `/sandbox`뿐), `--cap-drop ALL`,
`--security-opt no-new-privileges`, `--memory 512m`, `--pids-limit 128`로 실행한다.
