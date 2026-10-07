FROM python:3.12-slim

# The server needs only the MCP SDK; build context is the project root.
RUN pip install --no-cache-dir "mcp>=2.3.0" \
    && useradd --uid 1000 --create-home serana

WORKDIR /app
COPY src/serana_agent/__init__.py /app/serana_agent/__init__.py
COPY src/serana_agent/tools/ /app/serana_agent/tools/
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
USER serana
ENTRYPOINT ["python", "-m", "serana_agent.tools.server"]
