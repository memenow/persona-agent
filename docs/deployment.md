# Deployment

The server is a single FastAPI app served by uvicorn. Most deployments
just need to set the right environment variables and run the
`persona-agent api` entrypoint.

## Local

```bash
uv sync
uv run persona-agent api
```

Default bind: `http://127.0.0.1:8000`. REST under `/api/v1`, A2A under
`/a2a/`, aggregate agent card at `/.well-known/agent.json`, Swagger UI at
`/docs`.

## Exposing the server

To bind on all interfaces:

```bash
API_HOST=0.0.0.0 API_PORT=8000 uv run persona-agent api
```

When deploying behind a reverse proxy (nginx, Caddy, Traefik, …), set
`API_PUBLIC_BASE_URL` so the generated AgentCard URLs are reachable from
outside:

```bash
export API_PUBLIC_BASE_URL=https://persona-agent.example.com
```

The aggregate card's `url` will become `https://persona-agent.example.com/a2a/`,
and each persona's card will use `https://persona-agent.example.com/a2a/{persona_id}/`.

## Configuration files

By default the server expects `config/llm_config.json` and
`config/mcp_config.json` relative to the repository root. Override with
`LLM_CONFIG_PATH` and `MCP_CONFIG_PATH` to point at deployment-specific
files (for example mounted from a Kubernetes secret).

If `mcp_config.json` is missing or has no servers, MCP is simply
disabled and the agents respond without tools.

## Auth and CORS

See [authentication.md](authentication.md). The short version: set
`API_ENABLE_AUTH=true` with a non-empty `API_ALLOWED_KEYS`, and limit
`API_ALLOWED_ORIGINS` to your trusted hostnames in production.

## Process model

- The app stores singletons (LLM client, MCP manager, persona manager,
  agent factory) on `app.state` and reuses them across requests.
- Conversation state is in-process: per-context history lives on each
  `PersonaAgentExecutor` instance; sessions and agents live in the
  `AgentFactory`. Restarting the process drops them all.
- Run a single uvicorn worker per process for now — multiple workers
  would each maintain their own state and produce inconsistent results.
  Horizontal scaling needs an external session store (not implemented).

## Logging

The root logger is configured to INFO at startup with a simple format:

```
2026-05-16 00:00:00,000 - api_server - INFO - <message>
```

Set `API_DEBUG=true` to drop uvicorn's log level to `debug`. The app's
own loggers are unaffected by `API_DEBUG`; tune them via Python's
standard `logging` config if you need more verbosity.

Errors are logged with `logger.exception(...)` so the traceback is
captured; client responses use a generic `{ "detail": "..." }` message
with no exception text. Diagnose via the server log, not the client
response.

## Health and observability

- `GET /health` is always public and reports auth posture, CORS posture,
  default model, and the count of loaded MCP tools. Use it as your
  uptime probe.
- The startup log line `Effective config: host=… port=… auth=…
  api_keys=… cors=… wildcard_origin=…` lets you confirm the resolved
  configuration without hitting the network.

## Containerization

A minimal container image looks like:

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

# Install third-party dependencies first for layer caching. The
# --no-install-project flag skips installing persona-agent itself, so
# this layer doesn't need src/ or README.md yet.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the project sources and install persona-agent into the venv.
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY examples ./examples
COPY config ./config

ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PERSONAS_DIR=/app/examples/personas \
    LLM_CONFIG_PATH=/app/config/llm_config.json \
    MCP_CONFIG_PATH=/app/config/mcp_config.json
EXPOSE 8000
CMD ["uv", "run", "persona-agent", "api"]
```

The two `uv sync` invocations are intentional: the first installs only
third-party dependencies (cached unless `pyproject.toml`/`uv.lock`
change), and the second installs the local `persona-agent` package
after its sources and the `README.md` referenced by `pyproject.toml` are
in place. Collapsing them into a single `uv sync` before `COPY src`
would fail because the project itself cannot be installed without those
files.

Mount `config/` and `examples/personas/` as volumes (or build them into
the image with the right ownership) so credentials and persona files are
under your control, not baked into the image.
