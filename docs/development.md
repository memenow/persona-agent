# Development

## Prerequisites

- Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/) for dependency and venv management.

## Setup

```bash
git clone https://github.com/memenow/persona-agent.git
cd persona-agent
uv sync
```

`uv sync` creates `.venv/`, installs runtime and dev dependencies from
`pyproject.toml`/`uv.lock`, and exposes the `persona-agent` script.

## Project layout

```
persona-agent/
├── config/                          Configuration files
│   ├── llm_config.json              LLM API keys and tunables
│   └── mcp_config.json              MCP stdio server definitions
├── docs/                            Detailed documentation (this directory)
├── examples/
│   └── personas/                    Example persona definitions
├── src/persona_agent/
│   ├── a2a/
│   │   ├── agent_card.py            A2A AgentCard builder
│   │   └── executor.py              PersonaAgentExecutor (A2A AgentExecutor impl)
│   ├── api/
│   │   ├── agent_factory.py         AgentFactory + AgentSession
│   │   ├── auth.py                  API key dependency
│   │   ├── config.py                ApiConfig + load_config()
│   │   ├── dependencies.py          FastAPI dependency providers
│   │   ├── models.py                Pydantic request/response models
│   │   ├── persona_manager.py       Persona model + filesystem I/O
│   │   ├── routes/                  REST and A2A discovery routes
│   │   └── server.py                FastAPI app factory + lifespan
│   ├── llm/
│   │   └── client.py                LLMClient ABC + OpenAICompatibleClient
│   ├── mcp/
│   │   └── direct_mcp.py            DirectMCPManager (stdio lifecycle)
│   └── cli.py                       Command-line interface
├── tests/                           Test suite (pytest, pytest-asyncio)
├── pyproject.toml                   Project metadata, deps, tool config
└── uv.lock                          Resolved dependency lock
```

## Common commands

```bash
# Run the API server
uv run persona-agent api

# Inspect personas / agent cards without starting the server
uv run persona-agent list-personas
uv run persona-agent agent-card               # all personas
uv run persona-agent agent-card trump         # specific persona

# Import a persona file
uv run persona-agent import-persona path/to/persona.yaml

# Lint and format
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest
uv run pytest tests/test_executor.py -q       # focused
```

## Linting

Ruff is the only linter and formatter:

- `target-version = "py311"`
- `line-length = 88`
- `select = ["E", "F", "W", "I", "UP", "B", "SIM"]`
- `ignore = ["E501", "B008"]` — long lines are tolerated; `B008`
  (function calls in default args) is allowed because FastAPI's
  `Depends(...)` idiom relies on it.

Run `uv run ruff format .` before opening a pull request. CI is not
mandated by the project today, but ruff's defaults are good enough that
running it locally keeps diffs clean.

## Tests

`pytest-asyncio` is configured in `asyncio_mode = "auto"` so async tests
do not need an explicit decorator. Tests live under `tests/`; group
related cases per file (`test_auth.py`, `test_executor.py`,
`test_persona_id_security.py`, etc.).

Key test conventions:

- Build a real `Persona` and `PersonaAgentExecutor` rather than mocking
  them — the executor is small and the integration is the interesting
  surface.
- For LLM calls, stub `OpenAICompatibleClient` with a fake that returns
  a deterministic `ChatResponse` rather than hitting an external API.
- For MCP, point `DirectMCPManager.load_config` at a small JSON in a
  `tmp_path` fixture and use a no-op stdio server, or skip when the
  upstream binary is unavailable.

## Git workflow

The repository follows a fork-based workflow:

- `origin` is your fork (e.g. `github.com/<you>/persona-agent`).
- `upstream` is `github.com/memenow/persona-agent`.

Open pull requests against `upstream/main`. Keep commit messages
factual and in English; do not include AI-attribution lines.

## Adding documentation

This directory is the canonical place for in-depth documentation. When
you change behavior that's documented here, update the matching file in
the same change. The top-level `README.md` should stay short — link from
it into `docs/` rather than expanding the README.
