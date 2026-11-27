# Development

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/memenow/persona-agent.git
cd persona-agent
uv sync
```

## Architecture

Dependencies flow strictly downward from `cli` and `api` into domain and integration packages:

- `persona_agent.config` — `Settings`, `load_settings()`, `load_llm_config()`.
- `persona_agent.personas` — `Persona` model, prompt builder, and `PersonaRepository`.
- `persona_agent.llm` — `LLMClient` ABC and `OpenAICompatibleClient` (`openai>=3.60.0`).
- `persona_agent.mcp` — `MCPManager` and tool schema conversion (`mcp>=2.3.0`).
- `persona_agent.a2a` — A2A v1.0 `AgentCard` builders, `PersonaAgentExecutor`, and `A2ARegistry` (`a2a-sdk>=1.3.1`).
- `persona_agent.agents` — `AgentService`, `AgentRecord`, and `AgentSession`.
- `persona_agent.api` — Synchronous `create_app()`, `lifespan()`, API key auth, `app.state` dependencies, Pydantic schemas, and routers (`a2a`, `agents`, `personas`, `sessions`).
- `persona_agent.cli` — CLI wrapper around `run_server`, `PersonaRepository`, and `build_agent_card`.

## Commands

```bash
# Run server
uv run persona-agent api

# CLI inspection
uv run persona-agent list-personas
uv run persona-agent agent-card
uv run persona-agent agent-card trump
uv run persona-agent import-persona path/to/persona.yaml

# Lint & format (configured in pyproject.toml)
uv run ruff check .
uv run ruff format .

# Test suite
uv run pytest
uv run pytest tests/test_a2a.py -v
```

## Code style

- Type annotations on public functions and route handlers (`Annotated[..., Depends(...)]`).
- Ruff configuration lives exclusively in `pyproject.toml`: `target-version = "py311"`, `line-length = 88`, `select = ["E", "F", "W", "I", "UP", "B", "SIM"]`, `ignore = ["E501"]`.
- Pydantic v2 APIs only (`model_validate`, `model_dump`).
- Async-first with `pytest-asyncio` in `asyncio_mode = "auto"`.

## Test layout

Tests are grouped by module under `tests/`:

- `test_a2a.py` — A2A v1.0 AgentCard discovery, sub-app isolation, and v1.0/v0.3 JSON-RPC.
- `test_agents.py` — `AgentService`, `AgentSession` timestamps, and MCP reuse.
- `test_api.py` — REST routes, `/health`, error handling, upload validation, and rollback behavior.
- `test_auth.py` — API key enforcement, custom headers, and constant-time comparison.
- `test_config.py` — Env parsing, CSV sanitization, and public base URL resolution.
- `test_executor.py` — A2A `Task` lifecycle, concurrency locks, and tool-iteration cap.
- `test_llm.py` — `OpenAICompatibleClient` credential and model precedence.
- `test_mcp.py` — Env substitution, MCP v2 `input_schema` conversion, and error handling.
- `test_personas.py` — Persona ID safety, prompt rendering, and repository file I/O.

## Git workflow

- `origin`: personal fork.
- `upstream`: `memenow/persona-agent`.
- Open pull requests against `upstream/main` with factual English commit messages and updated docs when behavior changes.
