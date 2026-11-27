# AGENTS.md

Repository guide for working in `persona-agent`.

## Project

Python service exposing persona agents over the A2A v1.0 protocol and a FastAPI REST API, with OpenAI-compatible LLM backends and MCP stdio tool support. Example personas live in `examples/personas/`.

## Commands

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest

uv run persona-agent api
uv run persona-agent list-personas
uv run persona-agent agent-card [persona_id]
uv run persona-agent import-persona FILE
```

## Architecture

Layered packages under `src/persona_agent/`:

- `__init__.py` — Package version from `importlib.metadata`.
- `config.py` — `Settings`, `load_settings()`, `load_llm_config()`.
- `personas/` — `Persona` Pydantic model (`id` pattern `^[a-z0-9_-]{1,64}$`), `build_system_prompt()`, and `PersonaRepository` filesystem I/O with path-traversal checks.
- `llm/` — `LLMClient` ABC, `OpenAICompatibleClient` (`openai>=3.60.0`), `ChatResponse`, `ToolCall`.
- `mcp/` — `MCPManager` and `MCPConnection` (`mcp>=2.3.0`, `mcp.Client`, `tool.input_schema`, `MAX_RETRIES=3`, `${VAR_NAME}` env substitution).
- `a2a/` — A2A v1.0 (`a2a-sdk>=1.3.1`): `cards.py` (`AgentCard` Protobuf builders + `serialize_agent_card`), `executor.py` (`PersonaAgentExecutor` using `Task` + `TaskUpdater`, per-context `asyncio.Lock`, 200-entry LRU history, `MAX_TOOL_ITERATIONS=10`), `registry.py` (`A2ARegistry` mounting per-persona FastAPI sub-apps via `add_a2a_routes_to_fastapi` with `enable_v0_3_compat=True`).
- `agents/` — `AgentService`, `AgentRecord`, `AgentSession` (wall-clock timestamps).
- `api/` — Synchronous `create_app()` and `lifespan()` in `app.py`, `auth.py` (`make_api_key_dependency` with `secrets.compare_digest`), `dependencies.py` (`app.state`-backed), `schemas.py`, and `routes/` (`a2a.py`, `agents.py`, `personas.py`, `sessions.py`).
- `cli.py` — Command-line interface.

## Configuration

Precedence: environment variables > JSON config files > `Settings` defaults (except `api_key`, which checks per-model and file-level values before `OPENAI_API_KEY`).

- Server: `API_HOST` (default `127.0.0.1`), `API_PORT`, `API_DEBUG`, `API_PREFIX`, `API_PUBLIC_BASE_URL`.
- Auth/CORS: `API_ENABLE_AUTH`, `API_KEY_HEADER`, `API_ALLOWED_KEYS`, `API_ENABLE_CORS`, `API_ALLOWED_ORIGINS`.
- Paths: `PERSONAS_DIR`, `LLM_CONFIG_PATH`, `MCP_CONFIG_PATH`.
- LLM: `OPENAI_API_KEY`, `OPENAI_BASE_URL` / `OPENAI_API_BASE`, `DEFAULT_MODEL`.

## Endpoints

- Public A2A: `GET /.well-known/agent-card.json`, `GET /.well-known/agent.json`, `GET /a2a/personas`, `GET /a2a/{persona_id}/.well-known/agent-card.json`, `POST /a2a/{persona_id}/`.
- Protected REST (when `API_ENABLE_AUTH=true`): `/api/v1/personas`, `/api/v1/agents`, `/api/v1/sessions`.
- Public Ops: `GET /health`, `GET /docs`, `GET /openapi.json`.

## Documentation

- `README.md` — quick start and navigation.
- `CHANGELOG.md` — Keep a Changelog release history.
- `docs/` — `api.md`, `a2a.md`, `mcp.md`, `personas.md`, `configuration.md`, `development.md`. Keep matching docs updated when changing documented behavior.

## Code Style & Workflow

- Python 3.11+ with type hints and `Annotated[..., Depends(...)]`.
- Ruff configured in `pyproject.toml` (`target-version = "py311"`, `line-length = 88`, `select = E/F/W/I/UP/B/SIM`, `ignore = E501`).
- Async-first; `pytest-asyncio` in `asyncio_mode = "auto"`; Pydantic v2 APIs only.
- Fork-based Git workflow: `origin` is personal fork, `upstream` is `memenow/persona-agent`.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **persona-agent**. Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/persona-agent/context` | Codebase overview, check index freshness |
| `gitnexus://repo/persona-agent/clusters` | All functional areas |
| `gitnexus://repo/persona-agent/processes` | All execution flows |
| `gitnexus://repo/persona-agent/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
