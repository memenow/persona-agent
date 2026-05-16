# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

AI persona agents built on Google A2A protocol with FastAPI REST API and MCP tool integration. Personas are defined as YAML/JSON files loaded from `examples/personas/`.

## Commands

```bash
# Install dependencies
uv sync

# Lint and format
uv run ruff check .
uv run ruff format .

# CLI commands
uv run persona-agent api              # Start API server
uv run persona-agent list-personas    # List available personas
uv run persona-agent agent-card       # Show A2A agent cards
uv run persona-agent import-persona FILE  # Import persona file
```

## Architecture

Four-layer structure under `src/persona_agent/`:

- **api/** — FastAPI routes, models, dependency injection, API-key auth. REST routes prefixed with `/api/v1`, A2A routes at `/a2a/`.
- **a2a/** — A2A protocol integration: `PersonaAgentExecutor` (implements `AgentExecutor`), `AgentCard` builder. Each persona is an A2A agent.
- **llm/** — LLM client abstraction with `OpenAICompatibleClient` using the `openai` SDK directly. Supports any OpenAI-compatible provider.
- **mcp/** — `DirectMCPManager` using the `mcp` library directly for stdio server lifecycle, tool loading, and execution.

## Key Files

- `api/server.py` — FastAPI app factory and lifespan; wires MCP, persona manager, A2A registry, auth dependency, CORS, and exception handlers.
- `api/agent_factory.py` — `AgentFactory` + `AgentSession`; creates `PersonaAgentExecutor` instances and tracks sessions with wall-clock timestamps.
- `api/persona_manager.py` — `Persona` Pydantic model (id pattern `^[a-z0-9_-]{1,64}$`) and `PersonaManager` filesystem I/O with path-traversal defense.
- `api/auth.py` — `make_api_key_dependency`; `secrets.compare_digest` in a full allow-list loop. Returns 401 on miss, 503 if auth is enabled with no allowed keys.
- `api/routes/a2a.py` — `A2ARegistry`; each persona is mounted as an independent ASGI sub-app via `A2AFastAPIApplication.build()`.
- `a2a/executor.py` — Core agent logic: shared `chat()` entrypoint, per-context `asyncio.Lock`, LRU history (200 contexts), `MAX_TOOL_ITERATIONS=10`.
- `llm/client.py` — Abstract `LLMClient` + `OpenAICompatibleClient`; `from_config()` reads `default_model` + `model_configs`.
- `mcp/direct_mcp.py` — Direct MCP stdio management with per-server retries (`MAX_RETRIES=3`) and `${VAR_NAME}` env substitution.

## Configuration

Precedence: environment variables > JSON config files > hardcoded defaults in `api/config.py`.

Server env: `API_HOST` (default `127.0.0.1`), `API_PORT`, `API_DEBUG`, `API_PREFIX`, `API_PUBLIC_BASE_URL`.
Auth/CORS env: `API_ENABLE_AUTH`, `API_KEY_HEADER`, `API_ALLOWED_KEYS` (CSV, blanks filtered), `API_ENABLE_CORS`, `API_ALLOWED_ORIGINS`.
File env: `PERSONAS_DIR` (default `examples/personas`), `LLM_CONFIG_PATH` (`config/llm_config.json`), `MCP_CONFIG_PATH` (`config/mcp_config.json`).
LLM env: `OPENAI_API_KEY`, `OPENAI_API_BASE` / `OPENAI_BASE_URL`, `DEFAULT_MODEL`.

## A2A Endpoints

- `GET /.well-known/agent.json` — Aggregate agent card for all personas (public).
- `GET /a2a/personas` — List all A2A persona agents (public).
- `GET /a2a/{persona_id}/.well-known/agent-card.json` — Individual persona agent card (SDK sub-app).
- `POST /a2a/{persona_id}/` — A2A JSON-RPC endpoint (SDK sub-app).

A2A surfaces are public by protocol design. REST routes under `/api/v1` are gated by `verify_api_key` when `API_ENABLE_AUTH=true`.

## Persona Schema

Persona YAML/JSON files require only `name`. Optional fields: `id` (auto-generated if absent), `description`, `personal_background`, `language_style`, `knowledge_domains`, `interaction_samples`, `system_prompt`. The `Persona` Pydantic model (in `api/persona_manager.py`) constrains `id` to `^[a-z0-9_-]{1,64}$` and re-validates on save against path-traversal patterns.

## MCP Config

`config/mcp_config.json` supports both `"mcpServers"` (current) and `"services"` (legacy) section names. `${VAR_NAME}` in `command`, `args`, and `env` values is substituted from the process environment at load; missing variables expand to an empty string.

## Documentation

- `README.md` — quick start and orientation.
- `docs/` — in-depth references (API, A2A, persona schema, configuration, MCP, auth, deployment, development, changelog). Update the matching `docs/` file when changing behavior it documents.

## Git Workflow

Fork-based: `origin` is your fork, `upstream` is `memenow/persona-agent`. PR to upstream.

## Code Style

- Python with type hints throughout.
- PEP 8 via `ruff` (target-version py311, line-length 88, select E/F/W/I/UP/B/SIM, ignore E501/B008).
- Async-first; `pytest-asyncio` in `asyncio_mode="auto"`.
- Pydantic v2 only (`.model_dump()`, not `.dict()`).

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **persona-agent** (762 symbols, 1141 relationships, 12 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
