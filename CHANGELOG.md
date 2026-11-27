# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **Package structure**: Reorganized `src/persona_agent/` by domain responsibility (`config.py`, `personas/`, `llm/`, `mcp/`, `a2a/`, `agents/`, `api/`), added top-level `__version__` via `importlib.metadata`, and removed unused schema models and duplicate `ruff.toml`.
- **A2A v1.0 upgrade**: Upgraded `a2a-sdk` to `1.3.1`, migrated `AgentCard` and executor events to Protobuf v1.0 (`supportedInterfaces`, `Task` + `TaskUpdater`), replaced removed `A2AFastAPIApplication` with `add_a2a_routes_to_fastapi`, enabled v0.3 wire compatibility (`enable_v0_3_compat=True`), and exposed both `/.well-known/agent-card.json` and `/.well-known/agent.json`.
- **MCP v2 upgrade**: Upgraded `mcp` to `2.3.0`, renamed `DirectMCPManager` to `MCPManager` in `persona_agent.mcp.manager`, migrated connection lifecycle to `mcp.Client`, updated tool schema conversion to `input_schema`, and added `MCPError` / `is_error` handling.
- **OpenAI SDK v3 & dependency refresh**: Upgraded `openai` to `3.60.0` and updated all runtime and development dependencies (`fastapi`, `uvicorn`, `pydantic`, `ruff`, `pytest`, `pytest-asyncio`, `httpx2`); removed unused direct dependencies (`python-dotenv`, `typing-extensions`).
- **Application wiring**: Converted `create_app()` to a synchronous factory, moved `A2ARegistry` and service dependencies onto `app.state` (removing module-level globals and `@lru_cache` singletons), and consolidated `llm_config.json` loading so `DEFAULT_MODEL` and `api_settings.timeout` take effect consistently.
- **Tests and docs**: Reorganized `tests/` by module, expanded A2A v1.0/v0.3 and error-boundary coverage, and consolidated `docs/` into six reference guides.

### Fixed

- Preserved `HTTPException` propagation in `POST /api/v1/personas/upload` and `POST /api/v1/agents` instead of catching and rewriting them as generic `500` responses; invalid uploaded persona payloads now return `422`.
- Prevented in-memory state corruption when persona validation or file persistence fails during `PUT /api/v1/personas/{id}` or `POST /api/v1/personas`.
- Stripped whitespace and dropped empty items when parsing `API_ALLOWED_ORIGINS`.
- Deferred `logging.basicConfig()` from module import time into `run_server()`.

## [0.2.0]

### Changed

- Standardized session, agent, and message timestamps on Unix wall-clock seconds (`time.time()`).
- Hardened global exception handling so raw exception messages are logged server-side and never echoed to HTTP clients.
- Switched API key comparison to a constant-time full allow-list scan (`secrets.compare_digest`) and filtered blank entries from `API_ALLOWED_KEYS`.
- Enforced `^[a-z0-9_-]{1,64}$` persona ID validation with defense-in-depth filesystem path checks and safe tracked-file deletion.
- Added per-context `asyncio.Lock` serialization, bounded LRU conversation history (200 contexts), and `tool_choice="none"` fallback when `MAX_TOOL_ITERATIONS` is reached.
- Added opt-in API key auth (`API_ENABLE_AUTH`), wildcard-safe CORS defaults, `127.0.0.1` default bind host, and `API_PUBLIC_BASE_URL` support.
- Migrated agent execution from AutoGen to `a2a-sdk`, `openai`, and `mcp`.
