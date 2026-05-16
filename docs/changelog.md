# Changelog

Notable changes since the AutoGen → A2A migration. Each entry is keyed
by the merging pull request; full diffs live in `git log` and on
GitHub.

## Unreleased

- Documentation overhaul: split detailed reference material into
  `docs/`, tightened `README.md`, and aligned both with the current
  code surface.

## 0.2.0 (PR #9 — `17112b6`)

P0 review follow-ups closing the last batch of API hardening items.

- **Timestamps**: `AgentSession.created_at` / `last_active`, each
  `agents[*].created_at`, and every executor history entry are now Unix
  wall-clock seconds via `time.time()`. Clients can correlate session
  metadata with individual message timestamps.
- **Error responses**: every unhandled exception is logged with
  `logger.exception(...)` and surfaces a generic
  `{ "detail": "Internal server error" }` to clients. Exception text is
  no longer echoed in HTTP responses.
- **API-key comparison**: `verify_api_key` iterates the full allow-list
  without short-circuiting, using `secrets.compare_digest` on every
  entry. Empty headers are rejected outright.
- **`API_ALLOWED_KEYS` parsing**: blank entries from trailing commas or
  double commas are filtered, so a stray separator can no longer admit
  an empty key.
- **Repo metadata**: added `.gitnexus/` to `.gitignore`; appended
  GitNexus guidance to `CLAUDE.md`.

## PR #8 — `96baf48`

Runtime config and persona persistence hardening.

- **LLM credential cascade**: `load_config` reads `default_model` first,
  then loads `api_key`/`api_base` from the matching `model_configs`
  entry, with environment overrides winning. A global file-level
  `api_key` no longer overrides a per-model key.
- **`save_persona` re-validation**: `Persona.save_persona` re-checks
  the ID regex and rejects path-traversal patterns (`..`, `/`, `\`,
  null bytes) before touching the filesystem, even though the model's
  Pydantic validator already constrains the same field.
- **Tests**: added `tests/test_review_fixes.py` covering the new
  config and persistence paths.

## PR #7 — `3fa9e14`

A2A executor await fixes, path-traversal hardening, and broader
security tightening.

- **Awaitable event queue**: every
  `event_queue.enqueue_event(...)` call is awaited. Status updates that
  previously dropped silently now reach clients.
- **Shared `chat()` entry point**: the legacy `_run_llm_loop` was
  promoted to a public `chat(context_id, user_text)` method. REST
  `send_message` and the A2A `execute` path both call it, so behavior
  cannot diverge.
- **Tool-iteration cap fallback**: when `MAX_TOOL_ITERATIONS = 10` is
  exceeded, the executor forces a final completion with
  `tool_choice="none"`. Earlier versions stripped prior tool messages
  in a way that broke strict OpenAI-compatible providers.
- **Per-context concurrency**: each `context_id` is now serialized by
  an `asyncio.Lock`. Concurrent A2A or REST calls on the same
  conversation cannot interleave history writes. Histories are stored
  in an LRU `OrderedDict` capped at 200 contexts; the corresponding
  lock is evicted with the history.
- **Persona ID safety**: `Persona.id` is constrained by
  `^[a-z0-9_-]{1,64}$` at the Pydantic layer and re-checked on save and
  delete. The loader sanitizes legacy IDs in place (with a warning) so
  pre-existing files keep loading.
- **Networking defaults**: server binds to `127.0.0.1` by default; CORS
  `allow_credentials` is forced off when `allowed_origins` contains
  `*` (with a warning).
- **Auth dependency**: introduced `make_api_key_dependency(config)`
  producing a FastAPI dependency that enforces the configured header.
- **Pydantic v2**: replaced every `.dict()` with `.model_dump()`.
- **Tests**: added `test_auth.py`, `test_executor.py`,
  `test_persona_id_security.py`.

## PR #6 — `d580a6e`

Documentation pass after the A2A migration.

- Removed stale AutoGen references from `README.md` and docstrings.

## PR #5 — `bba117b`

Migrated from AutoGen 0.4 to the Google A2A protocol.

- **Removed**: `src/persona_agent/core/persona_agent.py`,
  `core/agent_factory.py`, and the AutoGen dependency.
- **Added**: `a2a/executor.py`, `a2a/agent_card.py`, `llm/client.py`,
  `mcp/direct_mcp.py`. Each persona is now an independent A2A ASGI
  sub-app built via the SDK's public `A2AFastAPIApplication.build()`
  API.
- **Shared infrastructure**: REST routes and the A2A executor share a
  single `LLMClient` and `DirectMCPManager` via `AgentFactory`.
- **Net change**: ~-2000 lines.
