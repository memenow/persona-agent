# REST API, Authentication, and CORS

REST routes are mounted under `/api/v1` (configurable via `API_PREFIX`). Interactive Swagger UI is served at `/docs` and the OpenAPI schema at `/openapi.json`.

## Conventions

- **Content type**: `application/json` (`POST /api/v1/personas/upload` uses `multipart/form-data`).
- **Timestamps**: Unix wall-clock seconds (`float` from `time.time()`).
- **Identifiers**: Persona IDs match `^[a-z0-9_-]{1,64}$`. Agent and session IDs are UUIDs.
- **Errors**: Standard HTTP errors return `{ "detail": "<message>" }` without raw exception text. Request validation errors (`422`) follow FastAPI's structured list format under `detail`.

## Health

### `GET /health`

Always public. Reports version, default model, loaded MCP tool count, and active auth/CORS settings.

```json
{
  "status": "ok",
  "api_version": "0.2.0",
  "protocol": "a2a",
  "default_model": "gpt-4o-mini",
  "mcp_tools": 1,
  "auth": { "enabled": false, "configured": false, "header": "X-API-Key" },
  "cors": { "enabled": true, "wildcard_origin": true, "credentials": false }
}
```

## Personas

### `GET /api/v1/personas`

List all loaded personas.

```json
{
  "personas": [
    { "id": "trump", "name": "Donald Trump", "description": "..." }
  ]
}
```

### `GET /api/v1/personas/{persona_id}`

Return full persona details (`personal_background`, `language_style`, `knowledge_domains`, `interaction_samples`). Returns `404` if missing.

### `POST /api/v1/personas`

Create a persona and save it as JSON under `PERSONAS_DIR`. Returns `201`.

```json
{
  "name": "Ada Lovelace",
  "description": "Mathematician and writer.",
  "personal_background": { "profession": "Mathematician" },
  "language_style": { "tone": "Precise" },
  "knowledge_domains": { "computing": ["analytical engine"] },
  "interaction_samples": [],
  "system_prompt": null
}
```

- `id` is generated as a UUID when omitted.
- Returns `500` if writing to disk fails (in-memory state is rolled back).

### `PUT /api/v1/personas/{persona_id}`

Partial update preserving omitted fields. Validates updates against a copy before replacing stored state and rolls back if persistence fails. Returns `404` if missing or `500` on save failure.

### `DELETE /api/v1/personas/{persona_id}`

Delete the persona from memory and unlink its tracked file(s) inside `PERSONAS_DIR`. Files resolving outside `PERSONAS_DIR` are skipped. Returns `404` if missing or `500` if file removal fails (in-memory state is preserved on failure).

### `POST /api/v1/personas/upload`

Upload a `.json`, `.yaml`, or `.yml` file via `multipart/form-data`. Returns `201` with the created persona.

- `400`: missing filename, unsupported extension, non-UTF-8 content, or invalid JSON/YAML syntax.
- `422`: parsed payload is not an object or fails `Persona` schema validation (including invalid `id`).
- `500`: disk write failure.

## Agents

An agent binds a persona to a `PersonaAgentExecutor` instance.

- `GET /api/v1/agents` — list active agents.
- `GET /api/v1/agents/{agent_id}` — fetch one agent (`404` if missing).
- `POST /api/v1/agents` — create an agent (`201`). Body: `{ "persona_id": "trump", "model": null }`. The `model` field is reserved for future per-agent overrides. Returns `404` if the persona does not exist.
- `DELETE /api/v1/agents/{agent_id}` — delete an agent, delete its bound sessions, and clear their conversation history (`404` if missing).

## Sessions

A session tracks a single conversation context on an agent's executor.

- `GET /api/v1/sessions` — list active sessions with `message_count`, `created_at`, and `last_active`.
- `GET /api/v1/sessions/{session_id}` — fetch session metadata (`404` if missing).
- `POST /api/v1/sessions` — create a session (`201`). Body: `{ "agent_id": "<uuid>" }`. Returns `404` if the agent does not exist.
- `DELETE /api/v1/sessions/{session_id}` — delete the session and clear its executor history (`404` if missing).
- `GET /api/v1/sessions/{session_id}/messages` — return recorded messages (`role`, `content`, `timestamp`).
- `POST /api/v1/sessions/{session_id}/messages` — send `{ "message": "..." }` and return `{ "session_id", "success", "response", "error_details" }`.

Notes on `POST /messages`:

- The user message is appended before the LLM call, so failed turns still appear in message history.
- `last_active` updates only on a successful reply.
- Internal errors return `success: false` with a generic message while logging tracebacks server-side.

## Status codes

| Code | Meaning |
|------|---------|
| `200` | Request succeeded |
| `201` | Resource created |
| `400` | Malformed upload filename, encoding, or JSON/YAML syntax |
| `401` | Auth enabled and API key header is missing or invalid |
| `404` | Resource not found |
| `422` | Payload failed Pydantic validation (`detail` is a list of error objects) |
| `500` | Server or persistence failure |
| `503` | Auth enabled with an empty `API_ALLOWED_KEYS` allow-list |

## Authentication

When `API_ENABLE_AUTH=true`:

1. Protected REST routes require the header named by `API_KEY_HEADER` (default `X-API-Key`).
2. `API_ALLOWED_KEYS` is parsed as a comma-separated list with whitespace stripped and empty entries dropped.
3. The dependency compares the incoming key against every non-empty allowed key using `secrets.compare_digest` without short-circuiting.
4. If `API_ENABLE_AUTH=true` and `API_ALLOWED_KEYS` is empty, protected requests return `503 Authentication misconfigured`.
5. Changing `API_KEY_HEADER` requires restarting the server because the header scheme is bound during `create_app()`.

Route protection matrix:

| Surface | Auth |
|---------|------|
| `/api/v1/personas*`, `/api/v1/agents*`, `/api/v1/sessions*` | Enforced when `API_ENABLE_AUTH=true` |
| `/.well-known/agent-card.json`, `/.well-known/agent.json`, `/a2a/*` | Public (per A2A discovery design) |
| `/health`, `/docs`, `/openapi.json` | Public |

Restrict public A2A routes at the network or ingress layer (private VPC, mTLS, reverse proxy ACL) when exposure must be limited.

## CORS

CORS is enabled by default (`API_ENABLE_CORS=true`) with `API_ALLOWED_ORIGINS=*`.

- When `*` appears in `API_ALLOWED_ORIGINS`, `allow_credentials` is forced to `False` and a startup warning is logged.
- To allow credentialed cross-origin requests, set `API_ALLOWED_ORIGINS` to an explicit comma-separated list of origins.
- Set `API_ENABLE_CORS=false` when an upstream reverse proxy owns CORS headers.
