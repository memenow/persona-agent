# REST API Reference

The REST API is served under the prefix `/api/v1` (configurable via
`API_PREFIX`). All REST routes are gated by the API-key dependency when
authentication is enabled — see [authentication.md](authentication.md).

Interactive Swagger UI is available at `/docs` and the OpenAPI schema at
`/openapi.json`.

## Conventions

- **Content type**: `application/json` unless noted (file upload uses
  `multipart/form-data`).
- **Timestamps**: Unix wall-clock seconds (float) via `time.time()`.
- **IDs**: persona IDs match `^[a-z0-9_-]{1,64}$`. Agent and session IDs
  are UUIDs.
- **Error shape**: `{ "detail": "<message>" }` for HTTP errors; messages
  are generic and stable (no exception text leaks).

## Health

### `GET /health`

Public. Returns effective runtime posture so operators can verify config
without authenticating.

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
    { "id": "trump", "name": "Donald Trump", "description": "45th President of the United States…" }
  ]
}
```

### `GET /api/v1/personas/{persona_id}`

Get full persona details (background, language style, knowledge domains,
interaction samples).

- `404` if not found.

### `POST /api/v1/personas`

Create a new persona. Returns `201`.

**Request:**

```json
{
  "name": "Albert Einstein",
  "description": "Theoretical physicist…",
  "personal_background": { "birth": "1879-03-14", "profession": "Physicist" },
  "language_style": { "tone": "Thoughtful" },
  "knowledge_domains": { "physics": ["relativity"] },
  "interaction_samples": [
    { "type": "quote", "content": "Imagination is more important than knowledge." }
  ],
  "system_prompt": null
}
```

- `id` is auto-generated as a UUID when omitted.
- The persona is persisted as a JSON file under `PERSONAS_DIR`.
- `500` if the file cannot be written.

### `PUT /api/v1/personas/{persona_id}`

Partial update; any field omitted from the body is preserved. Returns the
updated persona.

- `404` if not found.

### `DELETE /api/v1/personas/{persona_id}`

Delete the persona and any persona files inside `PERSONAS_DIR`. Files
outside the directory (e.g. via symlink) are skipped defensively.

- `404` if not found. `500` if persisted files cannot be removed.

### `POST /api/v1/personas/upload`

Upload a `.json`, `.yaml`, or `.yml` file via `multipart/form-data`.
Returns `201` with the created persona.

- `400` for missing filename, wrong extension, or invalid JSON/YAML.
- `500` if save fails.

## Agents

An agent binds a persona to an executor that manages chat with the LLM and
MCP tools.

### `GET /api/v1/agents`

List all active agents.

```json
{
  "agents": [
    { "id": "<uuid>", "persona_id": "trump", "name": "Donald Trump", "created_at": 1715846400.0 }
  ]
}
```

### `GET /api/v1/agents/{agent_id}`

Get a single agent. `404` if not found.

### `POST /api/v1/agents`

Create an agent for a persona. Returns `201`.

**Request:**

```json
{ "persona_id": "trump", "model": null }
```

- `model` is reserved for future multi-model support; current builds use
  the LLM configured in `llm_config.json`.
- `404` if the persona does not exist; `500` if agent creation fails.

### `DELETE /api/v1/agents/{agent_id}`

Delete an agent and any sessions bound to it. `404` if not found.

## Sessions

A session is a conversation with a single agent; messages are buffered in
the executor's per-context history.

### `GET /api/v1/sessions`

List all active sessions with metadata.

```json
{
  "sessions": [
    {
      "id": "<uuid>",
      "agent_id": "<uuid>",
      "persona_id": "trump",
      "message_count": 4,
      "created_at": 1715846400.0,
      "last_active": 1715846412.7
    }
  ]
}
```

### `GET /api/v1/sessions/{session_id}`

Get session metadata. `404` if not found.

### `POST /api/v1/sessions`

Create a new session bound to an existing agent. Returns `201`.

**Request:**

```json
{ "agent_id": "<uuid>" }
```

- `404` if the agent does not exist; `500` on internal failure.

### `DELETE /api/v1/sessions/{session_id}`

Delete the session and clear its executor history.

### `GET /api/v1/sessions/{session_id}/messages`

Return the full message history for the session.

```json
{
  "session_id": "<uuid>",
  "messages": [
    { "role": "user", "content": "Hi", "timestamp": 1715846400.0 },
    { "role": "assistant", "content": "Hello!", "timestamp": 1715846401.4 }
  ]
}
```

Older history entries written before timestamp tracking surface
`timestamp: 0.0`.

### `POST /api/v1/sessions/{session_id}/messages`

Send a message and wait for the agent's response.

**Request:**

```json
{ "message": "What's your view on trade policy?" }
```

**Response:**

```json
{
  "session_id": "<uuid>",
  "success": true,
  "response": "<assistant reply>",
  "error_details": null
}
```

- The user message is appended to executor history **before** the LLM
  call, so it remains in history even when the call fails;
  `GET /sessions/{id}/messages` will show the failed turn.
- `last_active` is updated only on success.
- On internal errors the response contains `success: false` and a generic
  message; details are logged server-side, never returned to the caller.
  Clients retrying a failed `POST /messages` should expect the previous
  user message to already be in the conversation.

## A2A discovery

The A2A protocol surfaces are documented in
[a2a-protocol.md](a2a-protocol.md). They are public (no API-key gate) by
protocol design.

## Status codes

| Code | Meaning |
|------|---------|
| `200` | Success. |
| `201` | Resource created (`POST /personas`, `/agents`, `/sessions`, `/personas/upload`). |
| `400` | Malformed input (bad upload filename or content). |
| `401` | Auth enabled and header missing or invalid. |
| `404` | Resource not found. |
| `422` | Request body fails Pydantic / FastAPI validation (e.g. wrong type, missing required field). |
| `500` | Internal failure (file write, agent creation, persona save). |
| `503` | Auth enabled but the allow-list is empty (misconfiguration). |

Most non-2xx responses use the `{ "detail": "<message>" }` shape and
never contain exception text or stack traces. The exception is `422`,
where FastAPI populates `detail` with a **list of per-field validation
error objects**:

```json
{
  "detail": [
    {
      "loc": ["body", "agent_id"],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

Clients should handle `422` separately from the string-`detail` shape.
See [FastAPI's validation error format](https://fastapi.tiangolo.com/tutorial/handling-errors/#requestvalidationerror)
for the full schema.
