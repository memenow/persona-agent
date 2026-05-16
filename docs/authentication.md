# Authentication and CORS

The REST surface under `/api/v1` is gated by an API-key dependency that
can be toggled at runtime. The A2A protocol routes are public by design.

## API-key model

When `API_ENABLE_AUTH=true`:

1. Every REST request must carry a header named `API_KEY_HEADER`
   (default `X-API-Key`).
2. The header value is compared against each entry of
   `API_ALLOWED_KEYS` using `secrets.compare_digest`. The loop scans
   **every** allowed key without short-circuiting so a timing
   side-channel cannot reveal which prefix matched.
3. A missing header returns `401 Invalid or missing API key`.
4. A header that doesn't match any allowed key also returns `401`.
5. When auth is enabled but `API_ALLOWED_KEYS` is empty, the server
   returns `503 Authentication misconfigured` for every protected
   request and logs an error. This is intentional — silently allowing
   traffic in this state would be a defect.

When `API_ENABLE_AUTH=false` (the default), the dependency is a no-op
and every REST route is reachable.

## Key allow-list parsing

`API_ALLOWED_KEYS` is comma-separated. The parser strips whitespace
around each entry and drops blanks, so the following are equivalent and
all yield two valid keys:

```
API_ALLOWED_KEYS="abc,def"
API_ALLOWED_KEYS="abc, def, "
API_ALLOWED_KEYS=" abc ,, def"
```

A defense-in-depth check inside `verify_api_key` also skips empty
entries that somehow slip past the parser.

## What's protected

| Surface | Auth |
|---------|------|
| `GET/POST/PUT/DELETE /api/v1/personas[/...]` | required when enabled |
| `GET/POST/DELETE /api/v1/agents[/...]` | required when enabled |
| `GET/POST/DELETE /api/v1/sessions[/...]` | required when enabled |
| `GET /a2a/*`, `POST /a2a/{persona_id}/`, `/.well-known/agent.json` | always public |
| `GET /health` | always public |
| `GET /docs`, `GET /openapi.json` | always public (Swagger UI) |

A2A is intentionally public so external A2A agents can perform
discovery. Lock down the A2A surface at the network layer (private VPC,
ingress allow-list, mTLS, ...) when needed.

## CORS

CORS is enabled by default with `allowed_origins=["*"]`. When the
wildcard is present:

- `allow_credentials` is forced to `False`.
- A warning is logged so the operator can see the downgrade.

To allow credentialed cross-origin requests (cookies, `Authorization`
headers), set `API_ALLOWED_ORIGINS` to an explicit comma-separated list.

To disable CORS entirely (e.g. when fronted by a reverse proxy that
handles it), set `API_ENABLE_CORS=false`.

## Recommended deployment posture

| Environment | `API_ENABLE_AUTH` | `API_ALLOWED_ORIGINS` | Notes |
|-------------|-------------------|------------------------|-------|
| Local dev | `false` | `*` | Default behavior. |
| Internal staging | `true` | List of staging hostnames | Generate strong random keys. |
| Production | `true` | List of production hostnames | Rotate keys via your secret manager; never check them into git. |

`config.host` defaults to `127.0.0.1` so the server is not exposed
outside the machine until you explicitly set `API_HOST=0.0.0.0` (or
bind through a reverse proxy).

## Verifying

`GET /health` returns the effective posture:

```json
{
  "auth": { "enabled": true, "configured": true, "header": "X-API-Key" },
  "cors": { "enabled": true, "wildcard_origin": false, "credentials": true }
}
```

- `enabled` reflects `API_ENABLE_AUTH`.
- `configured` is `true` only when the allow-list is non-empty.
- `wildcard_origin` is `true` when `*` is present in
  `API_ALLOWED_ORIGINS`.
- `credentials` reflects the resolved `allow_credentials` flag.
