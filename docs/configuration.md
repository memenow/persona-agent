# Configuration and Deployment

Runtime configuration is loaded by `persona_agent.config.load_settings()` from environment variables, JSON config files (`config/llm_config.json`, `config/mcp_config.json`), and defaults in `Settings`.

## Precedence

- **General settings**: Environment variables > JSON config files > `Settings` defaults.
- **LLM `api_key`**: Matched `model_configs[*].api_key` > top-level `llm_config.json` `api_key` > `OPENAI_API_KEY`. Leave file keys empty when injecting credentials via environment variables.
- **LLM `api_base`**: `OPENAI_BASE_URL` / `OPENAI_API_BASE` > matched `model_configs[*].api_base` > top-level `llm_config.json` `api_base`.
- **LLM `default_model`**: `DEFAULT_MODEL` env var > `llm_config.json` `default_model` > `"gpt-4o-mini"`.

## Environment variables

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `127.0.0.1` | Bind address. Set to `0.0.0.0` to accept external traffic. |
| `API_PORT` | `8000` | Bind port. |
| `API_DEBUG` | `false` | Sets uvicorn log level to `debug` (`true`, `1`, `yes`). |
| `API_PREFIX` | `/api/v1` | URL prefix for REST routes. |
| `API_PUBLIC_BASE_URL` | *(unset)* | Public URL used in A2A AgentCard `supportedInterfaces`. Required behind a reverse proxy or load balancer. |

### Auth and CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `API_ENABLE_AUTH` | `false` | Enables API key enforcement on `/api/v1/*` routes. |
| `API_KEY_HEADER` | `X-API-Key` | Header read for API keys. |
| `API_ALLOWED_KEYS` | *(empty)* | Comma-separated API keys (trimmed, empty items ignored). |
| `API_ENABLE_CORS` | `true` | Enables FastAPI `CORSMiddleware`. |
| `API_ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins (trimmed, empty items ignored). Wildcard forces `allow_credentials=False`. |

### Files and LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `PERSONAS_DIR` | `<repo>/examples/personas` | Directory scanned for persona files and used for saves. |
| `LLM_CONFIG_PATH` | `<repo>/config/llm_config.json` | Path to LLM JSON configuration. |
| `MCP_CONFIG_PATH` | `<repo>/config/mcp_config.json` | Path to MCP JSON configuration. |
| `DEFAULT_MODEL` | `gpt-4o-mini` | Overrides `default_model` in `llm_config.json` for both `/health` and `OpenAICompatibleClient`. |
| `OPENAI_API_KEY` | *(unset)* | Fallback API key when file-level and per-model keys are empty. |
| `OPENAI_BASE_URL` / `OPENAI_API_BASE` | *(unset)* | Overrides `api_base` in `llm_config.json`. |

## `config/llm_config.json`

```json
{
  "default_model": "gpt-4o-mini",
  "api_key": "",
  "api_base": "https://api.openai.com/v1",
  "model_configs": [
    {
      "name": "gpt-4o-mini",
      "model": "gpt-4o-mini",
      "api_key": "",
      "api_base": "https://api.openai.com/v1",
      "temperature": 0.7,
      "max_tokens": 4000
    }
  ],
  "api_settings": {
    "timeout": 120
  }
}
```

`OpenAICompatibleClient.from_config()` reads `default_model`, `api_key`, `api_base`, `model_configs[*]` (`name`, `model`, `api_key`, `api_base`, `temperature`, `max_tokens`), and `api_settings.timeout`.

## Deployment

### Process model

- Shared singletons (`Settings`, `OpenAICompatibleClient`, `MCPManager`, `PersonaRepository`, `AgentService`, `A2ARegistry`) live on `app.state`.
- Agents, sessions, A2A tasks, and conversation histories are in-memory and process-local. Run a single uvicorn worker per replica; multi-worker deployment requires an external store.

### Exposing behind a reverse proxy

```bash
API_HOST=0.0.0.0 \
API_PORT=8000 \
API_PUBLIC_BASE_URL=https://persona-agent.example.com \
API_ENABLE_AUTH=true \
API_ALLOWED_KEYS="replace-with-random-key" \
API_ALLOWED_ORIGINS="https://app.example.com" \
uv run persona-agent api
```

### Container image example

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

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

Note: `openai>=3.60.0` uses the OS trust store (`truststore`) for TLS verification. Ensure standard CA certificates (`ca-certificates`) are installed in minimal base images.

### Observability

- Startup logs print one summary line: `Effective config: host=... port=... auth=... api_keys=... cors=... wildcard_origin=...`.
- Use `GET /health` as a liveness/readiness probe and to inspect active auth, CORS, model, and MCP tool counts without credentials.
