# Configuration

The server reads configuration from three places, in order of precedence
(highest first):

1. **Environment variables** — preferred for runtime knobs that differ
   per deployment.
2. **JSON config files** — `config/llm_config.json` for LLM credentials
   and tunables, `config/mcp_config.json` for MCP servers.
3. **Hardcoded defaults** in `src/persona_agent/api/config.py`.

**LLM credentials are an exception** to the env-wins rule above, and
`api_key` and `api_base` follow different cascades:

- **`api_key` is file-first**: the `model_configs[*]` entry whose
  `name`/`model` matches `default_model` is checked first, then the
  file-level `api_key`. `OPENAI_API_KEY` only kicks in when both file
  values are empty. To rotate the active key with an environment
  variable, leave the file's `api_key` empty for the active model.
- **`api_base` is env-first**: `OPENAI_BASE_URL` (or `OPENAI_API_BASE`)
  wins when set; otherwise the per-model `api_base` is used, then the
  file-level `api_base`.

## Environment variables

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `127.0.0.1` | Bind host. Set to `0.0.0.0` to expose externally. |
| `API_PORT` | `8000` | Bind port. |
| `API_DEBUG` | `false` | Enable debug-level logging in uvicorn. Truthy values: `true`, `1`, `yes`. |
| `API_PREFIX` | `/api/v1` | URL prefix for the REST surface. |
| `API_PUBLIC_BASE_URL` | (unset) | Externally reachable URL used in A2A discovery metadata. Set this behind a reverse proxy or load balancer. |

### Authentication and CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `API_ENABLE_AUTH` | `false` | When true, all REST routes require a valid API key. A2A routes remain public. |
| `API_KEY_HEADER` | `X-API-Key` | HTTP header that carries the API key. |
| `API_ALLOWED_KEYS` | (empty) | Comma-separated allow-list. Blank entries are filtered, so a trailing comma is harmless. Auth misconfiguration (enabled with empty allow-list) returns `503`. |
| `API_ENABLE_CORS` | `true` | Toggle CORS middleware. |
| `API_ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. When `*` is present, `allow_credentials` is forced off and a warning is logged. |

See [authentication.md](authentication.md) for the full auth model.

### Personas and configs

| Variable | Default | Description |
|----------|---------|-------------|
| `PERSONAS_DIR` | `<repo>/examples/personas` | Directory scanned at startup for persona files. Newly created personas are persisted here. |
| `LLM_CONFIG_PATH` | `<repo>/config/llm_config.json` | Path to the LLM JSON config. |
| `MCP_CONFIG_PATH` | `<repo>/config/mcp_config.json` | Path to the MCP JSON config. |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MODEL` | `gpt-4o` (file overrides on load) | Sets `ApiConfig.default_model`, which is reported by `/health` and used to pick the matching `model_configs[*]` entry inside `load_config()`. The LLM client itself is built from the file's `default_model` via `OpenAICompatibleClient.from_config()` and does **not** read this env var directly — to change the active model, edit `llm_config.json`. |
| `OPENAI_API_KEY` | (unset) | Fallback API key when `llm_config.json` is missing or has no key. |
| `OPENAI_API_BASE` | (unset) | Override base URL (used by `OpenAICompatibleClient`). |
| `OPENAI_BASE_URL` | (unset) | Alternate alias accepted by `from_config()`. |

## `llm_config.json`

```json
{
  "default_model": "gpt-4o-mini",
  "api_key": "",
  "api_base": "https://api.openai.com/v1",
  "model_configs": [
    {
      "name": "gpt-4o-mini",
      "api_key": "",
      "api_base": "https://api.openai.com/v1",
      "model": "gpt-4o-mini",
      "temperature": 0.7,
      "max_tokens": 4000,
      "top_p": 1.0,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    }
  ],
  "api_settings": {
    "timeout": 120,
    "retry_attempts": 3,
    "retry_delay": 2,
    "log_requests": true,
    "log_responses": false
  }
}
```

- `default_model` selects which `model_configs` entry is loaded at
  startup. Matching is done on `model` first, then `name`. If no entry
  matches, the first one is used as a fallback.
- A per-model `api_key`/`api_base` overrides the file-level
  `api_key`/`api_base` for that model.
- `api_settings` is currently **ignored** by the loader.
  `OpenAICompatibleClient.from_config()` reads only `default_model`,
  `api_key`, `api_base`, `temperature`, and `max_tokens`. The OpenAI SDK
  timeout stays at the `OpenAICompatibleClient` constructor default of
  120 seconds regardless of `api_settings.timeout`.

Any provider exposing an OpenAI-compatible Chat Completions API works:
OpenAI, Azure OpenAI, Ollama, vLLM, LiteLLM proxies, etc.

## `mcp_config.json`

```json
{
  "mcpServers": {
    "mcp-server-fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"],
      "disabled": false,
      "description": "Web content fetching and conversion for efficient LLM usage"
    },
    "brave_search": {
      "command": "node",
      "args": ["path/to/mcp-brave-search/index.js"],
      "env": { "BRAVE_API_KEY": "${BRAVE_API_KEY}" },
      "description": "Brave Search MCP service"
    }
  }
}
```

- Both `"mcpServers"` (current) and `"services"` (legacy) section names
  are accepted. When both exist they are merged; `services` entries must
  set `type: "stdio"` and `enabled: true` (defaults).
- Each entry must define `command`; `args`, `env`, `description`, and
  `disabled` are optional.
- `${VAR_NAME}` references in `command`, `args`, and `env` values are
  substituted from the process environment at load time. Missing
  variables expand to an empty string — set defaults explicitly when that
  matters.

See [mcp-integration.md](mcp-integration.md) for runtime behavior.

## Verifying effective config

The startup log prints a one-line summary of the resolved config:

```
Effective config: host=127.0.0.1 port=8000 auth=False api_keys=0 cors=True wildcard_origin=True
```

The `/health` endpoint surfaces the same information as JSON without
requiring auth, so operators can verify a deployment without holding a
key.
