# Persona Agent

A Python service that exposes AI personas as
[Google A2A protocol](https://github.com/google/A2A) agents. Each persona
is also reachable through a REST API for management and direct chat.

Built directly on `a2a-sdk`, the `openai` SDK, and the `mcp` library —
no framework wrappers in between.

## Features

- **A2A-native personas**: every persona is mounted as an independent
  A2A ASGI sub-app with discoverable agent cards and a JSON-RPC endpoint.
- **REST API**: CRUD for personas, agents, and sessions, with file
  upload support.
- **MCP tool integration**: stdio MCP servers are loaded at startup; the
  LLM can call their tools mid-conversation.
- **OpenAI-compatible LLMs**: works with OpenAI, Azure OpenAI, Ollama,
  vLLM, LiteLLM, or any other OpenAI-API-compatible provider.
- **YAML / JSON personas**: declarative character files with background,
  language style, knowledge domains, and interaction samples.
- **Operations-friendly**: opt-in API key auth, configurable CORS,
  health endpoint exposing effective config.

## Quick start

```bash
git clone https://github.com/memenow/persona-agent.git
cd persona-agent
uv sync
uv run persona-agent api
```

The server binds to `http://127.0.0.1:8000` by default. Swagger UI is at
`/docs`, the aggregate A2A agent card at `/.well-known/agent.json`, the
REST surface under `/api/v1`.

Set your LLM credentials in `config/llm_config.json` (see
[docs/configuration.md](docs/configuration.md)):

```json
{
  "default_model": "gpt-4o-mini",
  "model_configs": [
    {
      "name": "gpt-4o-mini",
      "model": "gpt-4o-mini",
      "api_key": "sk-...",
      "api_base": "https://api.openai.com/v1",
      "temperature": 0.7,
      "max_tokens": 4000
    }
  ]
}
```

For a quick start, leave the `api_key` fields empty in the file and
export `OPENAI_API_KEY`. The LLM client falls back to that environment
variable only when both the per-model and file-level `api_key` are
empty — non-empty file values take precedence.

## CLI

```bash
uv run persona-agent api                    # start the API server
uv run persona-agent list-personas          # list personas from PERSONAS_DIR
uv run persona-agent agent-card             # dump A2A agent cards
uv run persona-agent agent-card trump       # dump a single persona's card
uv run persona-agent import-persona FILE    # import a JSON/YAML persona file
```

## Endpoints (overview)

| Surface | Path | Notes |
|---------|------|-------|
| REST | `/api/v1/personas`, `/agents`, `/sessions` | CRUD; details in [docs/api-reference.md](docs/api-reference.md). |
| REST | `POST /api/v1/personas/upload` | Upload a YAML/JSON persona file. |
| REST | `GET /api/v1/sessions/{id}/messages`, `POST .../messages` | Read history; send a message and get a reply. |
| A2A | `GET /.well-known/agent.json` | Aggregate card for every persona. |
| A2A | `GET /a2a/personas` | List registered A2A agents. |
| A2A | `GET /a2a/{persona_id}/.well-known/agent-card.json` | Per-persona agent card. |
| A2A | `POST /a2a/{persona_id}/` | A2A JSON-RPC endpoint. |
| Ops | `GET /health` | Public health/effective-config probe. |
| Ops | `GET /docs`, `/openapi.json` | Swagger UI / OpenAPI schema. |

REST routes are protected by an opt-in API-key dependency (off by
default). A2A surfaces are public by protocol design. See
[docs/authentication.md](docs/authentication.md).

## Persona example

```yaml
name: Donald Trump
description: 45th President of the United States…
personal_background:
  birth: June 14, 1946, Queens, New York City, USA
language_style:
  tone: Assertive, direct, hyperbolic
  common_phrases: ["Believe me", "Tremendous"]
knowledge_domains:
  politics: [American politics, Immigration policy, Trade policy]
interaction_samples:
  - type: quote
    content: "We will make America strong again."
```

The persona schema, including ID rules and the system-prompt generator,
is documented in [docs/persona-schema.md](docs/persona-schema.md).

## Project layout

```
persona-agent/
├── config/                          LLM and MCP JSON configs
├── docs/                            In-depth documentation
├── examples/personas/               Example persona files
├── src/persona_agent/
│   ├── a2a/        agent_card.py, executor.py
│   ├── api/        routes/, agent_factory.py, auth.py, config.py,
│   │               dependencies.py, models.py, persona_manager.py, server.py
│   ├── llm/        client.py (LLMClient + OpenAICompatibleClient)
│   ├── mcp/        direct_mcp.py (DirectMCPManager)
│   └── cli.py
├── tests/                           pytest suite
└── pyproject.toml
```

## Documentation

| File | What it covers |
|------|----------------|
| [docs/api-reference.md](docs/api-reference.md) | REST endpoints with payloads and status codes. |
| [docs/a2a-protocol.md](docs/a2a-protocol.md) | A2A integration, agent cards, mount strategy, executor behavior. |
| [docs/persona-schema.md](docs/persona-schema.md) | Persona fields, ID rules, examples, prompt generator. |
| [docs/configuration.md](docs/configuration.md) | Environment variables, `llm_config.json`, `mcp_config.json`. |
| [docs/mcp-integration.md](docs/mcp-integration.md) | MCP stdio lifecycle, env-var substitution, tool execution. |
| [docs/authentication.md](docs/authentication.md) | API keys, CORS, public vs protected routes. |
| [docs/deployment.md](docs/deployment.md) | Bind, reverse proxy, logging, containerization. |
| [docs/development.md](docs/development.md) | Setup, lint, format, test, project layout. |
| [docs/changelog.md](docs/changelog.md) | Notable changes by pull request. |

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format .
uv run pytest
```

Python 3.11+ required. Full development notes in
[docs/development.md](docs/development.md).

## License

See [LICENSE](LICENSE).
