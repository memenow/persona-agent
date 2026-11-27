# Persona Agent

Python service exposing character personas over the [A2A v1.0 protocol](https://github.com/a2aproject/A2A) and a FastAPI REST API, with OpenAI-compatible LLM backends and MCP stdio tool execution.

## Capabilities

- **A2A v1.0 agents**: Each persona is mounted as an isolated ASGI sub-app with an A2A v1.0 AgentCard and JSON-RPC endpoint (with optional v0.3 wire compatibility enabled).
- **REST API**: CRUD endpoints for personas, runtime agents, conversation sessions, and YAML/JSON persona uploads under `/api/v1`.
- **MCP tools**: Connects to configured stdio MCP servers at startup and exposes their tools to the LLM tool-calling loop.
- **OpenAI-compatible models**: Works with OpenAI, Azure OpenAI, Ollama, vLLM, LiteLLM, or any endpoint implementing OpenAI Chat Completions.
- **Declarative personas**: YAML or JSON character definitions with background, language style, knowledge domains, and optional custom system prompts.

## Quick start

```bash
git clone https://github.com/memenow/persona-agent.git
cd persona-agent
uv sync
uv run persona-agent api
```

By default the server binds to `http://127.0.0.1:8000`:

- OpenAPI UI: `/docs`
- Health probe: `/health`
- Aggregate A2A card: `/.well-known/agent-card.json` (legacy alias: `/.well-known/agent.json`)
- REST surface: `/api/v1`

Configure LLM credentials in `config/llm_config.json` or export `OPENAI_API_KEY` (with empty `api_key` values in the file). See [docs/configuration.md](docs/configuration.md) for the exact precedence rules.

## CLI

```bash
uv run persona-agent api                  # start the API server
uv run persona-agent list-personas        # list personas from PERSONAS_DIR
uv run persona-agent agent-card           # print A2A v1.0 agent cards
uv run persona-agent agent-card trump     # print one persona's agent card
uv run persona-agent import-persona FILE  # import a JSON/YAML persona file
```

## Endpoints

| Surface | Path | Auth |
|---------|------|------|
| REST | `/api/v1/personas`, `/api/v1/agents`, `/api/v1/sessions` | Optional API key |
| REST | `POST /api/v1/personas/upload` | Optional API key |
| REST | `GET/POST /api/v1/sessions/{id}/messages` | Optional API key |
| A2A | `GET /.well-known/agent-card.json`, `GET /.well-known/agent.json` | Public |
| A2A | `GET /a2a/personas` | Public |
| A2A | `GET /a2a/{persona_id}/.well-known/agent-card.json` | Public |
| A2A | `POST /a2a/{persona_id}/` | Public |
| Ops | `GET /health`, `GET /docs`, `GET /openapi.json` | Public |

## Project layout

```text
persona-agent/
├── config/                    LLM and MCP JSON configuration files
├── docs/                      Reference documentation
├── examples/personas/         Example persona definitions
├── src/persona_agent/
│   ├── __init__.py            Package version
│   ├── config.py              Settings and config file loaders
│   ├── cli.py                 Command-line entry point
│   ├── personas/              Persona model, prompt builder, file repository
│   ├── llm/                   LLMClient abstraction and OpenAI client
│   ├── mcp/                   MCPManager (stdio client lifecycle)
│   ├── a2a/                   A2A v1.0 cards, executor, and sub-app registry
│   ├── agents/                Runtime AgentService and AgentSession store
│   └── api/                   FastAPI app factory, auth, schemas, and routes
├── tests/                     pytest suite
├── CHANGELOG.md
└── pyproject.toml
```

## Documentation

| Document | Scope |
|----------|-------|
| [docs/api.md](docs/api.md) | REST endpoints, payloads, status codes, API key auth, and CORS |
| [docs/a2a.md](docs/a2a.md) | A2A v1.0 integration, AgentCard structure, v0.3 compat, and executor |
| [docs/mcp.md](docs/mcp.md) | MCP v2 stdio lifecycle, env substitution, and tool bridging |
| [docs/personas.md](docs/personas.md) | Persona file schema, ID constraints, and prompt rendering |
| [docs/configuration.md](docs/configuration.md) | Environment variables, JSON configs, deployment, and containerization |
| [docs/development.md](docs/development.md) | Local setup, formatting, linting, testing, and module conventions |

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Requires Python 3.11+. See [LICENSE](LICENSE).
