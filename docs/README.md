# Documentation

In-depth documentation for the Persona Agent project. The top-level
[`README.md`](../README.md) is the quick-start entry point; this directory
documents the details that don't fit there.

## Contents

| File | Use it for |
|------|------------|
| [api-reference.md](api-reference.md) | REST API endpoints, request/response shapes, status codes. |
| [a2a-protocol.md](a2a-protocol.md) | Google A2A protocol integration, agent cards, JSON-RPC endpoints, mount strategy. |
| [persona-schema.md](persona-schema.md) | Persona definition format, field semantics, ID rules, examples. |
| [configuration.md](configuration.md) | Environment variables, `llm_config.json`, `mcp_config.json`, precedence. |
| [mcp-integration.md](mcp-integration.md) | MCP tool integration, stdio lifecycle, env-var substitution, retries. |
| [authentication.md](authentication.md) | API key auth, allow-list, CORS interaction, public vs protected routes. |
| [deployment.md](deployment.md) | Running the server, reverse proxies, logging, process model. |
| [development.md](development.md) | Development setup, lint, format, test workflows. |
| [changelog.md](changelog.md) | Notable changes since the AutoGen → A2A migration. |

## Conventions

- Examples assume the server is reachable at `http://127.0.0.1:8000` (the
  default bind). Adjust the host/port for your environment.
- JSON bodies are shown as pretty-printed. The server accepts compact JSON
  as well.
- Environment variables follow the prefix `API_*` for server runtime,
  `OPENAI_*` for LLM client, `PERSONAS_DIR`/`LLM_CONFIG_PATH`/`MCP_CONFIG_PATH`
  for file locations.
