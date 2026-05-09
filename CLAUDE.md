# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AI persona agents built on Google A2A protocol with FastAPI REST API and MCP tool integration. Personas are defined as YAML/JSON files loaded from `examples/personas/`.

## Commands

```bash
# Install dependencies
uv sync

# Lint and format
uv run ruff check .
uv run ruff format .

# CLI commands
uv run persona-agent api              # Start API server
uv run persona-agent list-personas    # List available personas
uv run persona-agent agent-card       # Show A2A agent cards
uv run persona-agent import-persona FILE  # Import persona file
```

## Architecture

Three-layer structure under `src/persona_agent/`:

- **api/** — FastAPI routes, models, dependency injection. REST routes prefixed with `/api/v1`, A2A routes at `/a2a/`.
- **a2a/** — A2A protocol integration: `PersonaAgentExecutor` (implements `AgentExecutor`), `AgentCard` builder. Each persona is an A2A agent.
- **llm/** — LLM client abstraction with `OpenAICompatibleClient` using the `openai` SDK directly. Supports any OpenAI-compatible provider.
- **mcp/** — `DirectMCPManager` using the `mcp` library directly for stdio server lifecycle, tool loading, and execution.

## Key Files

- `api/agent_factory.py` — Creates `PersonaAgentExecutor` instances, manages sessions. Uses `LLMClient` + `DirectMCPManager`.
- `api/routes/a2a.py` — A2A agent card discovery and registry. Each persona is mounted as an independent ASGI sub-app via the SDK's `build()` API.
- `a2a/executor.py` — Core agent logic: LLM chat loop with MCP tool calling.
- `llm/client.py` — Abstract `LLMClient` + `OpenAICompatibleClient` implementation.
- `mcp/direct_mcp.py` — Direct MCP stdio server management without framework wrappers.

## Configuration

Precedence: environment variables > JSON config files > hardcoded defaults in `api/config.py`.

Key env vars: `OPENAI_API_KEY`, `PERSONAS_DIR` (default: `examples/personas`), `LLM_CONFIG_PATH` (default: `config/llm_config.json`), `MCP_CONFIG_PATH` (default: `config/mcp_config.json`).

## A2A Endpoints

- `GET /.well-known/agent.json` — Aggregate agent card for all personas
- `GET /a2a/{persona_id}/.well-known/agent-card.json` — Individual persona agent card (SDK default path)
- `POST /a2a/{persona_id}/` — A2A JSON-RPC endpoint (SDK sub-app)
- `GET /a2a/personas` — List all A2A persona agents

## Persona Schema

Persona YAML files require: `name`, `description`, `personal_background`, `language_style`, `knowledge_domains`, `interaction_samples`. The `Persona` Pydantic model (in `api/persona_manager.py`) validates structure and constrains `id` to `^[a-z0-9_-]{1,64}$`.

## MCP Config

`config/mcp_config.json` supports both `"services"` (old) and `"mcpServers"` (new) section names. Environment variables in commands/args use `${VAR_NAME}` syntax.

## Git Workflow

Fork-based: `origin` is your fork, `upstream` is `memenow/persona-agent`. PR to upstream.

## Code Style

- Python with type hints throughout
- PEP 8 conventions
- Use `ruff` for linting and formatting

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **persona-agent** (737 symbols, 1116 relationships, 12 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/persona-agent/context` | Codebase overview, check index freshness |
| `gitnexus://repo/persona-agent/clusters` | All functional areas |
| `gitnexus://repo/persona-agent/processes` | All execution flows |
| `gitnexus://repo/persona-agent/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
