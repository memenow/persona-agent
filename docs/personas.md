# Persona Schema

Personas are stored as YAML or JSON files under `PERSONAS_DIR` (default `examples/personas/`). The canonical schema is `persona_agent.personas.model.Persona`.

## Fields

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `id` | `string` | No | Matches `^[a-z0-9_-]{1,64}$`. Auto-generated UUID when omitted via REST/API; derived from filename stem on startup scan. |
| `name` | `string` | Yes | Display name. |
| `description` | `string` | No | Short character summary. Default `""`. |
| `personal_background` | `object` | No | Key-value biographical metadata. Default `{}`. |
| `language_style` | `object` | No | Style guidance. Prompt rendering reads `tone`, `vocabulary`, `speaking_style`, and `common_phrases` (list). Default `{}`. |
| `knowledge_domains` | `object` | No | Map of domain name to topic list. Each entry also generates an A2A `AgentSkill`. Default `{}`. |
| `interaction_samples` | `array` | No | Representative quotes or sample dialogues. Default `[]`. |
| `system_prompt` | `string \| null` | No | Overrides prompt generation verbatim when set. Default `null`. |

## ID validation and normalization

- Allowed pattern: `^[a-z0-9_-]{1,64}$`.
- **Startup directory scan**: `PersonaRepository` normalizes legacy non-matching IDs (lowercasing and replacing disallowed characters with `-`, logging a warning). Files whose declared ID and filename stem cannot produce a valid ID are skipped.
- **REST upload and CLI import**: `POST /api/v1/personas/upload` and `persona-agent import-persona` validate through `Persona` directly and reject non-matching IDs (`422` over HTTP, non-zero exit in CLI).
- **Persistence safety**: `save_persona()` and `delete_persona()` re-check ID syntax and verify resolved file paths remain inside `personas_dir`.

## Examples

### Minimal

```yaml
name: Ada Lovelace
description: Mathematician and writer known for work on the Analytical Engine.
```

### Full

See [examples/personas/trump.yaml](../examples/personas/trump.yaml) for a complete example including background, language style, knowledge domains, and interaction samples.

## System prompt rendering

`Persona.build_system_prompt()` returns `system_prompt` verbatim if provided. Otherwise it assembles:

1. Identity line (`You are <name>.`) and `description`.
2. `language_style` fields (`tone`, `vocabulary`, `speaking_style`, `common_phrases`).
3. `personal_background` entries as title-cased bullet points.
4. `knowledge_domains` entries whose values are lists.
5. Standard instructions for using available MCP tools and staying in character.

## Loading and persistence

At startup, `PersonaRepository` scans `PERSONAS_DIR` for `.json`, `.yaml`, and `.yml` files. Unparseable files are logged and skipped. Personas created via `POST /api/v1/personas`, `POST /api/v1/personas/upload`, or `persona-agent import-persona` are persisted as `<id>.json` inside `PERSONAS_DIR`.
