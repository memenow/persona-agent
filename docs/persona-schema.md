# Persona Schema

A persona is a YAML or JSON file describing a character. The
`Persona` Pydantic model in
`src/persona_agent/api/persona_manager.py` is the authoritative schema.

## Fields

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `id` | string | no | Auto-generated UUID when omitted. Must match `^[a-z0-9_-]{1,64}$`. Files **discovered at startup** under `PERSONAS_DIR` are run through `PersonaManager._load_persona_file`, which sanitizes non-matching values in place (with a warning log) so legacy files still load. Files supplied through `POST /api/v1/personas/upload` or `persona-agent import-persona` go through `add_persona()`, which constructs a `Persona` directly — Pydantic rejects non-matching IDs with a validation error rather than sanitizing them. |
| `name` | string | **yes** | Display name. |
| `description` | string | no | One- or two-sentence summary. Default `""`. |
| `personal_background` | object | no | Free-form key/value details (birth, education, profession, …). Default `{}`. |
| `language_style` | object | no | Free-form style hints. Default `{}`. Recognized keys used by the system-prompt generator: `tone`, `vocabulary`, `speaking_style`, `common_phrases` (list). |
| `knowledge_domains` | object | no | Map of domain → list of topics. Each domain becomes an A2A AgentSkill. Default `{}`. |
| `interaction_samples` | array | no | List of example exchanges. Default `[]`. |
| `system_prompt` | string \| null | no | Overrides the auto-generated prompt when set. Default `null`. |

### ID rules

- Allowed characters: lowercase ASCII letters, digits, hyphen,
  underscore.
- Length: 1–64 characters.
- The validator runs on **load**, **save**, and **delete**. The save path
  additionally rejects `..`, `/`, `\`, and null bytes as defense-in-depth
  against path traversal even though the regex already forbids them.

When `id` is missing from a file the loader derives one from the filename
stem (e.g. `trump.yaml` → `trump`), sanitizing if necessary.

## Examples

### Minimal

```yaml
name: Ada Lovelace
description: Mathematician and pioneer of computing.
```

### Full

```yaml
id: trump
name: Donald Trump
description: 45th President of the United States, businessman, and media personality known for his distinctive communication style.

personal_background:
  birth: June 14, 1946, Queens, New York City, USA
  education: Wharton School of the University of Pennsylvania
  profession: Businessman, Politician, Television Personality
  political_party: Republican

language_style:
  tone: Assertive, direct, hyperbolic
  vocabulary: Simple, repetitive, superlative-heavy
  speaking_style: Stream of consciousness, frequent use of simple sentences
  common_phrases:
    - "Believe me"
    - "Tremendous"
    - "Make America Great Again"

knowledge_domains:
  politics:
    - American politics
    - Immigration policy
    - Trade policy
  business:
    - Real estate
    - Branding
    - Negotiation

interaction_samples:
  - type: quote
    content: "We will make America strong again."
  - type: conversation
    content: |
      Q: What's your approach to international trade?
      A: We've been taken advantage of by other countries for decades. …
```

See `examples/personas/trump.yaml` for a working file used by the test
suite.

## System prompt generation

`Persona.generate_system_prompt()` builds the LLM system message. If the
persona file sets `system_prompt`, that value is returned verbatim.
Otherwise the generator stitches together:

1. `"You are <name>."`
2. The `description`.
3. **Language style** (when present): tone, vocabulary, speaking style,
   and a "Frequently use phrases like: …" line if `common_phrases` is a
   list.
4. **Background**: each `personal_background` key formatted as
   `- <Title-Cased Key>: <value>`.
5. **Knowledge domains** (when each value is a list): comma-joined topics
   per domain.
6. A reminder that MCP tools are available and should be used when
   helpful.
7. A reminder to stay in character.

Override with `system_prompt` when you need exact control — e.g. for
deterministic replay in tests or for personas whose voice needs more than
the templated blocks can express.

## Loading

Personas are loaded at startup from `PERSONAS_DIR` (default
`examples/personas/`). Any file ending in `.json`, `.yaml`, or `.yml` is
parsed. Files that fail to parse are skipped with a warning; the server
continues with the remaining personas.

Newly created personas (via `POST /api/v1/personas` or
`POST /api/v1/personas/upload`) are saved as JSON in the same directory.
The `import-persona` CLI command does the same from the command line.
