# A2A Protocol Integration

The project exposes every persona as an independent [Google A2A
protocol](https://github.com/google/A2A) agent. The integration is built
directly on `a2a-sdk` — there is no framework wrapper between the SDK and
our executor.

## Components

| Module | Role |
|--------|------|
| `persona_agent.a2a.executor.PersonaAgentExecutor` | Implements `a2a.server.agent_execution.AgentExecutor`; runs the LLM chat loop with MCP tool calls and emits A2A task status updates. |
| `persona_agent.a2a.agent_card.build_agent_card()` | Builds an `AgentCard` (skills, capabilities, URL) from a persona definition. |
| `persona_agent.api.routes.a2a.A2ARegistry` | Owns the per-persona `A2AFastAPIApplication`, calls its `.build()` method to obtain an ASGI sub-app, and mounts it onto the parent FastAPI app. |

## Endpoints

| Path | Method | Auth | Source |
|------|--------|------|--------|
| `/.well-known/agent.json` | `GET` | public | `A2ARegistry.build_aggregate_card()` — aggregate card listing every persona. |
| `/a2a/personas` | `GET` | public | `A2ARegistry.list_personas()` — JSON list of registered persona agents. |
| `/a2a/{persona_id}/.well-known/agent-card.json` | `GET` | public | Provided by the SDK sub-app. |
| `/a2a/{persona_id}/` | `POST` | public | JSON-RPC endpoint provided by the SDK sub-app; this is where A2A clients send messages and task control RPCs. |

> A2A surfaces are public by protocol design: external agents must be able
> to discover the hub without holding an API key. Authentication only
> gates the REST CRUD surface under `/api/v1` (see
> [authentication.md](authentication.md)).

> **The A2A registry is built once at startup.** `create_app()` constructs
> `A2ARegistry` inside its lifespan handler from the personas present
> when the server boots and mounts one sub-app per persona; the registry
> is not resynchronized afterwards. Personas created, updated, uploaded,
> or deleted via the REST API mutate `PersonaManager` and the persona
> files on disk, but the aggregate `/.well-known/agent.json`,
> `/a2a/personas`, and per-persona `/a2a/{persona_id}/...` routes keep
> serving the boot-time set until the process is restarted. Plan REST
> persona changes alongside a restart when those changes need to be
> reachable over A2A, or restrict REST mutation flows to internal
> tooling.

## AgentCard

The card is generated from the persona's `name`, `description`, and
`knowledge_domains`. It declares:

- **`url`** — `f"{base_url}/a2a/{persona_id}/"` where `base_url` is taken
  from `API_PUBLIC_BASE_URL` if set, otherwise built from `API_HOST`/`API_PORT`.
- **`capabilities`** — `streaming=True`, `state_transition_history=True`,
  `push_notifications=False`.
- **`default_input_modes` / `default_output_modes`** — `["text/plain"]`.
- **`skills`** — one general "conversation" skill plus one skill per
  knowledge domain in the persona file:

```json
{
  "id": "trump-politics",
  "name": "Donald Trump on Politics",
  "description": "Ask Donald Trump about Politics: American politics, Immigration policy, Trade policy, …",
  "tags": ["politics", "trump", "expertise"],
  "examples": ["What's your view on American politics?"]
}
```

You can inspect the generated card without starting the server:

```bash
uv run persona-agent agent-card               # all personas
uv run persona-agent agent-card trump         # a specific persona
```

## Mount strategy

For each persona the registry constructs:

```python
A2AFastAPIApplication(agent_card=card, http_handler=request_handler)
```

It then calls `.build()` and mounts the returned ASGI app at
`/a2a/{persona_id}`. Because each persona owns an isolated sub-app, the
SDK's default routes (`/`, `/.well-known/agent-card.json`) are namespaced
to that path prefix automatically.

The request handler is the SDK's `DefaultRequestHandler`, wired to:

- `agent_executor` — our `PersonaAgentExecutor`.
- `task_store` — `InMemoryTaskStore` (per-process; tasks are not
  durable across restarts).
- `queue_manager` — `InMemoryQueueManager` for streaming events.

## Executor behavior

`PersonaAgentExecutor.execute()` is the entry point invoked by the SDK
when an A2A task arrives.

1. Reads the user message via `context.get_user_input()`.
2. Emits a `working` status update.
3. Calls `chat(context_id, user_text)` — the shared chat method also used
   by the REST `send_message` path.
4. Emits a `completed` status update with the assistant message, or a
   `failed` status update with a generic error message if the chat raises.

`chat()` serializes concurrent calls sharing the same `context_id` via an
`asyncio.Lock`, so two clients on the same conversation cannot interleave
history writes. History is stored as an `OrderedDict` LRU capped at 200
contexts; the oldest entry is evicted (along with its lock) when the cap
is exceeded.

Tool calls are iterated up to `MAX_TOOL_ITERATIONS = 10`. When the cap is
hit, the executor forces a final completion with `tool_choice="none"` so
the model produces a textual answer without invoking another tool. If no
MCP tools are loaded, the loop short-circuits to a plain completion.

## Cancellation

`PersonaAgentExecutor.cancel()` emits a `canceled` task status. The
underlying LLM/MCP work is not preempted (the OpenAI SDK does not expose
cooperative cancellation here); the cancel signal is best-effort.
