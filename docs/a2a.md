# A2A Protocol Integration

Each loaded persona is exposed as an independent [A2A v1.0](https://github.com/a2aproject/A2A) agent via `a2a-sdk` (`>=1.3.1`).

## Modules

| Module | Responsibility |
|--------|----------------|
| `persona_agent.a2a.cards` | Builds Protobuf `AgentCard` objects (per-persona and aggregate) and serializes them via `agent_card_to_dict()`. |
| `persona_agent.a2a.executor` | Implements `AgentExecutor` (`PersonaAgentExecutor`), driving the LLM + MCP tool loop and emitting A2A v1.0 `Task` / `TaskUpdater` events. |
| `persona_agent.a2a.registry` | Builds and mounts isolated FastAPI sub-apps for each persona using `add_a2a_routes_to_fastapi`. |
| `persona_agent.api.routes.a2a` | Exposes aggregate discovery endpoints backed by `app.state.a2a_registry`. |

## Endpoints

| Path | Method | Notes |
|------|--------|-------|
| `/.well-known/agent-card.json` | `GET` | Canonical A2A v1.0 aggregate AgentCard |
| `/.well-known/agent.json` | `GET` | Compatibility alias returning the same aggregate AgentCard |
| `/a2a/personas` | `GET` | Summary list of mounted persona agents |
| `/a2a/{persona_id}/.well-known/agent-card.json` | `GET` | Per-persona A2A v1.0 AgentCard |
| `/a2a/{persona_id}/` | `POST` | Per-persona JSON-RPC endpoint (`A2A-Version: 1.0` and `0.3` compat) |

> [!NOTE]
> The A2A registry is built once during application startup from `PERSONAS_DIR`. Personas created, modified, or deleted later via the REST API update `PersonaRepository` and disk files, but A2A mounts reflect boot-time state until restart.

## AgentCard v1.0 format

`build_agent_card()` generates an A2A v1.0 Protobuf `AgentCard` containing:

- `supportedInterfaces`: one `JSONRPC` interface entry with `protocolVersion: "1.0"` and `url: "{base_url}/a2a/{persona_id}/"`, where `base_url` comes from `Settings.resolve_public_base_url()`.
- `capabilities`: `streaming: true`, `pushNotifications: false`.
- `defaultInputModes` / `defaultOutputModes`: `["text/plain"]`.
- `skills`: one general conversation skill plus one skill per `knowledge_domains` entry.

Inspect generated cards without starting the server:

```bash
uv run persona-agent agent-card
uv run persona-agent agent-card trump
```

## Sub-app mounting and v0.3 compatibility

For each persona, `A2ARegistry.register_persona()` constructs a `DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)` and wires routes onto a dedicated `FastAPI` sub-app:

```python
add_a2a_routes_to_fastapi(
    sub_app,
    agent_card_routes=create_agent_card_routes(card),
    jsonrpc_routes=create_jsonrpc_routes(
        request_handler,
        rpc_url="/",
        enable_v0_3_compat=True,
    ),
)
```

Mounting at `/a2a/{persona_id}` namespaces both `/.well-known/agent-card.json` and the root `/` JSON-RPC endpoint per persona. Setting `enable_v0_3_compat=True` allows both A2A v1.0 (`SendMessage`) and v0.3 (`message/send`) clients to interoperate on the same endpoint.

Task storage uses `InMemoryTaskStore` (per-process, non-durable across restarts).

## Executor lifecycle

When an A2A request reaches `PersonaAgentExecutor.execute()`:

1. If `context.current_task` is `None`, enqueues an initial `Task` in `TASK_STATE_SUBMITTED` (required by A2A v1.0 before any `TaskStatusUpdateEvent`).
2. Validates non-empty user input via `context.get_user_input()`; empty messages emit `TASK_STATE_FAILED`.
3. Emits `TASK_STATE_WORKING` via `TaskUpdater.start_work()`.
4. Runs `chat(context_id, user_text)` — the same execution path used by REST sessions.
5. Emits `TASK_STATE_COMPLETED` with a `Part(text=...)` agent message, or `TASK_STATE_FAILED` with a generic error message on exception.

Concurrency and tool limits inside `chat()`:

- Calls sharing a `context_id` are serialized with a per-context `asyncio.Lock`.
- Histories are kept in an LRU `OrderedDict` capped at `MAX_CONTEXT_HISTORIES = 200`; evicting a context also drops its lock.
- Tool calling iterates up to `MAX_TOOL_ITERATIONS = 10`, then forces a final text response with `tool_choice="none"` while retaining prior tool messages.

`cancel()` emits `TASK_STATE_CANCELED` via `TaskUpdater.cancel()`. Underlying in-flight LLM or MCP calls are not preempted.
