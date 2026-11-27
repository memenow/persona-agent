# MCP Integration

`persona_agent.mcp.manager.MCPManager` connects to stdio Model Context Protocol servers using the official `mcp` Python SDK (`>=2.3.0`).

## Lifecycle

- `create_app()` constructs an `MCPManager` and attaches it to `app.state.mcp_manager`.
- During FastAPI lifespan startup, if `MCP_CONFIG_PATH` exists, `load_config()` connects to every enabled server and discovers its tools.
- During shutdown, `close()` unwinds the shared `AsyncExitStack`, terminating all child stdio processes and clearing internal registries.

If `mcp_config.json` is missing or declares no enabled servers, the application starts normally with zero tools.

## Configuration format

`config/mcp_config.json` accepts `"mcpServers"` (current) and `"services"` (legacy; entries must have `type: "stdio"` and `enabled: true`). Both sections are merged if present.

```json
{
  "mcpServers": {
    "mcp-server-fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"],
      "disabled": false,
      "description": "Fetch and convert web pages"
    },
    "brave_search": {
      "command": "node",
      "args": ["path/to/mcp-brave-search/index.js"],
      "env": { "BRAVE_API_KEY": "${BRAVE_API_KEY}" }
    }
  }
}
```

### Environment substitution

`${VAR_NAME}` placeholders in `command`, `args`, and `env` values are expanded from `os.environ` at load time. Undefined variables expand to an empty string.

Each spawned stdio subprocess inherits the parent `os.environ` with the server entry's `env` dictionary merged on top.

## Connection and retry behavior

Each enabled server is connected via `mcp.Client(StdioServerParameters(...))` with up to `MAX_RETRIES = 3` attempts:

- Every attempt enters a local `AsyncExitStack` so failed handshakes or timeouts never leak child processes.
- After `Client` negotiates the protocol and `list_tools()` returns at least one tool, ownership of the local stack transfers to the manager's shared `AsyncExitStack`.
- Servers that advertise zero tools are closed immediately and skipped without blocking remaining servers.
- Tool names are mapped in a flat table (`tool_name -> service_name`), so tool names must be unique across configured servers.

## Tool schema conversion and execution

`mcp_tools_to_openai_functions()` translates each MCP v2 `Tool` into an OpenAI function tool schema using `tool.input_schema` (defaulting to `{"type": "object", "properties": {}}` when omitted).

During an agent turn, `MCPManager.call_tool(tool_name, arguments)`:

1. Returns `{"error": "Tool '<name>' not found"}` for unknown tool names.
2. Invokes `client.call_tool(tool_name, arguments)` and joins text blocks from `result.content`.
3. Checks `result.is_error` and returns a JSON error object if the server flagged a tool error.
4. Catches `MCPError` and unexpected exceptions, logs the traceback, and returns `{"error": "Tool execution failed: <name>"}` so tool failures never abort the chat loop.
