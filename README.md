# Persona Agent

A flexible and powerful framework for creating AI persona agents using AutoGen with Model Context Protocol (MCP) integration.

## Overview

Persona Agent is an open-source framework that enables the creation of AI agents capable of simulating specific personas with customizable personalities, knowledge domains, and behaviors. The framework leverages AutoGen for agent creation and integrates with the Model Context Protocol (MCP) to provide agents with access to external tools and resources.

## Features

- **Customizable Personas**: Create detailed persona profiles with personality traits, knowledge domains, and language style.
- **MCP Tool Integration**: Connect agents to Model Context Protocol (MCP) servers to access external tools and data sources.
- **Multi-agent Conversations**: Support for multi-agent scenarios where personas can interact with each other.
- **Flexible Deployment**: Run agents via CLI, HTTP API, or WebSocket server.
- **Extensible Architecture**: Easy to extend with new capabilities and integrations.

## Architecture

The project is organized into the following main components:

- **Core**: Contains the main `PersonaAgent` class and `PersonaProfile` for defining agent characteristics.
- **MCP**: Modules for integrating with Model Context Protocol, with multiple implementation strategies.
- **Tools**: Adapters for using external tools with personas.
- **API**: HTTP and WebSocket interfaces for interacting with agents.
- **CLI**: Command-line interface for running and managing agents.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/memenow/persona_agent.git
cd persona-agent

# Install the package
pip install -e .
```

### Installing with MCP support

To use MCP integration features, install with additional dependencies:

```bash
pip install -e ".[mcp]"
# Or
pip install -U "autogen-ext[mcp]"
```

## Configuration

### Persona Configuration

Personas are defined using YAML or JSON files with the following structure:

```yaml
name: "Albert Einstein"
description: "A simulation of the famous physicist"
personal_background:
  birth: "March 14, 1879, Ulm, Germany"
  education: "PhD in Physics from University of Zurich"
  occupation: "Theoretical Physicist"
language_style:
  tone: "Thoughtful and philosophical"
  complexity: "Uses analogies to explain complex concepts"
knowledge_domains:
  physics:
    - "Theory of Relativity"
    - "Photoelectric Effect"
    - "Brownian Motion"
  philosophy:
    - "Scientific Determinism"
    - "Pacifism"
interaction_samples:
  - type: "scientific_explanation"
    content: "When you sit with a nice girl for two hours, it seems like minutes; when you sit on a hot stove for a minute, it seems like hours. That's relativity."
```

### MCP Configuration

MCP servers are configured in a JSON file located at `config/mcp_config.json`:

```json
{
  "mcpServers": {
    "memory-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "autoApprove": ["create_entities", "read_graph"]
    },
    "web-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "autoApprove": ["brave_web_search"]
    }
  }
}
```

### LLM Configuration

Language models are configured in `config/llm_config.json`:

```json
{
  "model_configs": [
    {
      "model": "gpt-4o",
      "api_key": "your-api-key-here",
      "temperature": 0.7
    }
  ]
}
```

## Usage Examples

### Creating a Persona Agent

```python
from persona_agent.core.persona_profile import PersonaProfile
from persona_agent.core.persona_agent import PersonaAgent

# Create a persona profile
profile = PersonaProfile(
    name="Sherlock Holmes",
    description="The famous detective from Baker Street",
    personal_background={
        "occupation": "Consulting Detective",
        "residence": "221B Baker Street, London"
    },
    language_style={
        "tone": "Analytical and sometimes condescending",
        "phrases": "Elementary, my dear Watson"
    },
    knowledge_domains={
        "detection": ["Deductive reasoning", "Forensic science"],
        "criminology": ["Victorian era crime patterns"]
    }
)

# Create an agent from the profile
agent = PersonaAgent(profile=profile)

# Chat with the agent
response = agent.chat("Tell me about your methods of deduction.")
print(response)
```

### Loading a Persona from a File

```python
from persona_agent.core.persona_agent import PersonaAgent

# Load a persona from a YAML file
agent = PersonaAgent.load_persona("examples/personas/einstein.yaml")

# Chat with the agent
response = agent.chat("Can you explain relativity in simple terms?")
print(response)
```

### Enabling MCP Tools

```python
import json
from persona_agent.core.persona_agent import PersonaAgent

# Load a persona
agent = PersonaAgent.load_persona("examples/personas/tesla.yaml")

# Load MCP configuration
with open("config/mcp_config.json", "r") as f:
    mcp_config = json.load(f)

# Enable MCP tools
agent.enable_mcp_tools(mcp_config)

# Now the agent can use tools from MCP servers
response = agent.chat("Search for recent breakthroughs in renewable energy.")
print(response)
```

### Running the HTTP API

```bash
# Run the HTTP API server
python -m persona_agent.http_api --port 8000
```

### Running the WebSocket Server

```bash
# Run the WebSocket server
python -m persona_agent.websocket_server --port 8001
```

## MCP Integration

The framework supports integration with the Model Context Protocol (MCP), which allows persona agents to access external tools and resources from MCP servers. There are three implementation strategies:

1. **Primary Implementation**: Uses `autogen-ext[mcp]` for direct integration with MCP servers.
2. **Direct Connector**: A standalone implementation that works without external dependencies.
3. **Compatibility Layer**: For older code that uses the legacy interface.

To use MCP features:

1. Install required dependencies: `pip install -U "autogen-ext[mcp]"`
2. Configure MCP servers in `config/mcp_config.json`
3. Enable MCP tools on your agent using `agent.enable_mcp_tools(mcp_config)`

## API Reference

### Core Classes

- `PersonaProfile`: Represents a persona's characteristics, knowledge, and behavior.
- `PersonaAgent`: The main agent class that simulates a persona using AutoGen.

### MCP Classes

- `MCPToolAdapter`: Adapts MCP tools for use with persona agents.
- `DirectMCPConnector`: A direct implementation of MCP client functionality.

### Utility Functions

- `load_mcp_config()`: Load MCP configuration from a file.
- `get_available_servers()`: Get a list of available MCP servers.
- `use_mcp_tool()`: Execute an MCP tool directly.

## Development

### Project Structure

```
persona_agent/
├── core/              # Core agent functionality
│   ├── persona_agent.py
│   └── persona_profile.py
├── mcp/               # MCP integration
│   ├── direct_connector.py
│   ├── mcp.py
│   ├── server_config.py
│   └── tool_adapter.py
├── tools/             # Tool adapters
│   ├── mcp_connector.py
│   └── mcp_tool_registry.py
├── utils/             # Utility functions
├── api.py             # API interface
├── cli.py             # Command-line interface
├── http_api.py        # HTTP API server
└── websocket_server.py # WebSocket server
```

### Extending the Framework

#### Adding Custom Tools

1. Create a function that implements the tool logic
2. Register the tool with the agent's tool adapter
3. Update the agent's system message to include the new tool

Example:

```python
def custom_tool(param1: str, param2: int) -> Dict[str, Any]:
    """A custom tool that does something useful.
    
    Args:
        param1: A string parameter.
        param2: An integer parameter.
        
    Returns:
        The result of the tool execution.
    """
    result = do_something(param1, param2)
    return {"result": result}

# Register with the agent
agent.tool_adapter.register_tool(
    server_name="custom_server",
    tool_name="custom_tool",
    description="A custom tool that does something useful",
    input_schema={
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
            "param2": {"type": "integer"}
        },
        "required": ["param1", "param2"]
    }
)
```

#### Adding a New MCP Server

1. Create a server configuration in `config/mcp_config.json`
2. Install and run the MCP server
3. Enable the server's tools on your agent

```json
{
  "mcpServers": {
    "my-new-server": {
      "command": "npx",
      "args": ["-y", "my-mcp-server-package"],
      "autoApprove": ["tool1", "tool2"]
    }
  }
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.