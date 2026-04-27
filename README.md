# Persona Agent

A Python-based API server for creating and interacting with AI personas using the [Google A2A (Agent-to-Agent) protocol](https://github.com/google/A2A) and Model Context Protocol (MCP) tools integration.

## Overview

This project provides a robust API for creating AI personas that can interact with users through natural language. Built on the Google A2A protocol and the `a2a-sdk`, each persona is exposed as a discoverable A2A agent with standardized agent cards, JSON-RPC messaging, and external tool capabilities via MCP.

## Features

- **Persona-based AI Agents**: Create and interact with AI agents that simulate specific personas
- **Google A2A Protocol**: Each persona is an A2A-compliant agent with discoverable agent cards and JSON-RPC endpoints
- **Model Context Protocol Integration**: Enhance AI capabilities with external tools through MCP stdio servers
- **RESTful API**: Comprehensive REST API for managing personas, agents, and conversations
- **Tool-Augmented Responses**: Enable agents to use external tools to respond to user queries
- **Configurable Behavior**: Customize persona characteristics through YAML/JSON configuration files
- **OpenAI-Compatible LLM Support**: Works with any OpenAI-compatible provider (OpenAI, Azure, Ollama, vLLM, etc.)

## Architecture

The project is organized into several key components:

- **API Server**: FastAPI implementation for REST API endpoints and A2A sub-app mounting
- **A2A Integration**: `PersonaAgentExecutor` implements the A2A executor interface; each persona is mounted as an independent ASSI sub-app
- **Persona Management**: Load and manage persona definitions from JSON/YAML files
- **Agent Factory**: Create and configure persona agents with LLM clients and MCP tools
- **LLM Client**: Framework-agnostic abstraction using the `openai` SDK directly
- **MCP Integration**: `DirectMCPManager` for stdio server lifecycle, tool discovery, and execution
- **Session Management**: Handle conversation sessions between users and agents

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/memenow/persona-agent.git
   cd persona-agent
   ```

2. Install dependencies using [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync
   ```

3. Configure API keys:
   Create a `config/llm_config.json` file with your API keys and model configurations:
   ```json
   {
     "default_model": "gpt-4o",
     "api_key": "your-api-key-here",
     "api_base": "https://api.openai.com/v1"
   }
   ```

## Usage

### Running the API Server

Start the API server:

```bash
uv run persona-agent api
```

The API will be available at http://localhost:8000/api/v1/ with Swagger documentation at http://localhost:8000/docs.

### CLI Commands

```bash
uv run persona-agent api              # Start API server
uv run persona-agent list-personas    # List available personas
uv run persona-agent agent-card       # Show A2A agent cards
uv run persona-agent import-persona FILE  # Import persona file
```

### API Endpoints

#### A2A Endpoints

- `GET /.well-known/agent.json`: Aggregate agent card for all personas
- `GET /a2a/{persona_id}/.well-known/agent-card.json`: Individual persona agent card
- `POST /a2a/{persona_id}/`: A2A JSON-RPC endpoint
- `GET /a2a/personas`: List all A2A persona agents

#### Personas API

- `GET /api/v1/personas`: List all available personas
- `GET /api/v1/personas/{id}`: Get a specific persona's details
- `POST /api/v1/personas`: Create a new persona
- `PUT /api/v1/personas/{id}`: Update an existing persona
- `DELETE /api/v1/personas/{id}`: Delete a persona

#### Agents API

- `GET /api/v1/agents`: List all active agents
- `GET /api/v1/agents/{id}`: Get a specific agent's details
- `POST /api/v1/agents`: Create a new agent based on a persona
- `DELETE /api/v1/agents/{id}`: Delete an agent

#### Sessions API

- `GET /api/v1/sessions`: List all active sessions
- `GET /api/v1/sessions/{id}`: Get a specific session's details
- `POST /api/v1/sessions`: Create a new conversation session
- `DELETE /api/v1/sessions/{id}`: Delete a session
- `POST /api/v1/sessions/{id}/messages`: Send a message to an agent
- `GET /api/v1/sessions/{id}/events`: Stream session events (SSE)

### Persona Configuration

Personas can be defined in JSON or YAML format:

```json
{
  "name": "Albert Einstein",
  "description": "Theoretical physicist and Nobel laureate",
  "personal_background": {
    "birth": "March 14, 1879, Ulm, Germany",
    "education": "ETH Zurich, University of Zurich",
    "profession": "Physicist, Professor"
  },
  "language_style": {
    "tone": "Thoughtful, inquisitive, sometimes whimsical",
    "common_phrases": ["Imagination is more important than knowledge", "Everything should be made as simple as possible, but not simpler"]
  },
  "knowledge_domains": {
    "physics": ["Relativity theory", "Quantum mechanics", "Brownian motion"],
    "philosophy": ["Scientific determinism", "Pacifism", "Religious views"]
  },
  "interaction_samples": [
    {
      "type": "conversation",
      "content": "Q: What is the most important scientific principle?\nA: The principle of curiosity - to never stop questioning. That is the source of all knowledge and discovery."
    }
  ]
}
```

### MCP Configuration

To configure MCP services, create a `config/mcp_config.json` file:

```json
{
  "mcpServers": {
    "brave_search": {
      "command": "node",
      "args": ["path/to/mcp-brave-search/index.js"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      },
      "description": "Brave Search MCP service"
    }
  }
}
```

Environment variables in the configuration (like `${BRAVE_API_KEY}`) will be automatically resolved at runtime.

## Project Structure

```
persona-agent/
├── config/                  # Configuration files
│   ├── llm_config.json      # LLM API keys and settings
│   └── mcp_config.json      # MCP services configuration
├── examples/                # Example code and personas
│   └── personas/            # Example persona definitions
├── src/                     # Source code
│   └── persona_agent/       # Main package
│       ├── a2a/             # A2A protocol integration
│       │   ├── agent_card.py    # AgentCard builder
│       │   └── executor.py      # PersonaAgentExecutor
│       ├── api/             # API implementation
│       │   ├── routes/      # API route handlers
│       │   ├── agent_factory.py # Agent creation factory
│       │   ├── config.py    # API configuration
│       │   ├── dependencies.py  # FastAPI dependencies
│       │   ├── models.py    # Pydantic API models
│       │   ├── persona_manager.py # Persona data management
│       │   └── server.py    # FastAPI server + A2A registry
│       ├── core/            # Core functionality
│       │   └── persona_profile.py # PersonaProfile dataclass
│       ├── llm/             # LLM client abstraction
│       │   └── client.py    # OpenAI-compatible client
│       ├── mcp/             # MCP integration
│       │   └── direct_mcp.py # Direct MCP stdio manager
│       └── cli.py           # Command-line interface
├── tests/                   # Test suite
├── pyproject.toml           # Project dependencies and metadata
└── uv.lock                  # Dependency lock file
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/memenow/persona-agent.git
cd persona-agent

# Install dependencies
uv sync

# Lint and format
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest
```

### Adding New MCP Services

1. Add the service configuration to `config/mcp_config.json`
2. The service will be automatically loaded by the `DirectMCPManager` class

### Extending Personas

To add new persona capabilities:

1. Enhance the `PersonaProfile` class in `src/persona_agent/core/persona_profile.py`
2. Update the persona JSON/YAML schema accordingly
3. Update the API models in `src/persona_agent/api/models.py`
