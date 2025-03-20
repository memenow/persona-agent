# Persona Agent

A Python-based API server for creating and interacting with AI personas using the AutoGen framework and Model Context Protocol (MCP) tools integration.

## Overview

This project provides a robust API for creating AI personas that can interact with users through natural language. Built on top of AutoGen 0.4, it allows for the creation of agents that can use external tools and services through the Model Context Protocol (MCP) to enhance their capabilities.

## Features

- **Persona-based AI Agents**: Create and interact with AI agents that simulate specific personas
- **Model Context Protocol Integration**: Enhance AI capabilities with external tools through MCP
- **RESTful API**: Provides a comprehensive REST API for managing personas, agents, and conversations
- **Tool-Augmented Responses**: Enable agents to use external tools to respond to user queries
- **Configurable Behavior**: Customize persona characteristics through configuration files
- **AutoGen 0.4 Support**: Compatible with the latest AutoGen framework features

## Architecture

The project is organized into several key components:

- **API Server**: FastAPI implementation for the REST API endpoints
- **Persona Management**: Load and manage persona definitions from JSON/YAML files
- **Agent Factory**: Create and configure AutoGen agents based on personas
- **MCP Integration**: Connect to external MCP services for enhanced capabilities
- **Session Management**: Handle conversation sessions between users and agents

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/memenow/persona-agent.git
   cd persona-agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure API keys:
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

Start the API server using the provided script:

```bash
python run_api_server.py
```

The API will be available at http://localhost:8000/api/v1/ with Swagger documentation at http://localhost:8000/docs.

### API Endpoints

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
- `GET /api/v1/sessions/{id}/messages`: Get all messages in a session

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
│       ├── api/             # API implementation
│       │   ├── routes/      # API route handlers
│       │   ├── agent_factory.py # Agent creation factory
│       │   ├── config.py    # API configuration
│       │   ├── dependencies.py  # FastAPI dependencies
│       │   ├── models.py    # Pydantic API models
│       │   ├── persona_manager.py # Persona data management
│       │   └── server.py    # FastAPI server
│       ├── core/            # Core functionality
│       ├── mcp/             # MCP integration
│       └── cli.py           # Command-line interface
├── tests/                   # Test suite
├── run_api_server.py        # Server startup script
├── requirements.txt         # Project dependencies
└── LICENSE                  # License
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/memenow/persona-agent.git
cd persona-agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black isort mypy

# Run tests
pytest
```

### Adding New MCP Services

1. Create a new MCP service implementation
2. Add the service configuration to `config/mcp_config.json`
3. The service will be automatically loaded by the `McpManager` class

### Extending Personas

To add new persona capabilities:

1. Enhance the `PersonaProfile` class in `src/persona_agent/core/persona_profile.py`
2. Update the persona JSON/YAML schema accordingly
3. Update the API models in `src/persona_agent/api/models.py`

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.