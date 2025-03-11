"""Persona Agent Package.

A framework for creating AI agents that simulate specific personas
and can use MCP tools to enhance their capabilities.
"""

from persona_agent.api import (
    create_persona,
    get_persona,
    chat,
    list_personas,
    load_persona_from_file,
    save_persona,
    get_persona_info,
    get_persona_tools,
    PersonaAgentAPI,
)
from persona_agent.core.persona_profile import PersonaProfile
from persona_agent.core.persona_agent import PersonaAgent
from persona_agent.mcp.tool_adapter import MCPToolAdapter
from persona_agent.mcp.server_config import (
    load_mcp_config,
    save_mcp_config,
)

__version__ = "0.2.0"

__all__ = [
    # Main API functions
    "create_persona",
    "get_persona",
    "chat",
    "list_personas",
    "load_persona_from_file",
    "save_persona",
    "get_persona_info",
    "get_persona_tools",
    "PersonaAgentAPI",
    
    # Core classes
    "PersonaProfile",
    "PersonaAgent",
    
    # MCP integration
    "MCPToolAdapter",
    "load_mcp_config",
    "save_mcp_config",
]
