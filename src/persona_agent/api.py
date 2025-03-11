"""Persona Agent API module.

This module provides a high-level API for creating and managing persona agents.
It serves as the main entry point for using the persona agent system.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from persona_agent.core.persona_agent import PersonaAgent
from persona_agent.core.persona_profile import PersonaProfile
from persona_agent.utils.config_loader import ConfigLoader
from persona_agent.mcp.server_config import load_mcp_config


class PersonaAgentAPI:
    """API for creating and managing persona agents.
    
    This class provides a high-level API for creating and interacting with
    persona agents. It manages a pool of agents and handles the creation,
    retrieval, and deletion of agents.
    
    Attributes:
        agents: Dictionary mapping agent IDs to PersonaAgent instances.
        config_loader: Loader for persona configuration files.
    """
    
    def __init__(self):
        """Initialize a new PersonaAgentAPI."""
        self.agents = {}
        self.config_loader = ConfigLoader()
        self.logger = logging.getLogger(__name__)
        
        # Load MCP configuration if available
        self.mcp_config = None
        try:
            self.mcp_config = load_mcp_config("config/mcp_config.json")
            self.logger.info("Loaded MCP configuration")
        except Exception as e:
            self.logger.warning(f"Failed to load MCP configuration: {e}")
    
    def create_persona(
        self,
        profile_path: Optional[str] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        persona_id: Optional[str] = None,
        enable_mcp_tools: bool = False,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new persona agent.
        
        Args:
            profile_path: Path to a YAML or JSON file with persona configuration.
            profile_data: Dictionary containing persona configuration data.
            persona_id: Optional ID for the persona. If not provided, one will be generated.
            enable_mcp_tools: Whether to enable MCP tools for this persona.
            llm_config: Custom LLM configuration for this persona.
            
        Returns:
            The ID of the created persona.
            
        Raises:
            ValueError: If neither profile_path nor profile_data is provided.
        """
        if not profile_path and not profile_data:
            raise ValueError("Either profile_path or profile_data must be provided")
        
        # Load the profile
        if profile_path:
            profile = self.config_loader.load_from_yaml(profile_path)
        else:
            profile = PersonaProfile.from_dict(profile_data)
        
        # Generate a persona ID if not provided
        if not persona_id:
            # Create a valid Python identifier from the name
            # Replace spaces with underscores and remove any non-alphanumeric characters
            import re
            persona_id = re.sub(r'\W', '', profile.name.replace(' ', '_')).lower()
        
        # Create the persona agent
        agent = PersonaAgent(profile=profile, llm_config=llm_config)
        
        # Enable MCP tools if requested
        if enable_mcp_tools and self.mcp_config:
            try:
                agent.enable_mcp_tools(self.mcp_config)
                self.logger.info(f"Enabled MCP tools for persona {persona_id}")
            except Exception as e:
                self.logger.error(f"Failed to enable MCP tools for persona {persona_id}: {e}")
        
        # Store the agent
        self.agents[persona_id] = agent
        self.logger.info(f"Created persona {persona_id}")
        
        return persona_id
    
    def get_persona(self, persona_id: str) -> Optional[PersonaAgent]:
        """Get a persona agent by ID.
        
        Args:
            persona_id: ID of the persona to get.
            
        Returns:
            The persona agent, or None if it doesn't exist.
        """
        return self.agents.get(persona_id)
    
    def delete_persona(self, persona_id: str) -> bool:
        """Delete a persona agent.
        
        Args:
            persona_id: ID of the persona to delete.
            
        Returns:
            True if the persona was deleted, False if it doesn't exist.
        """
        if persona_id in self.agents:
            del self.agents[persona_id]
            self.logger.info(f"Deleted persona {persona_id}")
            return True
        return False
    
    def list_personas(self) -> List[str]:
        """List all persona IDs.
        
        Returns:
            List of persona IDs.
        """
        return list(self.agents.keys())
    
    def chat(self, persona_id: str, message: str) -> str:
        """Chat with a persona agent.
        
        Args:
            persona_id: ID of the persona to chat with.
            message: Message to send to the persona.
            
        Returns:
            The persona's response.
            
        Raises:
            ValueError: If the persona doesn't exist.
        """
        agent = self.get_persona(persona_id)
        if not agent:
            raise ValueError(f"Persona {persona_id} not found")
        
        return agent.chat(message)
    
    def save_persona(self, persona_id: str, file_path: str) -> None:
        """Save a persona to a file.
        
        Args:
            persona_id: ID of the persona to save.
            file_path: Path where to save the persona.
            
        Raises:
            ValueError: If the persona doesn't exist.
        """
        agent = self.get_persona(persona_id)
        if not agent:
            raise ValueError(f"Persona {persona_id} not found")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        agent.save_persona(file_path)
        self.logger.info(f"Saved persona {persona_id} to {file_path}")
    
    def load_persona_from_file(
        self,
        file_path: str,
        persona_id: Optional[str] = None,
        enable_mcp_tools: bool = False,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Load a persona from a saved file.
        
        Args:
            file_path: Path to the saved persona file.
            persona_id: Optional ID for the persona. If not provided, one will be generated.
            enable_mcp_tools: Whether to enable MCP tools for this persona.
            llm_config: Custom LLM configuration for this persona.
            
        Returns:
            The ID of the loaded persona.
        """
        # Load the persona
        agent = PersonaAgent.load_persona(file_path, llm_config)
        
        # Generate a persona ID if not provided
        if not persona_id:
            import re
            persona_id = re.sub(r'\W', '', agent.profile.name.replace(' ', '_')).lower()
        
        # Enable MCP tools if requested
        if enable_mcp_tools and self.mcp_config:
            try:
                agent.enable_mcp_tools(self.mcp_config)
                self.logger.info(f"Enabled MCP tools for persona {persona_id}")
            except Exception as e:
                self.logger.error(f"Failed to enable MCP tools for persona {persona_id}: {e}")
        
        # Store the agent
        self.agents[persona_id] = agent
        self.logger.info(f"Loaded persona {persona_id} from {file_path}")
        
        return persona_id
    
    def get_persona_info(self, persona_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a persona.
        
        Args:
            persona_id: ID of the persona to get information about.
            
        Returns:
            Dictionary with persona information, or None if the persona doesn't exist.
        """
        agent = self.get_persona(persona_id)
        if not agent:
            return None
        
        return {
            "id": persona_id,
            "name": agent.profile.name,
            "description": agent.profile.description,
            "tools": agent.tool_adapter.get_tool_configs(),
        }
    
    def get_persona_tools(self, persona_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get the tools available to a persona.
        
        Args:
            persona_id: ID of the persona to get tools for.
            
        Returns:
            List of tool configurations, or None if the persona doesn't exist.
        """
        agent = self.get_persona(persona_id)
        if not agent:
            return None
        
        return agent.tool_adapter.get_tool_configs()


# Create a global API instance for convenience
api = PersonaAgentAPI()


def create_persona(
    profile_path: Optional[str] = None,
    profile_data: Optional[Dict[str, Any]] = None,
    persona_id: Optional[str] = None,
    enable_mcp_tools: bool = False,
    llm_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a new persona agent.
    
    Args:
        profile_path: Path to a YAML or JSON file with persona configuration.
        profile_data: Dictionary containing persona configuration data.
        persona_id: Optional ID for the persona. If not provided, one will be generated.
        enable_mcp_tools: Whether to enable MCP tools for this persona.
        llm_config: Custom LLM configuration for this persona.
        
    Returns:
        The ID of the created persona.
    """
    return api.create_persona(
        profile_path=profile_path,
        profile_data=profile_data,
        persona_id=persona_id,
        enable_mcp_tools=enable_mcp_tools,
        llm_config=llm_config,
    )


def get_persona(persona_id: str) -> Optional[PersonaAgent]:
    """Get a persona agent by ID.
    
    Args:
        persona_id: ID of the persona to get.
        
    Returns:
        The persona agent, or None if it doesn't exist.
    """
    return api.get_persona(persona_id)


def delete_persona(persona_id: str) -> bool:
    """Delete a persona agent.
    
    Args:
        persona_id: ID of the persona to delete.
        
    Returns:
        True if the persona was deleted, False if it doesn't exist.
    """
    return api.delete_persona(persona_id)


def list_personas() -> List[str]:
    """List all persona IDs.
    
    Returns:
        List of persona IDs.
    """
    return api.list_personas()


def chat(persona_id: str, message: str) -> str:
    """Chat with a persona agent.
    
    Args:
        persona_id: ID of the persona to chat with.
        message: Message to send to the persona.
        
    Returns:
        The persona's response.
    """
    return api.chat(persona_id, message)


def save_persona(persona_id: str, file_path: str) -> None:
    """Save a persona to a file.
    
    Args:
        persona_id: ID of the persona to save.
        file_path: Path where to save the persona.
    """
    api.save_persona(persona_id, file_path)


def load_persona_from_file(
    file_path: str,
    persona_id: Optional[str] = None,
    enable_mcp_tools: bool = False,
    llm_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Load a persona from a saved file.
    
    Args:
        file_path: Path to the saved persona file.
        persona_id: Optional ID for the persona. If not provided, one will be generated.
        enable_mcp_tools: Whether to enable MCP tools for this persona.
        llm_config: Custom LLM configuration for this persona.
        
    Returns:
        The ID of the loaded persona.
    """
    return api.load_persona_from_file(
        file_path=file_path,
        persona_id=persona_id,
        enable_mcp_tools=enable_mcp_tools,
        llm_config=llm_config,
    )


def get_persona_info(persona_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a persona.
    
    Args:
        persona_id: ID of the persona to get information about.
        
    Returns:
        Dictionary with persona information, or None if the persona doesn't exist.
    """
    return api.get_persona_info(persona_id)


def get_persona_tools(persona_id: str) -> Optional[List[Dict[str, Any]]]:
    """Get the tools available to a persona.
    
    Args:
        persona_id: ID of the persona to get tools for.
        
    Returns:
        List of tool configurations, or None if the persona doesn't exist.
    """
    return api.get_persona_tools(persona_id)
