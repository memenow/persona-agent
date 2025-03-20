"""Dependencies for FastAPI dependency injection.

This module provides dependency functions for FastAPI's dependency injection system,
making it easy to access shared resources throughout the API server.
"""

from fastapi import Depends
from functools import lru_cache
from typing import Dict, Any

from src.persona_agent.api.config import ApiConfig, load_config
from src.persona_agent.api.persona_manager import PersonaManager
from src.persona_agent.api.agent_factory import AgentFactory


@lru_cache()
def get_config() -> ApiConfig:
    """Get the API configuration singleton.
    
    This function loads the API configuration once and caches it for future use,
    ensuring configuration is only loaded once during application lifetime.
    
    Returns:
        The API configuration instance.
    """
    return load_config()


# Maintain original function for backward compatibility
@lru_cache()
def get_api_config() -> ApiConfig:
    """Get the API configuration singleton (legacy name).
    
    This function exists for backward compatibility and delegates to get_config().
    
    Returns:
        The API configuration instance.
    """
    return load_config()


def get_persona_manager(config: ApiConfig = Depends(get_config)) -> PersonaManager:
    """Get the persona manager instance.
    
    This function creates a PersonaManager instance using the configured personas directory.
    
    Args:
        config: The API configuration with personas directory information.
        
    Returns:
        A PersonaManager instance for managing persona profiles.
    """
    return PersonaManager(config.personas_dir)


def get_agent_factory(config: ApiConfig = Depends(get_config)) -> AgentFactory:
    """Get the agent factory instance.
    
    This function creates an AgentFactory instance using the configured MCP and LLM settings.
    
    Args:
        config: The API configuration with MCP and LLM configuration paths.
        
    Returns:
        An AgentFactory instance for creating persona agents.
    """
    return AgentFactory(
        mcp_config_path=config.mcp_config_path,
        llm_config_path=config.llm_config_path
    ) 