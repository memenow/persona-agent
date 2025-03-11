"""MCP server configuration module.

This module provides functions for working with MCP server configurations,
following the standard Model Context Protocol server configuration format.
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.expanduser('~/.persona_agent/mcp_config.json')

def load_mcp_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load MCP server configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default path.
        
    Returns:
        Dictionary containing MCP server configurations.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        json.JSONDecodeError: If the configuration file contains invalid JSON.
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    
    logger.info(f"Loading MCP configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return config
    except FileNotFoundError:
        logger.warning(f"MCP configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in MCP configuration file: {config_path}")
        raise


def save_mcp_config(config: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """Save MCP server configuration to a JSON file.
    
    Args:
        config: Dictionary containing MCP server configurations.
        config_path: Path to save the configuration file. If None, uses the default path.
        
    Raises:
        IOError: If there's an error writing to the file.
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    logger.info(f"Saving MCP configuration to {config_path}")
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving MCP configuration to {config_path}: {str(e)}")
        raise


def get_available_servers(config: Dict[str, Any]) -> List[str]:
    """Get list of available MCP servers from configuration.
    
    Args:
        config: Dictionary containing MCP server configurations.
        
    Returns:
        List of server names that are not disabled.
    """
    servers = []
    
    for server_name, server_config in config.get('mcpServers', {}).items():
        if not server_config.get('disabled', False):
            servers.append(server_name)
    
    return servers


def get_auto_approved_tools(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Get dictionary of auto-approved tools for each server.
    
    Args:
        config: Dictionary containing MCP server configurations.
        
    Returns:
        Dictionary mapping server names to lists of auto-approved tool names.
    """
    auto_approved = {}
    
    for server_name, server_config in config.get('mcpServers', {}).items():
        if not server_config.get('disabled', False):
            auto_approved[server_name] = server_config.get('autoApprove', [])
    
    return auto_approved
