"""Agent factory module.

This module provides factory functions for creating different types of AutoGen agents,
supporting both AutoGen 0.2/0.3 and AutoGen 0.4 APIs.
"""

import importlib
import logging
import os
import re
import json
from typing import Any, Dict, List, Optional, Union, Callable

logger = logging.getLogger(__name__)

# Check which version of AutoGen is available
_has_autogen_v4 = False
try:
    # Try to import from AutoGen 0.4
    import autogen_core
    import autogen_agentchat
    _has_autogen_v4 = True
    logger.info("Using AutoGen 0.4 APIs")
except ImportError:
    # Fall back to AutoGen 0.2/0.3
    try:
        import autogen
        logger.info("Using AutoGen 0.2/0.3 APIs")
    except ImportError:
        logger.error("No version of AutoGen found. Please install autogen-core, autogen-agentchat, or autogen.")
        raise

# Check if MCP tools are available
_has_mcp_tools = False
try:
    # Try to import autogen 0.4 MCP tools
    import autogen_ext.tools.mcp
    from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
    _has_mcp_tools = True
except ImportError:
    _has_mcp_tools = False


def create_assistant_agent(
    name: str,
    system_message: str,
    llm_config: Dict[str, Any],
    tools: Optional[List[Any]] = None,
    description: Optional[str] = None,
    mcp_services: Optional[Dict[str, str]] = None,
    **kwargs
) -> Any:
    """Create an assistant agent based on the available AutoGen version.
    
    This function creates an assistant agent with the specified configuration,
    automatically detecting and using the appropriate AutoGen API version.
    
    Args:
        name: Name of the assistant agent.
        system_message: System message that defines the agent's behavior.
        llm_config: Configuration for the language model to use.
        tools: Optional list of tools available to the agent.
        description: Optional description of the agent's purpose.
        mcp_services: Optional dictionary mapping service names to MCP server URLs.
        **kwargs: Additional keyword arguments to pass to the agent constructor.
        
    Returns:
        An assistant agent instance compatible with the installed AutoGen version.
        
    Raises:
        ImportError: If no compatible AutoGen version is found.
    """
    try:
        # If AutoGen 0.4 is available, use it
        if _has_autogen_v4:
            logger.info(f"Creating AssistantAgent {name} with AutoGen 0.4")
            
            # Import the necessary classes
            from autogen_agentchat.agents import AssistantAgent
            
            # Extract the model client from llm_config if available
            model_client = llm_config.get("client")
            
            # Create a valid Python identifier for the agent name
            valid_name = re.sub(r'\W', '_', name.replace(' ', '_'))
            logger.info(f"Using valid identifier {valid_name} for agent {name}")
            
            # Create an AssistantAgent with the model client and tools
            agent = AssistantAgent(
                name=valid_name,
                system_message=system_message,
                model_client=model_client,
                tools=tools or [],
                description=description or f"An AI assistant named {name}",
                **kwargs
            )
            
            # Store the original name
            agent.original_name = name
            
            # If MCP service configuration is provided, register them
            if mcp_services:
                # Directly pass the service configuration
                mcp_manager = McpManager()
                mcp_manager.load_services_from_config(mcp_services)
            
            return agent
        
        # Otherwise, use AutoGen 0.2/0.3
        else:
            logger.info(f"Creating AssistantAgent {name} with AutoGen 0.2/0.3")
            
            # Import the necessary classes
            from autogen import AssistantAgent
            
            # Create an AutoGen 0.2/0.3 AssistantAgent
            agent = AssistantAgent(
                name=name,
                system_message=system_message,
                llm_config=llm_config,
                function_map={tool.__name__: tool for tool in (tools or []) if hasattr(tool, "__name__")},
                description=description or f"An AI assistant named {name}",
                **kwargs
            )
            
            # AutoGen 0.2/0.3 has limited MCP support, show a warning
            if mcp_services:
                logger.warning("MCP services configuration is only supported in AutoGen 0.4 or later")
            
            return agent
            
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise


def create_user_agent(
    name: str,
    human_input_mode: str = "NEVER",
    description: Optional[str] = None,
    **kwargs
) -> Any:
    """Create a user agent based on the available AutoGen version.
    
    This function creates a user agent with the specified configuration,
    automatically detecting and using the appropriate AutoGen API version.
    A user agent simulates a user or client in an agent conversation.
    
    Args:
        name: Name of the user agent.
        human_input_mode: Mode for human input handling, typically "NEVER" for automated agents.
        description: Optional description of the agent's purpose.
        **kwargs: Additional keyword arguments to pass to the agent constructor.
        
    Returns:
        A user agent instance compatible with the installed AutoGen version.
        
    Raises:
        ImportError: If no compatible AutoGen version is found.
    """
    try:
        # If AutoGen 0.4 is available, use it
        if _has_autogen_v4:
            # Import the necessary classes
            from autogen_agentchat.agents import UserAgent
            
            # Create a valid Python identifier for the agent name
            valid_name = re.sub(r'\W', '_', name.replace(' ', '_'))
            
            # Create a UserAgent
            agent = UserAgent(
                name=valid_name,
                human_input_mode=human_input_mode,
                description=description or f"A user named {name}",
                **kwargs
            )
            
            return agent
        
        # Otherwise, use AutoGen 0.2/0.3
        else:
            # Import the necessary classes
            from autogen import UserProxyAgent
            
            # Create a UserProxyAgent (equivalent to UserAgent in v0.4)
            agent = UserProxyAgent(
                name=name,
                human_input_mode=human_input_mode,
                description=description or f"A user named {name}",
                **kwargs
            )
            
            return agent
            
    except Exception as e:
        logger.error(f"Error creating user agent: {e}")
        raise


def create_user_proxy_agent(
    name: str,
    human_input_mode: str = "NEVER",
    code_execution_config: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Any]] = None,
    description: Optional[str] = None,
    **kwargs
) -> Any:
    """Create a user proxy agent based on the available AutoGen version.
    
    This function creates a user proxy agent with the specified configuration,
    automatically detecting and using the appropriate AutoGen API version.
    A user proxy agent can execute code and tools on behalf of a user.
    
    Args:
        name: Name of the user proxy agent.
        human_input_mode: Mode for human input handling, typically "NEVER" for automated agents.
        code_execution_config: Optional configuration for code execution capabilities.
        tools: Optional list of tools available to the agent.
        description: Optional description of the agent's purpose.
        **kwargs: Additional keyword arguments to pass to the agent constructor.
        
    Returns:
        A user proxy agent instance compatible with the installed AutoGen version.
        
    Raises:
        ImportError: If no compatible AutoGen version is found.
    """
    try:
        # If AutoGen 0.4 is available, use it
        if _has_autogen_v4:
            logger.info(f"Creating UserProxyAgent {name} with AutoGen 0.4")
            
            # Import the necessary classes
            from autogen_agentchat.agents import UserProxyAgent
            
            # Create a UserProxyAgent with minimal parameters to ensure compatibility
            # AutoGen 0.4 has a different API for UserProxyAgent
            logger.info("Creating UserProxyAgent with only essential parameters for AutoGen 0.4")
            
            # Only use the most basic parameters required for initialization
            agent = UserProxyAgent(
                name=name,
                description=description or f"A user proxy named {name}"
            )
            
            # If we have tools, add them after initialization if possible
            if tools and hasattr(agent, 'register_for_tools'):
                logger.info(f"Registering {len(tools)} tools with UserProxyAgent")
                for tool in tools:
                    try:
                        agent.register_for_tools([tool])
                    except Exception as tool_error:
                        logger.warning(f"Could not register tool: {str(tool_error)}")
            
            logger.info(f"Successfully created UserProxyAgent for AutoGen 0.4")
            
            return agent
        
        # Otherwise, use AutoGen 0.2/0.3
        else:
            logger.info(f"Creating UserProxyAgent {name} with AutoGen 0.2/0.3")
            
            # Import the necessary classes
            from autogen import UserProxyAgent
            
            # Create an AutoGen 0.2/0.3 UserProxyAgent
            agent = UserProxyAgent(
                name=name,
                human_input_mode=human_input_mode,
                code_execution_config=code_execution_config,
                function_map={tool.__name__: tool for tool in (tools or []) if hasattr(tool, "__name__")},
                description=description or f"A user proxy named {name}",
                **kwargs
            )
            
            return agent
    except Exception as e:
        logger.error(f"Error creating UserProxyAgent: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def get_cancellation_token() -> Any:
    """Create a cancellation token for async operations.
    
    This function creates a cancellation token that can be used to cancel
    ongoing async operations in AutoGen agents. It's compatible with the
    installed AutoGen version.
    
    Returns:
        A cancellation token instance compatible with the installed AutoGen version.
        Returns None if cancellation is not supported in the installed version.
    """
    if _has_autogen_v4:
        try:
            from autogen_core import CancellationToken
            return CancellationToken()
        except ImportError:
            logger.warning("CancellationToken not available in autogen_core")
            return None
    return None


def create_group_chat(agents: List[Any], messages: Optional[List[Any]] = None) -> Any:
    """Create a group chat with the specified agents.
    
    This function creates a group chat environment where multiple agents can
    interact with each other. It automatically detects and uses the appropriate
    AutoGen API version.
    
    Args:
        agents: List of agent instances to include in the group chat.
        messages: Optional list of initial messages for the chat.
        
    Returns:
        A group chat instance compatible with the installed AutoGen version.
        
    Raises:
        ImportError: If no compatible AutoGen version is found.
    """
    try:
        # If AutoGen 0.4 is available, use it
        if _has_autogen_v4:
            logger.info("Creating GroupChat with AutoGen 0.4")
            
            # Import the necessary classes
            from autogen_agentchat.group_chat import GroupChat
            
            # Create a GroupChat
            return GroupChat(
                agents=agents,
                messages=messages or []
            )
        
        # Otherwise, use AutoGen 0.2/0.3
        else:
            logger.info("Creating GroupChat with AutoGen 0.2/0.3")
            
            # Import the necessary classes
            from autogen.agentchat.groupchat import GroupChat
            
            # Create an AutoGen 0.2/0.3 GroupChat
            return GroupChat(
                agents=agents,
                messages=messages or []
            )
    except Exception as e:
        logger.error(f"Error creating GroupChat: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
