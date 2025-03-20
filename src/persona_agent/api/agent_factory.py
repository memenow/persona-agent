"""Agent factory for creating and managing AutoGen agents.

This module provides a factory class for creating and managing AI agents powered by AutoGen,
with support for MCP (Model Context Protocol) tools integration.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# Third-party imports
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core import CancellationToken
    # Try importing from different paths
    try:
        from autogen_agentchat.config import config_list_from_json
        _NEED_CUSTOM_CONFIG_FUNCTION = False
    except ImportError:
        # If the config module is not found, we will implement this function ourselves
        _NEED_CUSTOM_CONFIG_FUNCTION = True
except ImportError:
    raise ImportError("Required AutoGen packages not found. Install with: pip install autogen-core autogen-agentchat autogen-ext")

# Local application imports
from src.persona_agent.api.persona_manager import Persona
from src.persona_agent.mcp.mcp_manager import register_mcp_tools_for_agent

# Configure logging
logger = logging.getLogger("agent_factory")
logging.basicConfig(level=logging.INFO)

# Constants
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MCP_CONFIG_FILENAME = "mcp_config.json"

class AgentSession:
    """Represents an agent session with conversation history.
    
    Attributes:
        id: Unique identifier for the session.
        agent_id: ID of the agent associated with this session.
        persona_id: ID of the persona used by the agent.
        agent: Instance of AssistantAgent used in this session.
        messages: List of messages exchanged in this session.
        created_at: Timestamp when the session was created.
        last_active: Timestamp of the last activity in this session.
    """
    
    def __init__(self, agent_id: str, persona_id: str, agent: AssistantAgent):
        """Initialize a new agent session.
        
        Args:
            agent_id: ID of the agent associated with this session.
            persona_id: ID of the persona used by the agent.
            agent: Instance of AssistantAgent used in this session.
        """
        self.id: str = str(uuid.uuid4())
        self.agent_id: str = agent_id
        self.persona_id: str = persona_id
        self.agent: AssistantAgent = agent
        self.messages: List[Dict[str, Any]] = []
        self.created_at: float = asyncio.get_event_loop().time()
        self.last_active: float = self.created_at
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: Role of the message sender (e.g., "user" or "assistant").
            content: Content of the message.
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": asyncio.get_event_loop().time()
        })
        self.last_active = asyncio.get_event_loop().time()
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation.
        
        Returns:
            List[Dict[str, Any]]: A list of all messages in the conversation.
        """
        return self.messages


class AgentFactory:
    """Factory for creating and managing AutoGen agents.
    
    This class provides methods for creating, retrieving, and managing agents and sessions.
    It handles the loading of configuration files, initialization of MCP services, and
    creation of agents with appropriate tools.
    
    Attributes:
        llm_config_path: Path to the LLM configuration file.
        agents: Dictionary mapping agent IDs to agent information.
        sessions: Dictionary mapping session IDs to AgentSession instances.
        llm_configs: Dictionary containing LLM configurations.
    """
    
    # Global MCP tools cache
    _mcp_tools_cache: Dict[str, List[Any]] = {}  # Service name -> tool list
    _mcp_services_initialized: bool = False
    
    def __init__(self, llm_config_path: Optional[str] = None):
        """Initialize the agent factory.
        
        Args:
            llm_config_path: Path to the LLM configuration file.
        """
        self.llm_config_path = llm_config_path
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, AgentSession] = {}
        self.llm_configs: Dict[str, Any] = {}
        self.load_llm_configs()
    
    def load_llm_configs(self) -> None:
        """Load LLM configurations from the specified file."""
        if not self.llm_config_path or not os.path.exists(self.llm_config_path):
            logger.warning(f"LLM configuration file not found: {self.llm_config_path}")
            return
        
        try:
            with open(self.llm_config_path, 'r', encoding='utf-8') as f:
                self.llm_configs = json.load(f)
                logger.info(f"Loaded LLM configurations from {self.llm_config_path}")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading LLM configurations: {e}")
    
    def _create_valid_identifier(self, name: str) -> str:
        """Create a valid Python identifier from a name.
        
        Args:
            name: Original name that may contain invalid characters.
            
        Returns:
            str: A valid Python identifier derived from the original name.
        """
        # Replace spaces and special characters with underscores
        valid_name = re.sub(r'\W', '_', name)
        
        # Ensure the name starts with a letter or underscore
        if not valid_name or not re.match(r'^[a-zA-Z_]', valid_name):
            valid_name = 'agent_' + valid_name
            
        return valid_name
    
    async def initialize_mcp_services(self) -> bool:
        """Initialize all MCP services, executing only once on first need.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        if AgentFactory._mcp_services_initialized:
            logger.info("MCP services already initialized, using cached tools")
            return True
            
        # Initialize the tool list
        AgentFactory._mcp_tools_cache = {}
        
        # Load MCP tools
        if not self.llm_config_path:
            logger.warning("LLM config path not set, cannot load MCP configuration")
            return False
            
        # Get MCP config path from the same directory as LLM config
        mcp_config_path = os.path.join(os.path.dirname(self.llm_config_path), "mcp_config.json")
        if not os.path.exists(mcp_config_path):
            logger.warning(f"MCP configuration file does not exist: {mcp_config_path}")
            return False
            
        # Load MCP tools
        mcp_config = self.config_list_from_json(mcp_config_path)
        if not mcp_config or "mcpServers" not in mcp_config:
            logger.warning(f"Invalid MCP configuration format or no mcpServers section: {mcp_config_path}")
            return False
            
        for service_name, service_config in mcp_config["mcpServers"].items():
            # Skip disabled services
            if not service_config.get("enabled", True):
                logger.info(f"Skipping disabled MCP service: {service_name}")
                continue
            
            logger.info(f"Initializing MCP service: {service_name}")
            
            # If already in cache, skip initialization
            if service_name in AgentFactory._mcp_tools_cache:
                logger.info(f"Using cached MCP service: {service_name}")
                continue
                
            try:
                from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
                
                # Create service parameters
                server_params = StdioServerParams(
                    command=service_config.get("command"),
                    args=service_config.get("args", []),
                    env=service_config.get("env", {})
                )
                
                # Load tools
                service_tools = await mcp_server_tools(server_params)
                if service_tools:
                    logger.info(f"Loaded {len(service_tools)} tools from service {service_name}: {[t.name for t in service_tools]}")
                    
                    # Cache tools for reuse
                    AgentFactory._mcp_tools_cache[service_name] = service_tools
            except Exception as e:
                logger.error(f"Error loading MCP service {service_name} tools: {str(e)}", exc_info=True)
                
        # Mark as initialized
        AgentFactory._mcp_services_initialized = True
        return True
    
    async def get_mcp_tools(self) -> List[Any]:
        """Get all available MCP tools, prioritizing cached tools.
        
        This method returns a list of all MCP tools available from registered services,
        initializing services if necessary.
        
        Returns:
            List[Any]: List of MCP tool objects.
        """
        if not AgentFactory._mcp_services_initialized:
            await self.initialize_mcp_services()
            
        # Collect all cached tools
        all_tools = []
        for service_tools in AgentFactory._mcp_tools_cache.values():
            all_tools.extend(service_tools)
            
        return all_tools
    
    async def create_agent(self, persona: Persona, model: Optional[str] = None) -> str:
        """Create a new agent for a persona.
        
        Args:
            persona: The persona to create an agent for.
            model: Optional model name to use for the agent. If not provided,
                the default model from the configuration will be used.
                  
        Returns:
            str: The ID of the created agent.
        """
        # Create a unique ID for the agent
        agent_id = str(uuid.uuid4())
        
        # Use provided model or default from config
        model = model or self.llm_configs.get("default_model", DEFAULT_MODEL)
        
        # Get API key and base URL from config or environment
        api_key = self.llm_configs.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
        api_base = self.llm_configs.get("api_base", os.environ.get("OPENAI_API_BASE", None))
        
        if not api_key:
            logger.warning("OpenAI API key not found in config or environment")
        
        # Create the model client
        model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            api_base=api_base
        )
        
        # Create a valid Python identifier for the agent name
        valid_name = self._create_valid_identifier(persona.name)
        
        # Get MCP tools
        tools = await self.get_mcp_tools()
        logger.info(f"Provided {len(tools)} tools to agent {valid_name}")
        
        # Create the agent
        agent = AssistantAgent(
            name=valid_name,
            system_message=persona.generate_system_prompt(),
            model_client=model_client,
            tools=tools,  # Provide tools at creation time
            reflect_on_tool_use=True  # Enable tool use reflection
        )
        
        # Store the original name
        agent.original_name = persona.name
        
        # Store agent info
        self.agents[agent_id] = {
            "id": agent_id,
            "persona_id": persona.id,
            "agent": agent,
            "created_at": asyncio.get_event_loop().time()
        }
        
        return agent_id
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve.
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing agent information, or None if not found.
        """
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing information about all available agents.
        """
        return [
            {
                "id": agent_id,
                "persona_id": info["persona_id"],
                "name": getattr(info["agent"], "original_name", info["agent"].name),
                "created_at": info["created_at"]
            }
            for agent_id, info in self.agents.items()
        ]
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID.
        
        This method also deletes any sessions using this agent.
        
        Args:
            agent_id: ID of the agent to delete.
            
        Returns:
            bool: True if the agent was found and deleted, False otherwise.
        """
        if agent_id not in self.agents:
            return False
        
        # Also delete any sessions using this agent
        session_ids_to_delete = [
            session_id for session_id, session in self.sessions.items()
            if session.agent_id == agent_id
        ]
        
        for session_id in session_ids_to_delete:
            del self.sessions[session_id]
        
        del self.agents[agent_id]
        return True
    
    def create_session(self, agent_id: str) -> Optional[str]:
        """Create a new conversation session with an agent.
        
        Args:
            agent_id: ID of the agent to create a session with.
            
        Returns:
            Optional[str]: ID of the created session, or None if the agent was not found.
        """
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            return None
        
        session = AgentSession(
            agent_id=agent_id,
            persona_id=agent_info["persona_id"],
            agent=agent_info["agent"]
        )
        
        self.sessions[session.id] = session
        return session.id
    
    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get a session by ID.
        
        Args:
            session_id: ID of the session to retrieve.
            
        Returns:
            Optional[AgentSession]: The session with the specified ID, or None if not found.
        """
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing information about all available sessions.
        """
        return [
            {
                "id": session.id,
                "agent_id": session.agent_id,
                "persona_id": session.persona_id,
                "message_count": len(session.messages),
                "created_at": session.created_at,
                "last_active": session.last_active
            }
            for session in self.sessions.values()
        ]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.
        
        Args:
            session_id: ID of the session to delete.
            
        Returns:
            bool: True if the session was found and deleted, False otherwise.
        """
        if session_id not in self.sessions:
            return False
        del self.sessions[session_id]
        return True
    
    async def send_message(self, session_id: str, message: str) -> Tuple[bool, Optional[str]]:
        """Send a message to an agent and get a response.
        
        Args:
            session_id: ID of the session to send the message to.
            message: The message to send.
            
        Returns:
            Tuple[bool, Optional[str]]: A tuple containing a success flag and the response message (or error message).
        """
        session = self.sessions.get(session_id)
        if not session:
            return False, "Session not found"
        
        try:
            # Add user message to session
            session.add_message("user", message)
            
            # Get a response from the agent
            result = await session.agent.run(
                task=message,
                cancellation_token=CancellationToken()
            )
            
            # Handle different result formats
            if isinstance(result, tuple) and len(result) >= 2:
                success = result[0]
                response = result[1]
            elif isinstance(result, dict):
                success = result.get("success", False)
                response = result.get("content", "")
            else:
                success = True
                response = str(result)
            
            if not response:
                return False, "Failed to get response from agent"
            
            # Add assistant response to session
            session.add_message("assistant", response)
            return True, response
                
        except Exception as e:
            logger.error(f"Exception in send_message: {str(e)}", exc_info=True)
            return False, f"Error: {str(e)}"


if _NEED_CUSTOM_CONFIG_FUNCTION:
    def config_list_from_json(config_path: str) -> List[Dict[str, Any]]:
        """Custom implementation of config_list_from_json for backward compatibility.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            List[Dict[str, Any]]: A list of configuration dictionaries.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return []
        
        # If config is a list, return as is
        if isinstance(configs, list):
            return configs
        
        # If config is a dict with "config_list" key, return that
        if isinstance(configs, dict) and "config_list" in configs:
            return configs["config_list"]
        
        # If config is just a dict, wrap it in a list
        if isinstance(configs, dict):
            return [configs]
        
        # Default empty list
        return []
