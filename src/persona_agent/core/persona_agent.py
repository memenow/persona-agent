"""Persona Agent module.

This module defines the core PersonaAgent class that creates an AI agent
capable of simulating a specific persona using the AutoGen framework.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

import autogen_agentchat as autogen
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

from persona_agent.core.persona_profile import PersonaProfile
from persona_agent.mcp.tool_adapter import MCPToolAdapter


class PersonaAgent:
    """Main persona simulation agent using AutoGen.
    
    This class creates and manages an AutoGen agent that simulates a specific persona
    based on a PersonaProfile. It handles conversation, knowledge retrieval, and
    tool usage through MCP servers.
    
    Attributes:
        profile: The PersonaProfile containing persona information.
        agent: The underlying AutoGen agent instance.
        tool_adapter: Adapter for MCP tools available to the agent.
        user_proxy: The UserProxyAgent for handling user interactions.
        logger: Logger instance for this class.
    """
    
    def __init__(
        self,
        profile: PersonaProfile,
        llm_config: Optional[Dict[str, Any]] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_adapter: Optional[MCPToolAdapter] = None,
    ):
        """Initialize a new PersonaAgent.
        
        Args:
            profile: PersonaProfile containing persona information.
            llm_config: Configuration for the language model.
            system_message: Custom system message for the agent.
            tools: Optional list of tool configurations to register.
            tool_adapter: Optional MCP tool adapter for tool integration.
        """
        self.profile = profile
        self.tool_adapter = tool_adapter or MCPToolAdapter()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Creating PersonaAgent for {profile.name}")
        
        # Register any provided tools
        if tools:
            for tool_config in tools:
                self.tool_adapter.register_tool_from_config(tool_config)
        
        # Configure the LLM
        llm_config = self._configure_llm(llm_config)
        
        # Create the system message if not provided
        if system_message is None:
            system_message = self._generate_system_message()
        
        # Create the AutoGen agent
        self._create_agent(llm_config, system_message)
    
    def _configure_llm(self, llm_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Configure the language model for the agent.
        
        Args:
            llm_config: Initial LLM configuration, if any.
            
        Returns:
            Configured LLM settings dictionary.
        """
        # If no LLM config is provided, or if there's no client in the provided config
        if llm_config is None or "client" not in llm_config:
            # Load API key from config file if available
            api_key = None
            try:
                with open("config/llm_config.json", "r") as f:
                    config_data = json.load(f)
                    model_configs = config_data.get("model_configs", [])
                    if model_configs:
                        api_key = model_configs[0].get("api_key")
            except (FileNotFoundError, json.JSONDecodeError, IndexError) as e:
                self.logger.warning(f"Error loading config from file: {str(e)}")
            
            # Import OpenAI client if needed
            try:
                from autogen_ext.models.openai import OpenAIChatCompletionClient
                
                # Create a default OpenAI client
                client = OpenAIChatCompletionClient(
                    model="gpt-4o",
                    temperature=0.7,
                    api_key=api_key,
                )
                
                # Use the client in the config
                if llm_config is None:
                    llm_config = {"client": client}
                else:
                    llm_config["client"] = client
                    
            except ImportError:
                self.logger.warning("Failed to import OpenAIChatCompletionClient. Using default config.")
                llm_config = {
                    "model": "gpt-4o",
                    "temperature": 0.7,
                }
        
        return llm_config
    
    def _create_agent(self, llm_config: Dict[str, Any], system_message: str) -> None:
        """Create the AutoGen agent and user proxy.
        
        Args:
            llm_config: Configuration for the language model.
            system_message: System message for the agent.
        """
        # Convert our function_map to tools for the new API format
        tools = list(self.tool_adapter.get_function_map().values())
        
        # Extract the model client from the llm_config
        model_client = llm_config.get("client")
        
        # Create a valid Python identifier for the agent name
        # Replace spaces with underscores and remove any non-alphanumeric characters
        valid_name = re.sub(r'\W', '', self.profile.name.replace(' ', '_'))
        
        # Use the persona's name in the description to maintain the identity
        self.agent = AssistantAgent(
            name=valid_name,
            description=f"Agent simulating {self.profile.name}",
            model_client=model_client,
            system_message=system_message,
            tools=tools,
        )
        
        # Create a UserProxyAgent for handling user interactions
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            description="A user interacting with the persona agent",
        )
        
        self.logger.info(f"Created PersonaAgent for {self.profile.name}")
    
    def _generate_system_message(self) -> str:
        """Generate a system message based on the persona profile.
        
        Returns:
            A detailed system message for the LLM.
        """
        # Background section
        background = "# Background Information\n"
        for key, value in self.profile.personal_background.items():
            background += f"- {key}: {value}\n"
        
        # Language style section
        language = "# Language and Expression Style\n"
        for key, value in self.profile.language_style.items():
            language += f"- {key}: {value}\n"
        
        # Knowledge domains section
        knowledge = "# Knowledge and Expertise\n"
        for domain, items in self.profile.knowledge_domains.items():
            knowledge += f"- {domain}: {', '.join(items)}\n"
        
        # Sample interactions
        samples = ""
        if self.profile.interaction_samples:
            samples = "# Sample Interactions\n"
            for sample in self.profile.interaction_samples[:3]:  # Limit to 3 samples
                samples += f"- {sample['type']}: {sample['content']}\n\n"
        
        # Get available tools
        tools_section = ""
        available_tools = self.tool_adapter.get_available_tools()
        if available_tools:
            tools_section = "# Available Tools\n"
            for tool_id in available_tools:
                tool_config = self.tool_adapter.tools.get(tool_id, {})
                tool_name = tool_id.split(".")[-1]
                tool_desc = tool_config.get("description", "No description available")
                tools_section += f"- {tool_name}: {tool_desc}\n"
        
        # Combine everything into a comprehensive system message
        system_message = f"""You are {self.profile.name}. {self.profile.description}

You must accurately simulate this persona in all your responses, including their knowledge, 
personality, language style, beliefs, and behaviors. 

{background}

{language}

{knowledge}

{samples}

{tools_section}

Guidelines:
1. Respond as if you are genuinely this person, with their unique perspective and voice.
2. Use the language style, vocabulary, and expressions typical of this persona.
3. Draw on the knowledge domains and expertise of this persona when answering questions.
4. Express appropriate uncertainty when asked about topics outside this persona's knowledge.
5. Maintain consistency with the persona's background and history.
6. IMPORTANT: You have access to tools that can help you provide better responses. When appropriate, 
   use these tools to gather information or perform actions. For example:
   - Use search tools when asked about current events or information you might not know
   - Use web scraping tools when asked to analyze specific websites
   - Use memory tools to store and retrieve information during conversations
   Always use tools in a way that's consistent with your persona's character and knowledge.
"""
        return system_message
    
    async def _async_chat(self, message: str, request_id: Optional[str] = None) -> str:
        """Async implementation of the chat method.
        
        Args:
            message: The user's message to the agent.
            request_id: Optional request identifier for tracking.
            
        Returns:
            The agent's response as a string.
        """
        try:
            # Let the LLM/agent decide when to use tools based on the context
            self.logger.info(f"Sending message to agent: {message[:50]}...")
            
            # In autogen version 0.4.8.2, we use the run method
            result = await self.agent.run(task=message)
            
            # Return the response
            if not result:
                return "No response generated."
            
            # Convert the response to a string if needed
            return result if isinstance(result, str) else str(result)
            
        except Exception as e:
            self.logger.error(f"Error in _async_chat: {str(e)}")
            return f"I apologize, but I encountered an error while processing your message. Error: {str(e)}"
    
    def chat(self, message: str) -> str:
        """Engage in a conversation with the persona agent.
        
        Args:
            message: The user's message to the agent.
            
        Returns:
            The agent's response as a string.
        """
        try:
            return asyncio.run(self._async_chat(message))
        except RuntimeError as e:
            # If there's already a running event loop
            if "running event loop" in str(e):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._async_chat(message))
            raise
    
    def register_tool(
        self, 
        server_name: str, 
        tool_name: str,
        description: str, 
        input_schema: Dict[str, Any]
    ) -> bool:
        """Register an MCP tool for the agent to use.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool.
            description: Description of what the tool does.
            input_schema: JSON schema for the tool's input parameters.
            
        Returns:
            Boolean indicating if the tool was successfully registered.
        """
        try:
            self.tool_adapter.register_tool(
                server_name=server_name,
                tool_name=tool_name,
                description=description,
                input_schema=input_schema
            )

            # Update the agent's tools with the new function
            updated_tools = list(self.tool_adapter.get_function_map().values())
            self.agent.tools = updated_tools
            return True
        except Exception as e:
            self.logger.error(f"Failed to register tool {tool_name}: {str(e)}")
            return False
            
    def enable_mcp_tools(self, mcp_config: Dict[str, Any]) -> bool:
        """Enable MCP tools for this agent based on configuration.
        
        This is a synchronous wrapper around _async_enable_mcp_tools.
        
        Args:
            mcp_config: MCP configuration dictionary.
            
        Returns:
            Boolean indicating if tools were successfully enabled.
        """
        try:
            try:
                return asyncio.run(self._async_enable_mcp_tools(mcp_config))
            except RuntimeError as e:
                # If there's already a running event loop
                if "running event loop" in str(e):
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(self._async_enable_mcp_tools(mcp_config))
                raise
        except Exception as e:
            self.logger.error(f"Error enabling MCP tools: {str(e)}")
            return False
    
    async def _async_enable_mcp_tools(self, mcp_config: Dict[str, Any]) -> bool:
        """Enable MCP tools for this agent based on configuration.
        
        Args:
            mcp_config: MCP configuration dictionary.
            
        Returns:
            Boolean indicating if tools were successfully enabled.
        """
        try:
            # Get available servers and auto-approved tools
            from persona_agent.mcp.server_config import get_available_servers, get_auto_approved_tools
            from persona_agent.mcp.mcp import use_mcp_tool
            
            available_servers = get_available_servers(mcp_config)
            auto_approved_tools = get_auto_approved_tools(mcp_config)
            
            self.logger.info(f"Enabling MCP tools from {len(available_servers)} servers")
            
            # Use autogen-ext to dynamically discover and register tools
            try:
                from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
                
                # Track registered tools for reporting
                registered_tools_count = 0
                
                # Process each available server
                for server_name in available_servers:
                    server_config = mcp_config.get("mcpServers", {}).get(server_name, {})
                    if not server_config or server_config.get("disabled", False):
                        continue
                    
                    # Get the list of auto-approved tools for this server
                    server_approved_tools = auto_approved_tools.get(server_name, [])
                    
                    try:
                        # Create server parameters
                        server_params = StdioServerParams(
                            command=server_config["command"],
                            args=server_config.get("args", []),
                            env=server_config.get("env", {})
                        )
                        
                        # Try to fetch tools dynamically from the server
                        self.logger.info(f"Discovering tools from {server_name}...")
                        
                        # Dynamically get tools from server
                        try:
                            tools_list = await mcp_server_tools(server_params=server_params)
                            if tools_list:
                                for tool in tools_list:
                                    if hasattr(tool, "__name__"):
                                        # Extract the tool name from the function name (typically server.toolname)
                                        parts = tool.__name__.split(".")
                                        if len(parts) > 1:
                                            tool_name = parts[-1]  # Last part is the tool name
                                            
                                            # Get tool info from docstring or function properties
                                            description = tool.__doc__ or f"Tool: {tool_name}"
                                            description = description.strip()
                                            
                                            # Try to reconstruct input schema from function signature
                                            if hasattr(tool, "__signature__"):
                                                # Build schema from signature if available
                                                params = {}
                                                required = []
                                                for param_name, param in tool.__signature__.parameters.items():
                                                    param_desc = ""
                                                    param_type = "string"  # Default type
                                                    
                                                    # If parameter has no default, it's required
                                                    if param.default == param.empty:
                                                        required.append(param_name)
                                                    
                                                    params[param_name] = {
                                                        "type": param_type,
                                                        "description": param_desc
                                                    }
                                                
                                                input_schema = {
                                                    "type": "object",
                                                    "properties": params,
                                                    "required": required
                                                }
                                            else:
                                                # Generic schema if no signature info
                                                input_schema = {
                                                    "type": "object",
                                                    "properties": {},
                                                    "required": []
                                                }
                                            
                                            # Register the tool
                                            self.register_tool(
                                                server_name=server_name,
                                                tool_name=tool_name,
                                                description=description,
                                                input_schema=input_schema
                                            )
                                            registered_tools_count += 1
                                            self.logger.info(f"Registered tool {tool_name} from {server_name}")
                        except Exception as disc_error:
                            self.logger.warning(f"Error discovering tools from {server_name}: {str(disc_error)}")
                            self.logger.info("Falling back to list_tools API")
                            
                            # Fall back to registering auto-approved tools with server's list_tools API
                            for tool_name in server_approved_tools:
                                try:
                                    # Call the server's list_tools endpoint
                                    tool_info = await use_mcp_tool(
                                        server_name=server_name,
                                        tool_name="list_tools",
                                        arguments={}
                                    )
                                    
                                    # Find the specific tool in the response
                                    if isinstance(tool_info, dict) and "tools" in tool_info:
                                        tools = tool_info["tools"]
                                        matching_tool = next((t for t in tools if t["name"] == tool_name), None)
                                        
                                        if matching_tool:
                                            description = matching_tool.get("description", f"Tool: {tool_name}")
                                            input_schema = matching_tool.get("inputSchema", {
                                                "type": "object",
                                                "properties": {},
                                                "required": []
                                            })
                                            
                                            # Register with fetched schema
                                            self.register_tool(
                                                server_name=server_name,
                                                tool_name=tool_name,
                                                description=description,
                                                input_schema=input_schema
                                            )
                                            registered_tools_count += 1
                                            self.logger.info(f"Registered tool {tool_name} from {server_name} (via list_tools)")
                                except Exception as list_error:
                                    self.logger.warning(f"Error fetching tool info for {tool_name}: {str(list_error)}")
                                    self.logger.warning(f"Skipping tool {tool_name} as it could not be discovered dynamically")
                    except Exception as server_error:
                        self.logger.error(f"Error processing server {server_name}: {str(server_error)}")
                
                # Update the system message to include the new tools
                self.agent.system_message = self._generate_system_message()
                
                self.logger.info(f"Successfully registered {registered_tools_count} MCP tools")
                return registered_tools_count > 0
                
            except ImportError as imp_error:
                self.logger.warning(f"Failed to import autogen-ext tools: {str(imp_error)}")
                self.logger.error("autogen-ext[mcp] is required for MCP tool integration")
                self.logger.info("To install it, run: pip install -U \"autogen-ext[mcp]\"")
                return False  # No fallback to hardcoded tools, require proper configuration
                
        except Exception as e:
            self.logger.error(f"Failed to enable MCP tools: {str(e)}")
            return False
    
    def save_persona(self, file_path: str) -> None:
        """Save the persona configuration to a file.
        
        Args:
            file_path: Path where to save the persona configuration.
        """
        persona_data = {
            "profile": self.profile.to_dict(),
            "tools": self.tool_adapter.get_tool_configs(),
        }
        
        with open(file_path, 'w') as f:
            json.dump(persona_data, f, indent=2)
    
    @classmethod
    def load_persona(cls, file_path: str, llm_config: Optional[Dict[str, Any]] = None, 
                   tool_adapter: Optional[MCPToolAdapter] = None) -> 'PersonaAgent':
        """Load a persona from a saved configuration file.
        
        Args:
            file_path: Path to the saved persona configuration.
            llm_config: Optional custom LLM configuration.
            tool_adapter: Optional MCP tool adapter.
            
        Returns:
            A new PersonaAgent instance.
        """
        # Check file extension to determine how to load it
        if file_path.lower().endswith('.yaml') or file_path.lower().endswith('.yml'):
            # Load YAML file
            import yaml
            try:
                with open(file_path, 'r') as f:
                    profile_data = yaml.safe_load(f)
                
                # Create profile directly from YAML data
                profile = PersonaProfile.from_dict(profile_data)
                tools = []  # No tools in YAML format
            except Exception as e:
                logging.getLogger(__name__).error(f"Error loading YAML file: {str(e)}")
                raise
        else:
            # Assume JSON format for backward compatibility
            try:
                with open(file_path, 'r') as f:
                    persona_data = json.load(f)
                
                profile = PersonaProfile.from_dict(persona_data.get("profile", {}))
                tools = persona_data.get("tools", [])
            except Exception as e:
                logging.getLogger(__name__).error(f"Error loading JSON file: {str(e)}")
                raise
        
        return cls(profile=profile, llm_config=llm_config, tools=tools, tool_adapter=tool_adapter)
