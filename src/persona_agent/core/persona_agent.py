"""Persona Agent module.

This module defines the core PersonaAgent class that creates an AI agent
capable of simulating a specific persona using the AutoGen framework.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union
import inspect

# Import AutoGen 0.4
import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen_core import CancellationToken
from asyncio import CancelledError as AutoGenCancelledError

from persona_agent.core.persona_profile import PersonaProfile
from persona_agent.mcp.tool_adapter import MCPToolAdapter

# Define constants to avoid hardcoding
DEFAULT_CONFIG_PATH = "config"
LLM_CONFIG_FILENAME = "llm_config.json"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4000
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 30
DEFAULT_FALLBACK_PROMPT = "I'm sorry, I'm having trouble connecting right now. Could you please repeat your message or try again in a moment?"


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
        config_path: Optional[str] = None,
    ):
        """Initialize a new PersonaAgent.
        
        Args:
            profile: PersonaProfile containing persona information.
            llm_config: Configuration for the language model.
            system_message: Custom system message for the agent.
            tools: Optional list of tool configurations to register.
            tool_adapter: Optional MCP tool adapter for tool integration.
            config_path: Optional path to the configuration directory.
        """
        self.profile = profile
        self.tool_adapter = tool_adapter or MCPToolAdapter()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Creating PersonaAgent for {profile.name}")
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        
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
        # If LLM config is already provided with all necessary components, use it
        if llm_config is not None and ("client" in llm_config or "api_key" in llm_config):
            return llm_config
            
        # Load API key from config file if available
        api_key = None
        llm_config_path = os.path.join(self.config_path, LLM_CONFIG_FILENAME)
        model_config = {}
        
        try:
            if os.path.exists(llm_config_path):
                with open(llm_config_path, "r") as f:
                    config_data = json.load(f)
                    model_configs = config_data.get("model_configs", [])
                    if model_configs:
                        # Use the first model config
                        model_config = model_configs[0]
                        api_key = model_config.get("api_key")
                        # Get default model name from config or use a fallback
                        default_model = config_data.get("default_model", DEFAULT_MODEL)
        except (FileNotFoundError, json.JSONDecodeError, IndexError) as e:
            self.logger.warning(f"Error loading config from file: {str(e)}")

        if not api_key:
            # Try environment variable as fallback
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No API key found. Please provide an API key in llm_config.json or through the OPENAI_API_KEY environment variable.")
        
        # Create AutoGen 0.4 compatible config
        try:
            # Determine if we're using AutoGen 0.4 or legacy
            try:
                import autogen_ext
                from autogen_core.tools import Tool
                is_autogen_v4 = True
                self.logger.info("Detected AutoGen 0.4")
            except ImportError:
                is_autogen_v4 = False
                self.logger.info("Using legacy AutoGen")
            
            # Configure for AutoGen 0.4
            if is_autogen_v4:
                # For AutoGen 0.4, we need a config without a client
                # Extract parameters from the model config
                temperature = model_config.get("temperature", DEFAULT_TEMPERATURE)
                max_tokens = model_config.get("max_tokens", DEFAULT_MAX_TOKENS)
                model_name = model_config.get("model", default_model)
                
                # Create a config compatible with AutoGen 0.4
                result_config = {
                    "config_list": [
                        {
                            "model": model_name,
                            "api_key": api_key,
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                    ]
                }
                
                # If a base LLM config was provided, merge with our values
                if llm_config is not None:
                    # Don't overwrite our critical values
                    for key, value in llm_config.items():
                        if key not in ["config_list", "api_key"]:
                            result_config[key] = value
                
                self.logger.info(f"Configured LLM using AutoGen 0.4 format with model {model_name}")
                return result_config
            else:
                # Legacy AutoGen: Try to import from autogen_ext first as a fallback
                try:
                    from autogen_ext.models.openai import OpenAIChatCompletionClient
                    self.logger.info("Using OpenAIChatCompletionClient from autogen_ext.models.openai")
                    
                    # For legacy AutoGen, use a config dict instead of a client object
                    # This avoids JSON serialization issues
                    config = {
                        "model": model_config.get("model", default_model),
                        "temperature": model_config.get("temperature", 0.7),
                        "api_key": api_key,
                    }
                    
                    # Use the config dict in the llm_config
                    if llm_config is None:
                        return config
                    else:
                        # Merge with existing config
                        for key, value in config.items():
                            llm_config[key] = value
                        return llm_config
                        
                except ImportError:
                    # Fall back to standard autogen
                    try:
                        import autogen
                        from autogen.oai import OpenAIWrapper
                        self.logger.info("Using OpenAIWrapper from autogen.oai")
                        
                        # For legacy AutoGen, set environment variable and use wrapper
                        os.environ["OPENAI_API_KEY"] = api_key
                        
                        # Create a config with necessary parameters for OpenAIWrapper
                        if llm_config is None:
                            llm_config = {}
                        
                        llm_config["model"] = model_config.get("model", default_model)
                        llm_config["temperature"] = model_config.get("temperature", 0.7) 
                        llm_config["api_key"] = api_key
                        
                        return llm_config
                    except ImportError:
                        self.logger.error("Neither autogen_ext nor autogen is available. Cannot create OpenAI client.")
                        raise ImportError("No suitable OpenAI client implementation found")
                        
        except Exception as e:
            self.logger.error(f"Error configuring LLM: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _create_agent(self, llm_config: Dict[str, Any], system_message: str) -> None:
        """Create the AutoGen agent and user proxy.
        
        Args:
            llm_config: Configuration for the language model.
            system_message: System message for the agent.
        """
        # Convert our function_map to tools for the new API format
        tools = list(self.tool_adapter.get_function_map().values())
        
        # Create a valid Python identifier for the agent name
        valid_name = re.sub(r'\W', '', self.profile.name.replace(' ', '_'))
        
        # Check if llm_config contains a client object which is not JSON serializable
        # If it does, we need to convert it to a serializable configuration
        if "client" in llm_config and hasattr(llm_config["client"], "__class__"):
            self.logger.info("Converting client object to serializable configuration")
            # Extract basic parameters from the client for a serializable config
            client = llm_config.pop("client")
            
            # Get client attributes in a safe way
            def safe_get_attr(obj, attr, default=None):
                try:
                    return getattr(obj, attr, default)
                except Exception:
                    return default
            
            # Create a serializable configuration
            model = safe_get_attr(client, "model", "gpt-4o-mini")
            temperature = safe_get_attr(client, "temperature", 0.7)
            max_tokens = safe_get_attr(client, "max_tokens", 4000)
            
            # Create a new config without the client object
            base_url = safe_get_attr(client, "base_url", None)
            api_key = safe_get_attr(client, "api_key", os.environ.get("OPENAI_API_KEY"))
            
            # Use AutoGen 0.4 config_list format
            llm_config = {
                "config_list": [
                    {
                        "model": model,
                        "api_key": api_key,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                ]
            }
            
            if base_url:
                llm_config["config_list"][0]["base_url"] = base_url
        
        # Create the assistant agent
        self.logger.info(f"Creating assistant agent for {self.profile.name} using factory")
        self.agent = AssistantAgent(
            name=valid_name,  # Use the validated name
            system_message=system_message,
            llm_config=llm_config
        )
        if tools:
            self.agent.register_function_map({tool.__name__: tool for tool in tools})

        
        # Create the user proxy agent
        self.logger.info("Creating user proxy agent using factory")
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode = "NEVER",
            code_execution_config = False,
        )
        
        self.logger.info(f"Created PersonaAgent for {self.profile.name}")
    
    def _generate_system_message(self) -> str:
        """Generate system message for the agent.
        
        Uses information from the persona profile and configuration to generate a system message.
        
        Returns:
            str: The system message text.
        """
        # Start building the system message
        base_system_message = f"You are {self.profile.name}, an AI assistant."
        
        # Add personal background information
        if hasattr(self.profile, "personal_background") and self.profile.personal_background:
            # Check if personal background is a string or dictionary
            if isinstance(self.profile.personal_background, str):
                # Directly add string background
                base_system_message += f"\n\nPersonal Background:\n{self.profile.personal_background}"
            elif isinstance(self.profile.personal_background, dict):
                # Process dictionary format background
                base_system_message += "\n\nPersonal Background:"
                for key, value in self.profile.personal_background.items():
                    base_system_message += f"\n- {key}: {value}"
            else:
                # Unsupported background format, log warning
                self.logger.warning(f"Unsupported personal background format: {type(self.profile.personal_background)}")
        
        # Add language style information
        if hasattr(self.profile, "language_style") and self.profile.language_style:
            # Check if language style is a string or dictionary
            if isinstance(self.profile.language_style, str):
                # Directly add string language style
                base_system_message += f"\n\nLanguage Style:\n{self.profile.language_style}"
            elif isinstance(self.profile.language_style, dict):
                # Process dictionary format language style
                base_system_message += "\n\nLanguage Style:"
                for key, value in self.profile.language_style.items():
                    base_system_message += f"\n- {key}: {value}"
            else:
                # Unsupported language style format, log warning
                self.logger.warning(f"Unsupported language style format: {type(self.profile.language_style)}")
        
        # Add instructions information
        if hasattr(self.profile, "instructions") and self.profile.instructions:
            # Check if instructions are a string or dictionary
            if isinstance(self.profile.instructions, str):
                # Directly add string instructions
                base_system_message += f"\n\nInstructions:\n{self.profile.instructions}"
            elif isinstance(self.profile.instructions, dict):
                # Process dictionary format instructions
                base_system_message += "\n\nInstructions:"
                for key, value in self.profile.instructions.items():
                    base_system_message += f"\n- {key}: {value}"
            else:
                # Unsupported instructions format, log warning
                self.logger.warning(f"Unsupported instructions format: {type(self.profile.instructions)}")
        
        # Add tool usage instructions
        base_system_message += "\n\nTool Usage Instructions:\n"
        base_system_message += "You must use tools to provide up-to-date information. If the user's query involves current events, news, or any information that might change over time, "
        base_system_message += "you must use appropriate tools (such as brave_web_search) to get accurate information. "
        base_system_message += "Don't reply with placeholders or promises to search - actually use the available tools. "
        base_system_message += "If you don't use tools, your answers may be inaccurate or outdated."
        
        # Add other information
        if hasattr(self.profile, "knowledge") and self.profile.knowledge:
            # Check if knowledge is a string or dictionary
            if isinstance(self.profile.knowledge, str):
                # Directly add string knowledge
                base_system_message += f"\n\nKnowledge:\n{self.profile.knowledge}"
            elif isinstance(self.profile.knowledge, dict):
                # Process dictionary format knowledge
                base_system_message += "\n\nKnowledge:"
                for key, value in self.profile.knowledge.items():
                    base_system_message += f"\n- {key}: {value}"
            else:
                # Unsupported knowledge format, log warning
                self.logger.warning(f"Unsupported knowledge format: {type(self.profile.knowledge)}")
        
        return base_system_message
    
    def chat(self, message: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        """Synchronous chat method for interacting with the agent.
        
        This method sends a message to the agent and returns its response.
        It uses an event loop to run the asynchronous chat method.
        
        Args:
            message: The message to send to the agent.
            timeout: The maximum time in seconds to wait for a response.
            
        Returns:
            The agent's response as a string.
        """
        try:
            # Run the async _async_chat method synchronously
            loop = asyncio.get_event_loop()
            # If we're already in an event loop, run the coroutine directly
            if loop.is_running():
                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    # Run the async chat method with a timeout
                    return asyncio.run_coroutine_threadsafe(
                        self._async_chat(message), loop
                    ).result(timeout=timeout)
                finally:
                    # Clean up the new event loop
                    new_loop.close()
                    asyncio.set_event_loop(loop)
            else:
                # If we're not in an event loop, we can use run_until_complete
                return loop.run_until_complete(asyncio.wait_for(
                    self._async_chat(message), timeout
                ))
                
        except asyncio.TimeoutError:
            self.logger.warning(f"Chat timed out after {timeout} seconds")
            return self._generate_fallback_response(message)
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            return self._generate_fallback_response(message)
    
    async def _async_chat(self, message: str, request_id: str = None, cancellation_token = None) -> str:
        """Asynchronous version of chat for AutoGen 0.4 compatibility.
        
        This method handles chat requests asynchronously and supports cancellation tokens
        from AutoGen 0.4. It's used primarily by the HTTP API for websocket communication.
        
        Args:
            message: Message to send to the persona.
            request_id: Optional ID for tracking the request (for websocket updates).
            cancellation_token: Optional cancellation token from AutoGen 0.4.
            
        Returns:
            The persona's response as a string.
        """
        self.logger.info(f"Persona {self.profile.name}: Starting async chat with message: {message[:50]}...")
        
        # Check if the cancellation token is already cancelled
        if cancellation_token is not None and hasattr(cancellation_token, 'cancelled') and cancellation_token.cancelled:
            self.logger.warning("Cancellation token is already cancelled, returning fallback response")
            return self._generate_fallback_response(message)
        
        try:
            # For AutoGen 0.4, we need to use the async API
            if hasattr(self.user_proxy, "a_initiate_chat"):
                self.logger.info("Using a_initiate_chat for AutoGen 0.4")
                
                # Use a longer timeout for complex responses
                CHAT_TIMEOUT = 120  # 2 minutes
                
                try:
                    # Call a_initiate_chat directly without creating a separate task
                    # This avoids the issue with task cancellation
                    self.logger.info("Calling a_initiate_chat directly without creating a separate task")
                    
                    try:
                        # Call a_initiate_chat directly with timeout
                        # Fix counters before initiating chat
                        # This helps prevent issues with non-English text and counters
                        if hasattr(self.agent, "_consecutive_auto_reply_counter"):
                            if isinstance(self.agent._consecutive_auto_reply_counter, dict):
                                # If it's a dict, set the counter for this specific sender
                                self.agent._consecutive_auto_reply_counter[self.user_proxy] = 0
                            else:
                                # If it's an int, set it directly
                                self.agent._consecutive_auto_reply_counter = 0
                                
                        # Also, ensure _max_consecutive_auto_reply_dict is properly set if it exists
                        if hasattr(self.agent, "_max_consecutive_auto_reply_dict"):
                            if isinstance(self.agent._max_consecutive_auto_reply_dict, dict):
                                # Ensure the sender is in the dictionary
                                if self.user_proxy not in self.agent._max_consecutive_auto_reply_dict:
                                    self.agent._max_consecutive_auto_reply_dict[self.user_proxy] = 0
                                    
                        # Now initiate the chat
                        chat_result = await asyncio.wait_for(
                            self.user_proxy.a_initiate_chat(
                                self.agent, 
                                message=message, 
                                clear_history=False,
                                cancellation_token=cancellation_token
                            ),
                            timeout=CHAT_TIMEOUT
                        )
                        
                        self.logger.info(f"Chat completed with result type: {type(chat_result)}")
                        
                        # For AutoGen 0.4, we should get a ChatResult object
                        if hasattr(chat_result, "summary"):
                            return chat_result.summary
                        
                        # Try to get the response from chat history
                        try:
                            if self.agent in self.user_proxy.chat_messages:
                                for msg in reversed(self.user_proxy.chat_messages[self.agent]):
                                    if msg["role"] == "assistant":
                                        return msg["content"]
                        except Exception as e:
                            self.logger.warning(f"Error retrieving chat messages: {str(e)}")
                        
                        # If we don't have a result yet, generate a fallback
                        return self._generate_fallback_response(message)
                        
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Chat task timed out after {CHAT_TIMEOUT} seconds")
                        # Cancel the token if available
                        if cancellation_token and hasattr(cancellation_token, 'cancel'):
                            cancellation_token.cancel()
                            self.logger.info("Cancelled token due to timeout")
                        return self._generate_fallback_response(message)
                    except asyncio.CancelledError:
                        self.logger.warning("Chat operation was cancelled")
                        return self._generate_fallback_response(message)
                    
                except Exception as e:
                    self.logger.error(f"Error in a_initiate_chat: {str(e)}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    return self._generate_fallback_response(message)
            else:
                # Fall back to the synchronous version for legacy AutoGen
                self.logger.info("Using synchronous initiate_chat for chat")
                # Run the synchronous version in a thread pool to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    chat_result = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: self.user_proxy.initiate_chat(self.agent, message=message, clear_history=False)
                    )
                
                # Try to get summary from chat_result
                if hasattr(chat_result, "summary"):
                    return chat_result.summary
                
                # Otherwise check chat history
                if self.agent in self.user_proxy.chat_messages:
                    for msg in reversed(self.user_proxy.chat_messages[self.agent]):
                        if msg["role"] == "assistant":
                            return msg["content"]
                
                # Fallback if no response found
                self.logger.warning("No assistant message found in chat history")
                return self._generate_fallback_response(message)
        except asyncio.CancelledError:
            self.logger.error("Chat task was explicitly cancelled")
            return self._generate_fallback_response(message)
        except AutoGenCancelledError:
            self.logger.error("AutoGen cancelled error occurred")
            return self._generate_fallback_response(message)
        except Exception as e:
            self.logger.error(f"Error during async chat: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"An error occurred during the chat: {str(e)}"
    
    def _generate_fallback_response(self, message: str) -> str:
        """Generate a fallback response when the LLM is unavailable.
        
        If the LLM is unavailable or times out, this method generates a 
        fallback response that attempts to maintain the persona.
        
        Args:
            message: The user message that was being processed.
            
        Returns:
            A fallback message from the persona.
        """
        # Use a consistent fallback message instead of hardcoding it everywhere
        return DEFAULT_FALLBACK_PROMPT

    
    def enable_mcp_tools(self, mcp_config=None):
        """Enable MCP tools for this Persona"""
        import traceback
        try:
            from src.persona_agent.mcp import register_mcp_tools_for_persona
        
            # Try to get MCP configuration
            if not mcp_config:
                # Try to get global configuration from API
                from src.persona_agent.api import get_mcp_config
                mcp_config = get_mcp_config()
                if not mcp_config:
                    self.logger.warning("No MCP configuration provided, cannot enable MCP tools")
                    return
        
            self.logger.info(f"MCP configuration content: {json.dumps(mcp_config, indent=2)}")
            
            # Use generic MCP adapter directly
            self.logger.info(f"Enabling MCP tools for {self.profile.name}")
            
            # Call the API's tool registration function
            result = register_mcp_tools_for_persona(self, mcp_config)
            
            if result:
                self.logger.info(f"Successfully enabled MCP tools for {self.profile.name}")
            else:
                self.logger.warning(f"Failed to enable MCP tools for {self.profile.name}")
        
        except Exception as e:
            self.logger.error(f"Error enabling MCP tools: {str(e)}")
            self.logger.error(traceback.format_exc())
    
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

    def get_tool_list(self):
        """Retrieve the list of available tools."""
        if not hasattr(self, 'tool_adapter') or self.tool_adapter is None:
            self.logger.warning("Tool adapter not found, returning empty list")
            return []

        self.logger.info(f"Tool adapter type: {type(self.tool_adapter)}")
        self.logger.info(f"Tool adapter attributes: {dir(self.tool_adapter)}")

        has_function_map = hasattr(self.tool_adapter, 'function_map')
        self.logger.info(f"Tool adapter has function_map attribute: {has_function_map}")

        if has_function_map:
            function_map = self.tool_adapter.function_map
            self.logger.info(f"function_map type: {type(function_map)}")
            self.logger.info(f"function_map is empty: {not bool(function_map)}")
            if function_map:
                self.logger.info(f"function_map keys: {list(function_map.keys())}")

        has_tools_config = hasattr(self.tool_adapter, 'tools_config')
        self.logger.info(f"Tool adapter has tools_config attribute: {has_tools_config}")

        if has_tools_config:
            tools_config = self.tool_adapter.tools_config
            self.logger.info(f"tools_config type: {type(tools_config)}")
            self.logger.info(f"tools_config is empty: {not bool(tools_config)}")
            if tools_config:
                self.logger.info(f"tools_config keys: {list(tools_config.keys())}")

        has_tools = hasattr(self.tool_adapter, 'tools')
        self.logger.info(f"Tool adapter has tools attribute: {has_tools}")

        if has_tools:
            tools = self.tool_adapter.tools
            self.logger.info(f"tools type: {type(tools)}")
            self.logger.info(f"tools is empty: {not bool(tools)}")
            if tools:
                self.logger.info(f"tools keys: {list(tools.keys())}")

        if hasattr(self.tool_adapter, 'function_map') and self.tool_adapter.function_map:
            tool_count = len(self.tool_adapter.function_map)
            self.logger.info(f"Retrieved {tool_count} tools from tool adapter")

            tools = []
            for tool_id, _ in self.tool_adapter.function_map.items():
                if hasattr(self.tool_adapter, 'tools_config') and self.tool_adapter.tools_config:
                    tool_config = self.tool_adapter.tools_config.get(tool_id, {})
                elif hasattr(self.tool_adapter, 'tools') and self.tool_adapter.tools:
                    tool_config = self.tool_adapter.tools.get(tool_id, {})
                else:
                    tool_config = {}

                self.logger.info(f"Tool config {tool_id}: {tool_config}")

                tools.append({
                    "id": tool_id,
                    "name": tool_config.get("name", tool_id),
                    "description": tool_config.get("description", ""),
                    "requires_approval": tool_config.get("requires_approval", True)
                })
            return tools

        if hasattr(self.tool_adapter, 'tools') and self.tool_adapter.tools:
            tool_count = len(self.tool_adapter.tools)
            self.logger.info(f"Retrieved {tool_count} tools from tools attribute")

            tools = []
            for tool_id, tool_config in self.tool_adapter.tools.items():
                self.logger.info(f"Tool config {tool_id}: {tool_config}")

                tools.append({
                    "id": tool_id,
                    "name": tool_config.get("name", tool_id),
                    "description": tool_config.get("description", ""),
                    "requires_approval": tool_config.get("requires_approval", True)
                })
            return tools

        if hasattr(self.tool_adapter, 'tools_config') and self.tool_adapter.tools_config:
            tool_count = len(self.tool_adapter.tools_config)
            self.logger.info(f"Retrieved {tool_count} tools from tools_config attribute")

            tools = []
            for tool_id, tool_config in self.tool_adapter.tools_config.items():
                self.logger.info(f"Tool config {tool_id}: {tool_config}")

                tools.append({
                    "id": tool_id,
                    "name": tool_config.get("name", tool_id),
                    "description": tool_config.get("description", ""),
                    "requires_approval": tool_config.get("requires_approval", True)
                })
            return tools

        self.logger.warning("No tools found, returning empty list")
        return []

    def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with the given parameters.
        
        This method handles both synchronous and asynchronous tool executions.

        Args:
            tool_id: The unique identifier of the tool to execute.
            parameters: The parameters to pass to the tool.

        Returns:
            The result of the tool execution.
        """
        self.logger.info(f"Executing tool: {tool_id}")
        self.logger.info(f"Parameters: {json.dumps(parameters, indent=2, ensure_ascii=False)}")
        
        # Check if the tool exists
        if tool_id not in self.tool_adapter.function_map:
            # Try to find a simplified tool ID
            simplified_id = tool_id.split(".")[-1]
            if simplified_id in self.tool_adapter.function_map:
                self.logger.info(f"Found simplified tool ID: {simplified_id}")
                tool_id = simplified_id
            else:
                # Log all available tools
                available_tools = list(self.tool_adapter.function_map.keys())
                self.logger.error(f"Tool {tool_id} does not exist, available tools: {available_tools}")
                return {
                    "success": False,
                    "error": f"Tool {tool_id} not found",
                    "error_type": "ToolNotFoundError"
                }
        
        # Get the tool function
        tool_func = self.tool_adapter.function_map[tool_id]
        
        try:
            # Execute the tool
            self.logger.info(f"Starting to execute tool {tool_id}")
            
            # Check if the function is a coroutine
            if inspect.iscoroutinefunction(tool_func):
                self.logger.info(f"Tool {tool_id} is a coroutine function, will use asyncio to run")
                try:
                    # Use asyncio to run the coroutine function
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # If no event loop, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                if loop.is_running():
                    # If the event loop is already running, use asyncio.run_coroutine_threadsafe
                    concurrent_future = asyncio.run_coroutine_threadsafe(
                        tool_func(**parameters), loop
                    )
                    result = concurrent_future.result(timeout=60)  # 60 seconds timeout
                else:
                    # If the event loop is not running, use loop.run_until_complete
                    result = loop.run_until_complete(tool_func(**parameters))
            else:
                # Synchronous function directly call
                self.logger.info(f"Tool {tool_id} is a synchronous function, directly call")
                result = tool_func(**parameters)
                
            self.logger.info(f"Tool {tool_id} execution completed")
            
            # Record the result
            if isinstance(result, dict):
                self.logger.info(f"Result type: Dictionary, keys: {list(result.keys())}")
            else:
                self.logger.info(f"Result type: {type(result)}")
            
            # Standardize the result format
            if isinstance(result, dict) and "success" in result:
                return result
            else:
                return {
                    "success": True,
                    "result": result
                }
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_id}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "error_type": e.__class__.__name__
            }
