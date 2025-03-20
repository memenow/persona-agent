"""MCP service and tool manager for AutoGen 0.4 framework.

This module provides classes for managing Model Context Protocol (MCP) services and tools
that can be used with AutoGen 0.4 agents.
"""

import asyncio
import os
import json
import re
import sys
import subprocess
import inspect
import traceback
from typing import Dict, Any, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_manager")

# Variable to track if AutoGen MCP is available
HAS_AUTOGEN_MCP = False

# Try multiple import paths for autogen-ext MCP components
try:
    # First try with autogen_ext
    from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
    HAS_AUTOGEN_MCP = True
    logger.info("Successfully imported MCP tools from autogen_ext.tools.mcp")
except ImportError as e1:
    logger.warning(f"Failed to import from autogen_ext.tools.mcp: {e1}")
    
    try:
        # Try alternative import path (sometimes there's a version difference)
        import autogen_ext
        logger.info(f"Found autogen_ext at {autogen_ext.__file__}")
        
        # Try to find the MCP module in autogen_ext
        import pkgutil
        for loader, name, is_pkg in pkgutil.iter_modules([os.path.dirname(autogen_ext.__file__)]):
            logger.info(f"Found module in autogen_ext: {name}, is_package: {is_pkg}")
            
        # Try direct import with sys.path modification
        sys.path.append(os.path.dirname(os.path.dirname(autogen_ext.__file__)))
        from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
        HAS_AUTOGEN_MCP = True
        logger.info("Successfully imported MCP tools after path modification")
    except (ImportError, AttributeError) as e2:
        logger.error(f"Failed alternative import approach: {e2}")
        logger.error("AutoGen MCP extension not installed, MCP tools will not be available")
        logger.error("Please install: pip install 'autogen-ext[mcp]'")
        
        # Define placeholder classes to avoid syntax errors
        class StdioServerParams:
            pass
            
        async def mcp_server_tools(*args, **kwargs):
            logger.error("MCP tools not available. Please install: pip install 'autogen-ext[mcp]'")
            return []


class McpService:
    """MCP service class representing a single MCP service configuration and state.
    
    Attributes:
        name: Service name.
        service_type: Type of service (currently only stdio supported).
        command: Command to run for the service.
        args: Command line arguments for the service.
        env: Environment variables for the service.
        encoding: Character encoding for service I/O.
        description: Human-readable description of the service.
        enabled: Whether the service is enabled.
        connection_params: Connection parameters for the service.
        tools: List of tools loaded from the service.
    """
    
    def __init__(self, name: str, description: Optional[str] = None, 
                 command: str = None, args: List[str] = None, 
                 env: Dict[str, str] = None, encoding: str = "utf-8"):
        """Initialize a new MCP service.
        
        Args:
            name: Service name.
            description: Human-readable description of the service.
            command: Command to run for the service.
            args: Command line arguments for the service.
            env: Environment variables for the service.
            encoding: Character encoding for service I/O.
        """
        self.name = name
        self.service_type = "stdio"
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.encoding = encoding
        self.description = description or f"MCP service: {name}"
        self.enabled = True
        self.connection_params = None
        self.tools = []  # Store tools loaded from the service
        
    def is_available(self) -> bool:
        """Check if the service is enabled and available.
        
        Returns:
            True if the service is enabled and available, False otherwise.
        """
        if not self.enabled:
            return False
        
        # Verify command exists
        if not self.command:
            return False
            
        # Check if command path exists (if absolute path)
        if os.path.isabs(self.command) and not os.path.exists(self.command):
            logger.warning(f"Command path does not exist: {self.command}")
            return False
            
        return True
    
    def verify_command(self) -> bool:
        """Verify that the command can be executed.
        
        Returns:
            True if command can be executed, False otherwise.
        """
        if not self.is_available():
            return False
            
        # Check if command is an absolute path
        if os.path.isabs(self.command):
            return os.path.exists(self.command)
            
        # Check if command is in PATH
        try:
            subprocess.run(
                ["which", self.command], 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError:
            logger.warning(f"Command not found in PATH: {self.command}")
            return False
    
    def get_display_command(self) -> str:
        """Return formatted command string for display.
        
        Returns:
            String representation of the command and its arguments.
        """
        if not self.args:
            return self.command
        
        # Format command and arguments
        cmd_parts = [self.command] + self.args
        return " ".join(cmd_parts)


class McpManager:
    """MCP service and tool manager for AutoGen."""
    
    def __init__(self, auto_load=True):
        self.services = {}
        self.tools_by_service = {}
        self.all_tools = []
        self.auto_load = auto_load
        self.connect_timeout = 15.0  # Default connection timeout in seconds
        
    def add_stdio_service(self, name: str, command: str, args: List[str] = None, 
                          env: Dict[str, str] = None, description: Optional[str] = None) -> None:
        """Add a stdio service with command and arguments."""
        self.services[name] = McpService(
            name=name, 
            description=description, 
            command=command, 
            args=args or [], 
            env=env or {}
        )
    
    def load_config(self, config_path: str) -> bool:
        """Load MCP configuration from a YAML or JSON file."""
        if not os.path.exists(config_path):
            logger.error(f"MCP configuration file does not exist: {config_path}")
            return False
            
        try:
            # Choose parser based on file extension
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path}")
                return False
                
            if not config or not isinstance(config, dict):
                logger.error(f"Invalid MCP configuration format: {config_path}")
                return False
                
            # Clear existing services
            self.services = {}
            
            # Load service configurations
            services_section = "services"
            if services_section in config and isinstance(config[services_section], dict):
                for name, service_config in config[services_section].items():
                    if not isinstance(service_config, dict):
                        continue
                        
                    # Skip disabled services
                    if not service_config.get("enabled", True):
                        continue
                        
                    # Get service configuration
                    service_type = service_config.get("type", "stdio")
                    description = service_config.get("description", f"MCP service: {name}")
                    
                    # Only support stdio type
                    if service_type != "stdio":
                        logger.info(f"Skipping non-stdio service: {name}")
                        continue
                    
                    # Handle stdio server configuration
                    command = service_config.get("command", "")
                    args = service_config.get("args", [])
                    
                    # Skip if no command is provided
                    if not command:
                        logger.warning(f"No command provided for stdio service: {name}")
                        continue
                    
                    # Handle environment variables in command
                    if "${" in command:
                        for env_var in re.findall(r'\${([^}]+)}', command):
                            if "." not in env_var:  # Skip configuration references
                                env_value = os.environ.get(env_var, "")
                                command = command.replace(f"${{{env_var}}}", env_value)
                    
                    # Handle environment variables in arguments
                    for i, arg in enumerate(args):
                        if isinstance(arg, str) and "${" in arg:
                            for env_var in re.findall(r'\${([^}]+)}', arg):
                                if "." not in env_var:  # Skip configuration references
                                    env_value = os.environ.get(env_var, "")
                                    args[i] = arg.replace(f"${{{env_var}}}", env_value)
                    
                    # Store stdio service
                    self.add_stdio_service(
                        name=name,
                        command=command,
                        args=args,
                        env=service_config.get("env", {}),
                        description=description
                    )
                    logger.info(f"Added stdio service: {name}")
            
            # Handle mcpServers section (AutoGen 0.4 format)
            if "mcpServers" in config and isinstance(config["mcpServers"], dict):
                for name, server_config in config["mcpServers"].items():
                    # Only support stdio type (mcpServers default to stdio)
                    if server_config.get("disabled", False):
                        logger.info(f"Skipping disabled mcpServer: {name}")
                        continue
                        
                    command = server_config.get("command")
                    if not command:
                        logger.warning(f"No command provided for mcpServer: {name}")
                        continue
                    
                    self.add_stdio_service(
                        name=name,
                        command=command,
                        args=server_config.get("args", []),
                        env=server_config.get("env", {}),
                        description=server_config.get("description", f"MCP service: {name}")
                    )
                    
                    print(f"Added mcpServer: {name}")
                    # Verify command is executable
                    service = self.services.get(name)
                    if service and service.verify_command():
                        print(f"✓ Command verified: {command}")
                    else:
                        print(f"✗ Warning: Command might not be executable: {command}")
                        
            logger.info(f"Loaded {len(self.services)} MCP services from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading MCP configuration from {config_path}: {e}")
            traceback.print_exc()
            return False
    
    async def load_tools(self) -> bool:
        """Load tools from all configured and enabled MCP services."""
        if not HAS_AUTOGEN_MCP:
            logger.error("MCP tools not available. Please install: pip install 'autogen-ext[mcp]'")
            return False
            
        success = True
        self.tools_by_service = {}
        self.all_tools = []
        
        # Maximum attempts and delays for ensuring all tools are loaded
        max_retries = 3  # Increase retry attempts
        retry_delays = [1, 2, 3]  # Short delays
        
        # Load tools from each service
        for service_name, service in self.services.items():
            if not service.is_available():
                logger.info(f"Service {service_name} not available, skipping...")
                continue
                
            # Regular loading approach
            service_success = False
            
            for attempt in range(1, max_retries + 1):
                retry_delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                
                try:
                    # If this is not the first attempt, print retry information
                    if attempt > 1:
                        logger.info(f"Retrying {attempt}/{max_retries} service {service_name}...")
                    
                    # Create stdio server parameters
                    logger.info(f"Starting stdio MCP service {service_name}: {service.get_display_command()}")
                    
                    # Check if command exists
                    if os.path.isabs(service.command) and not os.path.exists(service.command):
                        logger.error(f"Command does not exist: {service.command}")
                        if attempt == max_retries:
                            break
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    # Add environment variables
                    env = service.env.copy()
                    if "NODE_PATH" not in env and os.environ.get("NODE_PATH"):
                        env["NODE_PATH"] = os.environ.get("NODE_PATH")
                    if "PATH" not in env and os.environ.get("PATH"):
                        env["PATH"] = os.environ.get("PATH")
                    
                    # Use longer timeout for initialization
                    timeout = 30.0
                    logger.info(f"Connecting to MCP service {service_name} (timeout: {timeout}s)...")
                    
                    params = StdioServerParams(
                        command=service.command,
                        args=service.args,
                        env=env
                    )
                    
                    # Save connection parameters
                    service.connection_params = params
                    
                    # Try to load tools - handle notifications
                    try:
                        logger.info(f"Loading tools from service {service_name}...")
                        tools = await asyncio.wait_for(mcp_server_tools(params), timeout=timeout)
                        
                        # Check if tools are list (expected) or exception/other
                        if not isinstance(tools, list):
                            logger.warning(f"Unexpected result from mcp_server_tools: {type(tools)}, {tools}")
                            if attempt < max_retries:
                                await asyncio.sleep(retry_delay)
                                continue
                            break
                            
                        logger.info(f"Service {service_name} returned {len(tools)} tools")
                    except asyncio.CancelledError:
                        raise  # Don't catch cancellation
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout loading tools from service {service_name}")
                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay)
                            continue
                        break
                    except Exception as e:
                        error_str = str(e)
                        # If the error contains "notification" or "message", it might be a JSON-RPC notification issue
                        if "notification" in error_str.lower() or "message" in error_str.lower() or "unhandled" in error_str.lower():
                            logger.warning(f"Possible notification handling error: {error_str}")
                            # For FireCrawl MCP, we can continue after initialization errors
                            if "firecrawl" in service_name.lower():
                                logger.info("Continuing despite FireCrawl notification error...")
                                # Wait a bit for the service to initialize
                                await asyncio.sleep(3)
                                # Try again directly with a second call to load tools
                                try:
                                    tools = await asyncio.wait_for(mcp_server_tools(params), timeout=timeout)
                                    if not isinstance(tools, list):
                                        logger.warning(f"Second attempt: Unexpected result from mcp_server_tools: {type(tools)}")
                                        if attempt < max_retries:
                                            await asyncio.sleep(retry_delay)
                                            continue
                                        break
                                except Exception as e2:
                                    logger.error(f"Second attempt error: {e2}")
                                    if attempt < max_retries:
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    break
                            else:
                                if attempt < max_retries:
                                    await asyncio.sleep(retry_delay)
                                    continue
                                break
                        else:
                            logger.error(f"Error loading tools from service {service_name}: {error_str}")
                            if attempt < max_retries:
                                await asyncio.sleep(retry_delay)
                                continue
                            break
                    
                    # Save tools
                    if tools:
                        # Add service name
                        for tool in tools:
                            setattr(tool, "service_name", service_name)
                        
                        # Save tools
                        service.tools = tools
                        self.tools_by_service[service_name] = tools
                        self.all_tools.extend(tools)
                        
                        tool_names = [tool.name for tool in tools]
                        logger.info(f"✓ Successfully loaded {len(tools)} tools from service {service_name}: {', '.join(tool_names)}")
                        service_success = True
                        break  # Successfully loaded, exit retry loop
                    else:
                        logger.warning(f"Service {service_name} did not return any tools")
                        if attempt < max_retries:
                            # Wait longer between retries for no tools
                            await asyncio.sleep(retry_delay * 2)
                            continue
                
                except Exception as e:
                    error_str = str(e)
                    # Catch all exceptions and record detailed information
                    logger.error(f"Unexpected error loading tools from service {service_name}: {error_str}")
                    
                    # Check if need to continue retrying
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to load tools from service {service_name} after {max_retries} attempts")
            
            if not service_success:
                success = False
        
        # Final statistics
        if self.all_tools:
            logger.info(f"Successfully loaded {len(self.all_tools)} MCP tools from {len(self.tools_by_service)} services")
            for service_name, tools in self.tools_by_service.items():
                tool_names = [tool.name for tool in tools]
                logger.info(f"  - {service_name} tools: {', '.join(tool_names)}")
        else:
            logger.error("Failed to load any MCP tools")
            success = False
            
        return success
    
    def get_all_tools(self) -> List[Any]:
        """Get all loaded MCP tools."""
        return self.all_tools
    
    def get_service_tools(self, service_name: str) -> List[Any]:
        """Get tools for a specific service."""
        return self.tools_by_service.get(service_name, [])
    
    def get_available_services(self) -> List[str]:
        """Get names of available services."""
        return list(self.services.keys())


async def register_mcp_tools_for_agent(agent, services_config: Dict[str, Dict[str, Any]]) -> bool:
    """Register MCP tools for an AutoGen agent using service configuration."""
    logger.info(f"Starting to register MCP tools for agent {getattr(agent, 'name', 'unknown')}")
    
    if not HAS_AUTOGEN_MCP:
        logger.error("AutoGen MCP extension not available. Please install: pip install 'autogen-ext[mcp]'")
        return False
    
    try:
        # Create MCP manager
        manager = McpManager()
        
        # Log services configuration
        logger.info(f"Services config: {json.dumps(services_config, indent=2)}")
        
        # Load services
        for name, config in services_config.items():
            logger.info(f"Adding service {name}: {config.get('command')} {' '.join(config.get('args', []))}")
            manager.add_stdio_service(
                name=name,
                command=config.get("command"),
                args=config.get("args", []),
                env=config.get("env", {}),
                description=config.get("description", f"MCP service: {name}")
            )
        
        # Load tools
        logger.info("Loading MCP tools...")
        success = await manager.load_tools()
        if not success:
            logger.error("Failed to load MCP tools")
            return False
        
        # Register tools
        tools = manager.get_all_tools()
        logger.info(f"Found {len(tools)} tools to register: {[tool.name for tool in tools]}")
        
        if tools and hasattr(agent, "register_tools"):
            try:
                # Ensure each tool has correct attributes
                for tool in tools:
                    if hasattr(tool, "name") and tool.name and tool.name.startswith("firecrawl_"):
                        logger.info(f"Found firecrawl tool: {tool.name}")
                        # Make sure the tool has necessary attributes
                        if not hasattr(tool, "description") or not tool.description:
                            tool.description = f"FireCrawl tool: {tool.name}"
                
                # Add autoApprove configuration if available
                for service_name, service_config in services_config.items():
                    if "autoApprove" in service_config and isinstance(service_config["autoApprove"], list):
                        auto_approve_tools = service_config["autoApprove"]
                        logger.info(f"Found autoApprove configuration for service {service_name}: {auto_approve_tools}")
                        
                        # Set auto_approve attribute on tools
                        for tool in tools:
                            if tool.name in auto_approve_tools:
                                logger.info(f"Setting auto_approve=True for tool {tool.name}")
                                setattr(tool, "auto_approve", True)
                
                # Register tools with the agent
                agent.register_tools(tools)
                logger.info(f"Registered {len(tools)} MCP tools for agent {agent.name}")
                return True
            except Exception as e:
                logger.error(f"Error registering tools for agent: {e}")
                logger.error(f"Detailed error: {traceback.format_exc()}")
                return False
        else:
            if not tools:
                logger.error("No tools found to register")
            if not hasattr(agent, "register_tools"):
                logger.error(f"Agent {getattr(agent, 'name', 'unknown')} does not have register_tools method")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in register_mcp_tools_for_agent: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return False


def sync_register_mcp_tools_for_agent(agent, services_config: Dict[str, Dict[str, Any]]) -> bool:
    """Synchronous version of register_mcp_tools_for_agent."""
    return asyncio.run(register_mcp_tools_for_agent(agent, services_config)) 