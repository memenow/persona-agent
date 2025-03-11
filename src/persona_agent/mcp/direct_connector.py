"""Direct MCP connector implementation.

This module provides a standalone implementation for connecting to MCP servers
without relying on external dependencies like autogen-ext. It serves as a fallback
mechanism when the preferred implementation is unavailable.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class DirectMCPConnector:
    """A direct implementation of the MCP client connector.
    
    This class provides methods for connecting to MCP servers and executing
    tools without relying on external dependencies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize a new direct MCP connector.
        
        Args:
            config_path: Optional path to MCP configuration file.
        """
        self.config_path = config_path
        self.mcp_config = None
        self.server_processes = {}
        
        if config_path and os.path.exists(config_path):
            self._load_config()
    
    def _load_config(self):
        """Load the MCP configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                self.mcp_config = json.load(f)
                logger.info(f"Loaded MCP configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading MCP configuration: {str(e)}")
    
    async def start_server(self, server_name: str) -> bool:
        """Start an MCP server process.
        
        Args:
            server_name: Name of the MCP server to start.
            
        Returns:
            Boolean indicating if the server was started successfully.
        """
        if not self.mcp_config or "mcpServers" not in self.mcp_config:
            logger.error("No MCP configuration loaded")
            return False
        
        if server_name not in self.mcp_config["mcpServers"]:
            logger.error(f"Server {server_name} not found in MCP configuration")
            return False
        
        # Check if the server is already running
        if server_name in self.server_processes:
            proc = self.server_processes[server_name]
            if proc.poll() is None:  # Process is still running
                logger.info(f"Server {server_name} is already running")
                return True
            else:
                logger.info(f"Server {server_name} has exited, restarting")
                del self.server_processes[server_name]
        
        # Start the server
        try:
            server_config = self.mcp_config["mcpServers"][server_name]
            command = server_config["command"]
            args = server_config.get("args", [])
            env = {**os.environ, **server_config.get("env", {})}
            
            # Create a new subprocess for the server
            proc = subprocess.Popen(
                [command, *args],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            # Wait a moment to ensure the process starts
            await asyncio.sleep(1)
            
            # Check if the process is still running
            if proc.poll() is None:
                self.server_processes[server_name] = proc
                logger.info(f"Started MCP server {server_name}")
                return True
            else:
                stdout, stderr = proc.communicate()
                logger.error(f"Server {server_name} failed to start: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting MCP server {server_name}: {str(e)}")
            return False
    
    async def use_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """Execute a tool on an MCP server.
        
        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to execute.
            arguments: Dictionary of tool arguments.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            Exception: If there's an error executing the tool.
        """
        # Ensure the server is running
        if not await self.start_server(server_name):
            return {
                "success": False,
                "error": f"Failed to start MCP server {server_name}"
            }
        
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as input_file:
            json.dump({
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments
            }, input_file)
            input_path = input_file.name
        
        output_path = input_path + '.out'
        
        try:
            # Try to use mcp-cli if available
            try:
                logger.info(f"Calling MCP tool {tool_name} using mcp-cli")
                mcp_cli_process = subprocess.run(
                    ["mcp", "tool", input_path, "--output", output_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if mcp_cli_process.returncode == 0 and os.path.exists(output_path):
                    with open(output_path, 'r') as f:
                        result = json.load(f)
                    logger.info(f"MCP tool {tool_name} executed successfully")
                    return result
                else:
                    logger.warning(f"mcp-cli failed: {mcp_cli_process.stderr}")
                    raise RuntimeError(f"mcp-cli failed: {mcp_cli_process.stderr}")
            except Exception as mcp_cli_error:
                logger.warning(f"mcp-cli approach failed: {str(mcp_cli_error)}")
            
            # Try direct communication with a simple script
            try:
                logger.info(f"Attempting direct communication with server {server_name}")
                python_exe = sys.executable
                
                # Prepare a simple Python script to send the request
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as script_file:
                    script_content = f"""
import asyncio
import json
import sys
from mcp import use_mcp_tool

async def main():
    with open('{input_path}', 'r') as f:
        data = json.load(f)
    
    result = await use_mcp_tool(
        server_name=data['server_name'],
        tool_name=data['tool_name'],
        arguments=data['arguments']
    )
    
    with open('{output_path}', 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    asyncio.run(main())
"""
                    script_file.write(script_content)
                    script_path = script_file.name
                
                # Execute the script
                script_process = subprocess.run(
                    [python_exe, script_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if script_process.returncode == 0 and os.path.exists(output_path):
                    with open(output_path, 'r') as f:
                        result = json.load(f)
                    logger.info(f"MCP tool executed successfully using direct script")
                    return result
                else:
                    logger.warning(f"Direct script failed: {script_process.stderr}")
                    raise RuntimeError(f"Direct script failed: {script_process.stderr}")
                    
            except Exception as script_error:
                logger.warning(f"Direct script approach failed: {str(script_error)}")
            
            # All approaches failed
            logger.error(f"All approaches to execute MCP tool {tool_name} failed")
            return {
                "success": False,
                "error": f"Failed to execute MCP tool {tool_name}"
            }
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
                if 'script_path' in locals() and os.path.exists(script_path):
                    os.unlink(script_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")
    
    def stop_all_servers(self):
        """Stop all running MCP server processes."""
        for server_name, proc in list(self.server_processes.items()):
            try:
                if proc.poll() is None:  # Process is still running
                    proc.terminate()
                    logger.info(f"Stopped MCP server {server_name}")
                del self.server_processes[server_name]
            except Exception as e:
                logger.error(f"Error stopping MCP server {server_name}: {str(e)}")
    
    def __del__(self):
        """Clean up resources when the object is garbage collected."""
        self.stop_all_servers()


# Global connector instance
_connector = None

def get_connector() -> DirectMCPConnector:
    """Get or create the global MCP connector instance.
    
    Returns:
        The global DirectMCPConnector instance.
    """
    global _connector
    if _connector is None:
        # Try to find the config path
        config_path = None
        possible_paths = [
            os.path.join(os.getcwd(), 'config', 'mcp_config.json'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'mcp_config.json'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        _connector = DirectMCPConnector(config_path)
    
    return _connector

async def direct_use_mcp_tool(
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Any:
    """Use an MCP tool directly without dependencies.
    
    This is a standalone implementation that doesn't rely on external packages.
    
    Args:
        server_name: Name of the MCP server.
        tool_name: Name of the tool to use.
        arguments: Arguments for the tool.
        
    Returns:
        The result of the tool execution.
    """
    connector = get_connector()
    return await connector.use_tool(server_name, tool_name, arguments)
