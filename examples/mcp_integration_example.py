"""MCP Integration Example

This example demonstrates how to use MCP tools with a persona agent, including:
1. Loading a persona from a YAML file
2. Configuring and enabling MCP tools
3. Interacting with the agent

The example shows both the recommended approach using the newer MCPToolAdapter
as well as a legacy approach for backward compatibility.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any

# Add the project root to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from persona_agent.core.persona_agent import PersonaAgent
from persona_agent.core.persona_profile import PersonaProfile
from persona_agent.mcp.tool_adapter import MCPToolAdapter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def call_mcp_tool_example(server_name: str, tool_name: str, arguments: Dict[str, Any]):
    """Example of calling an MCP tool directly.
    
    Args:
        server_name: Name of the MCP server.
        tool_name: Name of the tool.
        arguments: Tool arguments.
    """
    from persona_agent.mcp.mcp import use_mcp_tool
    
    logger.info(f"Calling {tool_name} on server {server_name} with args: {arguments}")
    
    try:
        # Call the tool
        result = await use_mcp_tool(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments
        )
        
        logger.info(f"Tool result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calling tool: {str(e)}")
        return {"error": str(e)}


def create_example_persona() -> PersonaProfile:
    """Create an example persona profile.
    
    Returns:
        A PersonaProfile instance for example use.
    """
    profile = PersonaProfile(
        name="Albert Einstein",
        description="The famous physicist and Nobel laureate",
        personal_background={
            "birth": "March 14, 1879, Ulm, Germany",
            "death": "April 18, 1955, Princeton, NJ, USA",
            "education": "PhD in Physics from University of Zurich",
            "notable_work": "Theory of Relativity, Photoelectric Effect",
        },
        language_style={
            "tone": "Thoughtful and philosophical",
            "complexity": "Uses analogies to explain complex concepts",
            "phrases": "God does not play dice with the universe",
        },
        knowledge_domains={
            "physics": [
                "Theory of Relativity",
                "Photoelectric Effect",
                "Brownian Motion",
                "Quantum Mechanics"
            ],
            "philosophy": [
                "Scientific Determinism",
                "Pacifism",
                "Humanism"
            ]
        },
        interaction_samples=[
            {
                "type": "scientific_explanation",
                "content": "When you sit with a nice girl for two hours, it seems like minutes; when you sit on a hot stove for a minute, it seems like hours. That's relativity."
            },
            {
                "type": "philosophical_quote",
                "content": "The most beautiful thing we can experience is the mysterious. It is the source of all true art and science."
            }
        ]
    )
    
    return profile


def load_mcp_config() -> Dict[str, Any]:
    """Load MCP configuration from JSON file.
    
    Returns:
        Dictionary containing MCP configuration.
    """
    try:
        # Try to load from config directory
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'mcp_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Return a sample configuration
            logger.warning(f"Config file not found at {config_path}, using sample config")
            return {
                "mcpServers": {
                    "memory-server": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-memory"],
                        "autoApprove": ["create_entities", "read_graph"]
                    },
                    "brave-search": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                        "autoApprove": ["brave_web_search"]
                    }
                }
            }
    except Exception as e:
        logger.error(f"Error loading MCP config: {str(e)}")
        return {"mcpServers": {}}


async def run_example():
    """Run the MCP integration example."""
    # Create a persona
    profile = create_example_persona()
    
    # Create an agent for the persona
    agent = PersonaAgent(profile=profile)
    
    # Load MCP configuration
    mcp_config = load_mcp_config()
    
    # Enable MCP tools (will fail if autogen-ext[mcp] is not installed)
    success = agent.enable_mcp_tools(mcp_config)
    
    if success:
        logger.info("Successfully enabled MCP tools")
        
        # Example of chatting with the agent
        # The agent will now have access to MCP tools and can use them in responses
        response = agent.chat("What can you tell me about your theory of relativity?")
        print("\nAgent Response:")
        print(response)
        
        # Example of following up with a question that might trigger tool use
        response = agent.chat("What are the latest breakthroughs in quantum physics?")
        print("\nAgent Response (may use search tools):")
        print(response)
    else:
        logger.warning("Failed to enable MCP tools - ensure autogen-ext[mcp] is installed")
        print("\nTo install required packages: pip install -U \"autogen-ext[mcp]\"")
        
        # We can still chat with the agent, but it won't have access to tools
        response = agent.chat("What can you tell me about your theory of relativity?")
        print("\nAgent Response (without tools):")
        print(response)
    
    # Example of direct tool usage (without going through an agent)
    print("\nDirect MCP Tool Usage Example:")
    try:
        # Try to use a search tool directly
        result = await call_mcp_tool_example(
            server_name="brave-search",
            tool_name="brave_web_search",
            arguments={"query": "Einstein theory of relativity recent studies", "count": 3}
        )
        
        if isinstance(result, dict) and result.get("success") is False:
            print(f"Search failed: {result.get('error', 'Unknown error')}")
        else:
            print("Search Results:")
            for item in result.get("results", []):
                print(f"- {item.get('title')}: {item.get('url')}")
    except Exception as e:
        print(f"Error performing direct tool call: {str(e)}")


if __name__ == "__main__":
    # Run the async example
    asyncio.run(run_example())
