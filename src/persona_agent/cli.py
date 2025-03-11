#!/usr/bin/env python3
"""Command-line interface for Persona Agent.

This module provides command-line utilities for running the persona agent
service, both as a Python API and as an HTTP server.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

from persona_agent import __version__
from persona_agent.api import (
    create_persona,
    chat,
    list_personas,
    load_persona_from_file,
    save_persona,
    get_persona_info,
    get_persona_tools,
    PersonaAgentAPI,
)
from persona_agent.http_api import run_server
from persona_agent.llm_config import load_config as load_llm_config


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging for the CLI.
    
    Args:
        level: The logging level to use.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Persona Agent CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"Persona Agent v{__version__}",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    parser.add_argument(
        "--llm-config",
        type=str,
        default=os.path.join("config", "llm_config.json"),
        help="Path to the LLM configuration file",
    )
    
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=os.path.join("config", "mcp_config.json"),
        help="Path to the MCP configuration file",
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser(
        "server",
        help="Run the HTTP API server",
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on",
    )
    
    # Create persona command
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new persona",
    )
    create_parser.add_argument(
        "profile",
        type=str,
        help="Path to the persona profile file (YAML or JSON)",
    )
    create_parser.add_argument(
        "--id",
        type=str,
        help="Unique identifier for the persona",
    )
    create_parser.add_argument(
        "--model",
        type=str,
        help="Name of the LLM model to use",
    )
    create_parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable MCP tools",
    )
    
    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Chat with a persona",
    )
    chat_parser.add_argument(
        "persona_id",
        type=str,
        help="ID of the persona to chat with",
    )
    chat_parser.add_argument(
        "message",
        type=str,
        help="Message to send to the persona",
    )
    
    # List personas command
    list_parser = subparsers.add_parser(
        "list",
        help="List all available personas",
    )
    
    # Save persona command
    save_parser = subparsers.add_parser(
        "save",
        help="Save a persona to a file",
    )
    save_parser.add_argument(
        "persona_id",
        type=str,
        help="ID of the persona to save",
    )
    save_parser.add_argument(
        "file_path",
        type=str,
        help="Path to save the persona to",
    )
    
    # Load persona command
    load_parser = subparsers.add_parser(
        "load",
        help="Load a persona from a file",
    )
    load_parser.add_argument(
        "file_path",
        type=str,
        help="Path to load the persona from",
    )
    load_parser.add_argument(
        "--id",
        type=str,
        help="ID to assign to the loaded persona",
    )
    load_parser.add_argument(
        "--model",
        type=str,
        help="Name of the LLM model to use",
    )
    load_parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable MCP tools",
    )
    
    return parser.parse_args()


def main() -> None:
    """Run the CLI application."""
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    # Load LLM configuration
    if os.path.exists(args.llm_config):
        try:
            load_llm_config(args.llm_config)
            logging.info(f"Loaded LLM configuration from {args.llm_config}")
        except Exception as e:
            logging.error(f"Failed to load LLM configuration: {str(e)}")
            sys.exit(1)
    else:
        logging.warning(f"LLM configuration file not found: {args.llm_config}")
    
    # Load MCP configuration
    if os.path.exists(args.mcp_config):
        try:
            # Create API instance to load MCP config
            api = PersonaAgentAPI()
            api.mcp_config = args.mcp_config
            logging.info(f"Loaded MCP configuration from {args.mcp_config}")
        except Exception as e:
            logging.error(f"Failed to load MCP configuration: {str(e)}")
            sys.exit(1)
    
    # Handle commands
    if args.command == "server":
        logging.info(f"Starting HTTP API server on {args.host}:{args.port}")
        run_server(host=args.host, port=args.port, debug=args.debug)
    
    elif args.command == "create":
        try:
            persona_id = create_persona(
                profile=args.profile,
                persona_id=args.id,
                model_name=args.model,
                enable_mcp_tools=not args.no_mcp,
            )
            print(f"Created persona: {persona_id}")
        except Exception as e:
            logging.error(f"Failed to create persona: {str(e)}")
            sys.exit(1)
    
    elif args.command == "chat":
        try:
            response = chat(args.persona_id, args.message)
            print(f"{args.persona_id}: {response}")
        except Exception as e:
            logging.error(f"Failed to chat with persona: {str(e)}")
            sys.exit(1)
    
    elif args.command == "list":
        try:
            personas = list_personas()
            if not personas:
                print("No personas available")
            else:
                print("Available personas:")
                for persona in personas:
                    print(f"  {persona['id']}: {persona['name']} - {persona['description']}")
        except Exception as e:
            logging.error(f"Failed to list personas: {str(e)}")
            sys.exit(1)
    
    elif args.command == "save":
        try:
            save_persona(args.persona_id, args.file_path)
            print(f"Saved persona {args.persona_id} to {args.file_path}")
        except Exception as e:
            logging.error(f"Failed to save persona: {str(e)}")
            sys.exit(1)
    
    elif args.command == "load":
        try:
            persona_id = load_persona_from_file(
                file_path=args.file_path,
                persona_id=args.id,
                llm_config={"model": args.model} if args.model else None,
                enable_mcp_tools=not args.no_mcp,
            )
            print(f"Loaded persona: {persona_id}")
        except Exception as e:
            logging.error(f"Failed to load persona: {str(e)}")
            sys.exit(1)
    
    else:
        logging.error("No command specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
