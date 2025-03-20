#!/usr/bin/env python3
"""Persona Agent API Server Entry Point.

This script serves as the entry point for starting the Persona Agent API server.
It configures the Python path and initializes the server using the configuration
loaded from the default configuration file.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.persona_agent.api.server import run_server
from src.persona_agent.api.config import load_config


def main():
    """Initialize and run the API server.
    
    This function loads the configuration and starts the FastAPI server
    with the appropriate host and port settings.
    """
    config = load_config()
    # Configuration can be set in environment variables or config files, no hardcoding needed
    print(f"Starting Persona Agent API server at {config.host}:{config.port}")
    run_server(config)


if __name__ == "__main__":
    main() 