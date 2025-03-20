"""Command-line interface for the persona agent system.

This module provides the command-line interface for interacting with the persona agent system,
including commands for starting the API server, managing personas, and utility functions.
"""

import argparse
import os
import sys
import json
import yaml

from src.persona_agent.api.config import ApiConfig, load_config
from src.persona_agent.api.server import run_server
from src.persona_agent.api.persona_manager import PersonaManager, Persona


def start_api():
    """Start the API server.
    
    This function loads the API configuration and starts the API server
    using the configured host and port settings.
    """
    # Load configuration
    config = load_config()
    print(f"Starting Persona Agent API server at {config.host}:{config.port}")
    
    # Run the server
    run_server(config)


def list_personas():
    """List all available personas.
    
    This function displays a list of all personas available in the system,
    showing their ID, name, and description.
    """
    config = load_config()
    persona_manager = PersonaManager(config.personas_dir)
    
    personas = persona_manager.list_personas()
    if not personas:
        print("No personas found.")
        return
    
    print(f"Found {len(personas)} personas:")
    for i, persona in enumerate(personas, 1):
        print(f"{i}. {persona['id']}: {persona['name']} - {persona['description']}")


def import_persona(file_path: str):
    """Import a persona from a file.
    
    This function imports a persona definition from a JSON or YAML file
    into the system's persona storage.
    
    Args:
        file_path: Path to the JSON or YAML file containing the persona definition.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    config = load_config()
    persona_manager = PersonaManager(config.personas_dir)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                persona_data = json.load(f)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                persona_data = yaml.safe_load(f)
            else:
                print("Error: File must be a JSON or YAML file.")
                return
        
        # Create the persona
        persona = persona_manager.add_persona(persona_data)
        
        # Save the persona to a file
        saved_path = persona_manager.save_persona(persona, format='json')
        
        print(f"Successfully imported persona: {persona.name} (ID: {persona.id})")
        print(f"Saved to: {saved_path}")
    
    except Exception as e:
        print(f"Error importing persona: {str(e)}")


def main():
    """Execute the command-line interface.
    
    This function parses command-line arguments and dispatches to the appropriate
    handler function based on the specified command.
    
    Commands:
        api: Start the API server
        list: List all available personas
        import: Import a persona from a file
    """
    parser = argparse.ArgumentParser(description="Persona Agent CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start the API server')
    
    # List personas command
    list_parser = subparsers.add_parser('list-personas', help='List all available personas')
    
    # Import persona command
    import_parser = subparsers.add_parser('import-persona', help='Import a persona from a file')
    import_parser.add_argument('file', help='Path to the persona file (JSON or YAML)')
    
    args = parser.parse_args()
    
    if args.command == 'api':
        start_api()
    elif args.command == 'list-personas':
        list_personas()
    elif args.command == 'import-persona':
        import_persona(args.file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 