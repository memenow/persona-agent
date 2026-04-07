"""Command-line interface for the persona agent system.

Provides commands for starting the API server, managing personas,
and inspecting A2A agent cards.
"""

import argparse
import json
import os

import yaml

from persona_agent.api.config import load_config
from persona_agent.api.persona_manager import PersonaManager
from persona_agent.api.server import run_server


def start_api():
    """Start the API server."""
    config = load_config()
    print(f"Starting Persona Agent API server at {config.host}:{config.port}")
    print(f"  REST API: http://{config.host}:{config.port}{config.api_prefix}/")
    print(f"  A2A endpoint: http://{config.host}:{config.port}/a2a/")
    print(f"  Agent card: http://{config.host}:{config.port}/.well-known/agent.json")
    run_server(config)


def list_personas():
    """List all available personas."""
    config = load_config()
    persona_manager = PersonaManager(config.personas_dir)

    personas = persona_manager.list_personas()
    if not personas:
        print("No personas found.")
        return

    print(f"Found {len(personas)} personas:")
    for i, p in enumerate(personas, 1):
        print(f"  {i}. {p['id']}: {p['name']} - {p['description']}")


def import_persona(file_path: str):
    """Import a persona from a file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    config = load_config()
    persona_manager = PersonaManager(config.personas_dir)

    try:
        with open(file_path, encoding="utf-8") as f:
            if file_path.endswith(".json"):
                persona_data = json.load(f)
            elif file_path.endswith((".yaml", ".yml")):
                persona_data = yaml.safe_load(f)
            else:
                print("Error: File must be a JSON or YAML file.")
                return

        persona = persona_manager.add_persona(persona_data)
        saved_path = persona_manager.save_persona(persona, format="json")
        print(f"Successfully imported persona: {persona.name} (ID: {persona.id})")
        print(f"Saved to: {saved_path}")
    except Exception as e:
        print(f"Error importing persona: {e}")


def show_agent_card(persona_id: str | None = None):
    """Print the A2A agent card for a persona (or aggregate card)."""
    from persona_agent.a2a.agent_card import build_agent_card

    config = load_config()
    persona_manager = PersonaManager(config.personas_dir)

    if persona_id:
        p = persona_manager.get_persona(persona_id)
        if not p:
            print(f"Error: Persona '{persona_id}' not found.")
            return
        card = build_agent_card(
            persona_id=p.id,
            name=p.name,
            description=p.description,
            knowledge_domains=p.knowledge_domains or None,
        )
        print(json.dumps(card.model_dump(exclude_none=True), indent=2))
    else:
        # Show all persona cards
        personas = persona_manager.list_personas()
        if not personas:
            print("No personas found.")
            return
        for p_info in personas:
            p = persona_manager.get_persona(p_info["id"])
            if p:
                card = build_agent_card(
                    persona_id=p.id,
                    name=p.name,
                    description=p.description,
                    knowledge_domains=p.knowledge_domains or None,
                )
                print(f"--- {p.name} ({p.id}) ---")
                print(json.dumps(card.model_dump(exclude_none=True), indent=2))
                print()


def main():
    """Execute the command-line interface."""
    parser = argparse.ArgumentParser(description="Persona Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("api", help="Start the API server")
    subparsers.add_parser("list-personas", help="List all available personas")

    import_parser = subparsers.add_parser(
        "import-persona", help="Import a persona from a file"
    )
    import_parser.add_argument("file", help="Path to the persona file (JSON or YAML)")

    card_parser = subparsers.add_parser(
        "agent-card", help="Show A2A agent card for a persona"
    )
    card_parser.add_argument(
        "persona_id", nargs="?", default=None, help="Persona ID (omit for all)"
    )

    args = parser.parse_args()

    if args.command == "api":
        start_api()
    elif args.command == "list-personas":
        list_personas()
    elif args.command == "import-persona":
        import_persona(args.file)
    elif args.command == "agent-card":
        show_agent_card(args.persona_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
