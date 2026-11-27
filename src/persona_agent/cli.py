"""Command-line interface for running the server and inspecting personas."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from persona_agent.a2a.cards import build_agent_card, serialize_agent_card
from persona_agent.api.app import run_server
from persona_agent.config import load_settings
from persona_agent.personas.repository import PersonaRepository


def start_api() -> None:
    settings = load_settings()
    base_url = settings.resolve_public_base_url()
    print(f"Starting Persona Agent API server at {settings.host}:{settings.port}")
    print(f"  REST API: {base_url}{settings.api_prefix}/")
    print(f"  A2A endpoint: {base_url}/a2a/")
    print(f"  Agent card: {base_url}/.well-known/agent-card.json")
    run_server(settings)


def list_personas() -> None:
    settings = load_settings()
    repo = PersonaRepository(settings.personas_dir)
    personas = repo.list_personas()
    if not personas:
        print("No personas found.")
        return

    print(f"Found {len(personas)} personas:")
    for index, persona in enumerate(personas, 1):
        print(f"  {index}. {persona.id}: {persona.name} - {persona.description}")


def import_persona(file_path: str) -> None:
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: File {file_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    if path.suffix not in (".json", ".yaml", ".yml"):
        print("Error: File must be a JSON or YAML file.", file=sys.stderr)
        sys.exit(1)

    settings = load_settings()
    repo = PersonaRepository(settings.personas_dir)

    try:
        text = path.read_text(encoding="utf-8")
        persona_data = (
            json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        )
        if not isinstance(persona_data, dict):
            raise ValueError("Persona file must contain a JSON/YAML object.")

        persona = repo.add_persona(persona_data)
        saved_path = repo.save_persona(persona, fmt="json")
        print(f"Imported persona: {persona.name} (ID: {persona.id})")
        print(f"Saved to: {saved_path}")
    except Exception as exc:
        print(f"Error importing persona: {exc}", file=sys.stderr)
        sys.exit(1)


def show_agent_card(persona_id: str | None = None) -> None:
    settings = load_settings()
    repo = PersonaRepository(settings.personas_dir)
    base_url = settings.resolve_public_base_url()

    if persona_id:
        persona = repo.get_persona(persona_id)
        if persona is None:
            print(f"Error: Persona '{persona_id}' not found.", file=sys.stderr)
            sys.exit(1)
        card = build_agent_card(
            persona_id=persona.id,
            name=persona.name,
            description=persona.description,
            knowledge_domains=persona.knowledge_domains or None,
            base_url=base_url,
        )
        print(json.dumps(serialize_agent_card(card), indent=2))
        return

    personas = repo.list_personas()
    if not personas:
        print("No personas found.")
        return

    for persona in personas:
        card = build_agent_card(
            persona_id=persona.id,
            name=persona.name,
            description=persona.description,
            knowledge_domains=persona.knowledge_domains or None,
            base_url=base_url,
        )
        print(f"--- {persona.name} ({persona.id}) ---")
        print(json.dumps(serialize_agent_card(card), indent=2))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persona Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("api", help="Start the API server")
    subparsers.add_parser("list-personas", help="List available personas")

    import_parser = subparsers.add_parser(
        "import-persona", help="Import a JSON or YAML persona file"
    )
    import_parser.add_argument("file", help="Path to the persona file (JSON or YAML)")

    card_parser = subparsers.add_parser(
        "agent-card", help="Print A2A v1.0 agent card JSON"
    )
    card_parser.add_argument(
        "persona_id", nargs="?", default=None, help="Persona ID (omit to show all)"
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
