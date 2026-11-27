"""Filesystem-backed repository for loading, saving, and deleting personas."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .model import PERSONA_ID_PATTERN, Persona

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = (".json", ".yaml", ".yml")


class PersonaPersistenceError(RuntimeError):
    """Raised when persisted persona files cannot be updated or removed."""


class PersonaRepository:
    """In-memory persona store synchronized with JSON/YAML files on disk."""

    def __init__(self, personas_dir: str | Path) -> None:
        self.personas_dir = Path(personas_dir)
        self.personas: dict[str, Persona] = {}
        self._persona_files: dict[str, set[Path]] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.personas_dir.exists():
            self.personas_dir.mkdir(parents=True, exist_ok=True)
            return

        for file_path in sorted(self.personas_dir.iterdir()):
            if not file_path.is_file() or file_path.suffix not in _SUPPORTED_EXTENSIONS:
                continue
            persona = self._load_file(file_path)
            if persona is not None:
                self.personas[persona.id] = persona
                self._persona_files.setdefault(persona.id, set()).add(file_path)

    def _load_file(self, file_path: Path) -> Persona | None:
        try:
            text = file_path.read_text(encoding="utf-8")
            data = (
                json.loads(text)
                if file_path.suffix == ".json"
                else yaml.safe_load(text)
            )
            if not isinstance(data, dict):
                logger.warning("Skipping %s: expected a mapping", file_path)
                return None

            normalized_id = self._resolve_persona_id(data, file_path)
            if normalized_id is None:
                return None
            data["id"] = normalized_id
            return Persona.model_validate(data)
        except Exception as exc:
            logger.warning("Failed to load persona file %s: %s", file_path, exc)
            return None

    @staticmethod
    def _resolve_persona_id(data: dict[str, Any], file_path: Path) -> str | None:
        """Derive a pattern-compliant persona ID from file content or filename stem."""
        raw = data.get("id")
        if isinstance(raw, str) and re.fullmatch(PERSONA_ID_PATTERN, raw):
            return raw

        if isinstance(raw, str) and raw:
            sanitized = re.sub(r"[^a-z0-9_-]", "-", raw.lower()).strip("-")[:64]
            if sanitized:
                logger.warning(
                    "Persona id %r in %s does not match %s; normalized to %r",
                    raw,
                    file_path,
                    PERSONA_ID_PATTERN,
                    sanitized,
                )
                return sanitized

        sanitized = re.sub(r"[^a-z0-9_-]", "-", file_path.stem.lower()).strip("-")[:64]
        if sanitized:
            return sanitized

        logger.warning("Skipping persona file %s: cannot derive a valid id", file_path)
        return None

    def get_persona(self, persona_id: str) -> Persona | None:
        return self.personas.get(persona_id)

    def list_personas(self) -> list[Persona]:
        return list(self.personas.values())

    def add_persona(self, data: dict[str, Any]) -> Persona:
        persona = Persona.model_validate(data)
        self.personas[persona.id] = persona
        return persona

    def update_persona(
        self, persona_id: str, updates: dict[str, Any]
    ) -> Persona | None:
        """Validate updates against a copy before replacing the stored persona."""
        existing = self.personas.get(persona_id)
        if existing is None:
            return None

        merged = existing.model_dump()
        for key, value in updates.items():
            if value is not None:
                merged[key] = value
        merged["id"] = persona_id

        updated = Persona.model_validate(merged)
        self.personas[persona_id] = updated
        return updated

    def delete_persona(self, persona_id: str) -> bool:
        if persona_id not in self.personas:
            return False

        try:
            self._delete_persona_files(persona_id)
        except OSError as exc:
            logger.exception("Failed deleting persona files for %s", persona_id)
            raise PersonaPersistenceError(
                f"Failed deleting persona files for {persona_id}"
            ) from exc

        del self.personas[persona_id]
        return True

    def _delete_persona_files(self, persona_id: str) -> None:
        root = self.personas_dir.resolve()
        candidates = set(self._persona_files.get(persona_id, set()))

        for file_path in candidates:
            resolved = file_path.resolve()
            if not resolved.is_relative_to(root):
                logger.warning(
                    "Skipping persona file outside personas_dir: %s", file_path
                )
                continue
            if resolved.is_file():
                resolved.unlink()

        self._persona_files.pop(persona_id, None)

    def save_persona(self, persona: Persona, fmt: str = "json") -> Path:
        """Persist a persona as JSON or YAML under personas_dir."""
        if fmt not in ("json", "yaml"):
            raise ValueError("Format must be 'json' or 'yaml'")

        # Re-validate ID before filesystem access in case model_construct bypassed Pydantic.
        persona_id = persona.id
        if (
            not re.fullmatch(PERSONA_ID_PATTERN, persona_id)
            or ".." in persona_id
            or "/" in persona_id
            or "\\" in persona_id
            or "\x00" in persona_id
        ):
            raise ValueError("Invalid persona id")

        self.personas_dir.mkdir(parents=True, exist_ok=True)
        root = self.personas_dir.resolve()
        file_path = self.personas_dir / f"{persona_id}.{fmt}"
        if not file_path.resolve().is_relative_to(root):
            raise ValueError("Invalid persona id")

        payload = persona.model_dump()
        content = (
            json.dumps(payload, indent=2)
            if fmt == "json"
            else yaml.safe_dump(payload, default_flow_style=False)
        )
        file_path.write_text(content, encoding="utf-8")

        self._persona_files.setdefault(persona.id, set()).add(file_path)
        return file_path
