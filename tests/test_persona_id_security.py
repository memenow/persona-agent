"""Regression tests for the persona id constraints introduced as a fix for
the path-traversal vulnerability in ``PersonaManager.save_persona``."""

import pytest
from pydantic import ValidationError

from persona_agent.api.persona_manager import Persona, PersonaManager


def test_persona_id_rejects_path_traversal() -> None:
    with pytest.raises(ValidationError):
        Persona(id="../../etc/passwd", name="X")


def test_persona_id_rejects_uppercase() -> None:
    with pytest.raises(ValidationError):
        Persona(id="MixedCase", name="X")


def test_persona_id_rejects_separator() -> None:
    with pytest.raises(ValidationError):
        Persona(id="foo/bar", name="X")


def test_persona_id_rejects_null_byte() -> None:
    with pytest.raises(ValidationError):
        Persona(id="foo\x00bar", name="X")


def test_persona_id_rejects_overlong_value() -> None:
    with pytest.raises(ValidationError):
        Persona(id="a" * 65, name="X")


def test_persona_id_accepts_uuid() -> None:
    Persona(id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", name="X")


def test_persona_id_accepts_canonical_slug() -> None:
    Persona(id="trump_2024-1", name="X")


def test_save_persona_rejects_invalid_id_at_fs_layer(tmp_path) -> None:
    """Defense-in-depth: even if a Persona is constructed without going through
    the regex (e.g., via ``model_construct``), ``save_persona`` must refuse to
    write to the filesystem."""
    pm = PersonaManager(str(tmp_path))
    bypassed = Persona.model_construct(id="../escape", name="X")
    with pytest.raises(ValueError, match="Invalid persona id"):
        pm.save_persona(bypassed)


def test_load_persona_file_normalizes_legacy_id(tmp_path) -> None:
    """Existing on-disk files with non-conforming ids should be normalized
    rather than silently skipped."""
    file_path = tmp_path / "legacy.yaml"
    file_path.write_text(
        "id: Donald.Trump\nname: Donald Trump\ndescription: legacy file\n",
        encoding="utf-8",
    )
    pm = PersonaManager(str(tmp_path))
    loaded = next(iter(pm.personas.values()))
    assert loaded.id == "donald-trump"
    assert loaded.name == "Donald Trump"


def test_load_persona_file_skips_unrecoverable_id(tmp_path) -> None:
    """When neither the declared id nor the filename can produce a valid id,
    the file should be skipped."""
    file_path = tmp_path / "$$$.yaml"
    file_path.write_text("id: '...'\nname: X\n", encoding="utf-8")
    pm = PersonaManager(str(tmp_path))
    assert pm.personas == {}
