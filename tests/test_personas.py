"""Tests for Persona model validation, prompt rendering, and repository I/O."""

import pytest
from pydantic import ValidationError

from persona_agent.personas import Persona, PersonaPersistenceError, PersonaRepository


@pytest.mark.parametrize(
    "bad_id",
    [
        "../../etc/passwd",
        "MixedCase",
        "foo/bar",
        "foo\x00bar",
        "a" * 65,
    ],
)
def test_persona_id_rejects_invalid_values(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        Persona(id=bad_id, name="Test")


def test_persona_id_accepts_uuid_and_slug() -> None:
    assert Persona(id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", name="A").id.startswith(
        "aaaa"
    )
    assert Persona(id="trump_2024-1", name="B").id == "trump_2024-1"


def test_build_system_prompt_prefers_custom_prompt() -> None:
    persona = Persona(name="Ada", system_prompt="Custom prompt.")
    assert persona.build_system_prompt() == "Custom prompt."


def test_build_system_prompt_renders_fields() -> None:
    persona = Persona(
        name="Ada",
        description="Mathematician",
        language_style={"tone": "Clear", "common_phrases": ["Precisely"]},
        personal_background={"profession": "Mathematician"},
        knowledge_domains={"computing": ["analytical engine"]},
    )
    prompt = persona.build_system_prompt()
    assert "You are Ada." in prompt
    assert "- Tone: Clear" in prompt
    assert '"Precisely"' in prompt
    assert "- Profession: Mathematician" in prompt
    assert "- Computing: analytical engine" in prompt


def test_save_persona_rejects_bypassed_invalid_id(tmp_path) -> None:
    repo = PersonaRepository(tmp_path)
    bypassed = Persona.model_construct(id="../escape", name="X")
    with pytest.raises(ValueError, match="Invalid persona id"):
        repo.save_persona(bypassed)


def test_load_persona_file_normalizes_legacy_id(tmp_path) -> None:
    (tmp_path / "legacy.yaml").write_text(
        "id: Donald.Trump\nname: Donald Trump\ndescription: legacy\n",
        encoding="utf-8",
    )
    repo = PersonaRepository(tmp_path)
    loaded = repo.get_persona("donald-trump")
    assert loaded is not None
    assert loaded.name == "Donald Trump"


def test_load_persona_file_skips_unrecoverable_id(tmp_path) -> None:
    (tmp_path / "$$$.yaml").write_text("id: '...'\nname: X\n", encoding="utf-8")
    repo = PersonaRepository(tmp_path)
    assert repo.personas == {}


def test_delete_persona_removes_loaded_file(tmp_path) -> None:
    persona_file = tmp_path / "alice.yaml"
    persona_file.write_text(
        "id: alice\nname: Alice\ndescription: Example persona\n",
        encoding="utf-8",
    )
    repo = PersonaRepository(tmp_path)

    assert repo.delete_persona("alice") is True
    assert not persona_file.exists()
    assert PersonaRepository(tmp_path).get_persona("alice") is None


def test_delete_persona_only_removes_file_matching_declared_id(tmp_path) -> None:
    aliased_file = tmp_path / "alice.yaml"
    aliased_file.write_text(
        "id: bob\nname: Bob\ndescription: Stored in alice.yaml\n",
        encoding="utf-8",
    )
    unrelated_file = tmp_path / "bob.yaml"
    unrelated_file.write_text(
        "id: carol\nname: Carol\ndescription: Stored in bob.yaml\n",
        encoding="utf-8",
    )
    repo = PersonaRepository(tmp_path)

    assert repo.delete_persona("bob") is True
    assert not aliased_file.exists()
    assert unrelated_file.exists()

    reloaded = PersonaRepository(tmp_path)
    assert reloaded.get_persona("bob") is None
    assert reloaded.get_persona("carol") is not None


def test_delete_persona_preserves_state_on_os_error(tmp_path, monkeypatch) -> None:
    persona_file = tmp_path / "alice.yaml"
    persona_file.write_text(
        "id: alice\nname: Alice\ndescription: Example\n",
        encoding="utf-8",
    )
    repo = PersonaRepository(tmp_path)

    def fail_unlink(self, *args, **kwargs) -> None:
        raise OSError(f"cannot remove {self}")

    monkeypatch.setattr("pathlib.Path.unlink", fail_unlink)

    with pytest.raises(PersonaPersistenceError):
        repo.delete_persona("alice")

    assert repo.get_persona("alice") is not None
    assert persona_file in repo._persona_files["alice"]  # noqa: SLF001


def test_update_persona_does_not_mutate_on_validation_error(tmp_path) -> None:
    repo = PersonaRepository(tmp_path)
    repo.add_persona({"id": "alice", "name": "Alice", "description": "Original"})

    with pytest.raises(ValidationError):
        repo.update_persona("alice", {"name": 12345})

    current = repo.get_persona("alice")
    assert current is not None
    assert current.name == "Alice"
    assert current.description == "Original"
