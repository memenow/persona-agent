"""Persona domain model and system-prompt rendering."""

import uuid
from typing import Any

from pydantic import BaseModel, Field

# Lowercase ASCII letters, digits, hyphen, underscore; 1-64 chars.
# Shared between Pydantic validation and filesystem path-safety checks.
PERSONA_ID_PATTERN = r"^[a-z0-9_-]{1,64}$"


class Persona(BaseModel):
    """Character definition used to configure an agent's identity and prompt."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        pattern=PERSONA_ID_PATTERN,
    )
    name: str
    description: str = ""
    personal_background: dict[str, Any] = Field(default_factory=dict)
    language_style: dict[str, Any] = Field(default_factory=dict)
    knowledge_domains: dict[str, Any] = Field(default_factory=dict)
    interaction_samples: list[dict[str, Any]] = Field(default_factory=list)
    system_prompt: str | None = None

    def build_system_prompt(self) -> str:
        """Return the custom system prompt or render one from persona fields."""
        if self.system_prompt:
            return self.system_prompt

        parts = [f"You are {self.name}.", self.description]

        if self.language_style:
            parts.append("\nLanguage style:")
            for key, label in (
                ("tone", "Tone"),
                ("vocabulary", "Vocabulary"),
                ("speaking_style", "Speaking style"),
            ):
                if key in self.language_style:
                    parts.append(f"- {label}: {self.language_style[key]}")
            common_phrases = self.language_style.get("common_phrases")
            if isinstance(common_phrases, list) and common_phrases:
                phrases = ", ".join(f'"{phrase}"' for phrase in common_phrases)
                parts.append(f"- Frequently use phrases like: {phrases}")

        if self.personal_background:
            parts.append("\nBackground:")
            for key, value in self.personal_background.items():
                parts.append(f"- {key.replace('_', ' ').title()}: {value}")

        if self.knowledge_domains:
            domain_lines = [
                f"- {domain.replace('_', ' ').title()}: {', '.join(topics)}"
                for domain, topics in self.knowledge_domains.items()
                if isinstance(topics, list)
            ]
            if domain_lines:
                parts.append("\nYou have knowledge and expertise in:")
                parts.extend(domain_lines)

        parts.append(
            "\nYou have access to external tools through the Model Context Protocol (MCP). "
            "Use them when they help answer accurately or retrieve missing information."
        )
        parts.append(
            "\nStay in character and respond consistently with this persona's style, "
            "knowledge, and background."
        )

        return "\n".join(parts)
