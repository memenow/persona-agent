"""Persona manager for loading and managing persona definitions."""

import os
import json
import yaml
import uuid
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class Persona(BaseModel):
    """Persona model representing a character definition."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the persona")
    name: str = Field(..., description="Name of the persona")
    description: str = Field(default="", description="Description of the persona")
    personal_background: Dict[str, Any] = Field(default_factory=dict, description="Personal background information")
    language_style: Dict[str, Any] = Field(default_factory=dict, description="Language style characteristics")
    knowledge_domains: Dict[str, Any] = Field(default_factory=dict, description="Knowledge domains and expertise")
    interaction_samples: List[Dict[str, Any]] = Field(default_factory=list, description="Sample interactions")
    system_prompt: Optional[str] = None
    
    def generate_system_prompt(self) -> str:
        """Generate a system prompt for the persona based on its attributes."""
        if self.system_prompt:
            return self.system_prompt
        
        prompt_parts = []
        prompt_parts.append(f"You are {self.name}.")
        prompt_parts.append(self.description)
        
        # Add language style information
        if self.language_style:
            prompt_parts.append("\nLanguage style:")
            if "tone" in self.language_style:
                prompt_parts.append(f"- Tone: {self.language_style['tone']}")
            if "vocabulary" in self.language_style:
                prompt_parts.append(f"- Vocabulary: {self.language_style['vocabulary']}")
            if "speaking_style" in self.language_style:
                prompt_parts.append(f"- Speaking style: {self.language_style['speaking_style']}")
            if "common_phrases" in self.language_style and isinstance(self.language_style["common_phrases"], list):
                phrases = ", ".join(f'"{phrase}"' for phrase in self.language_style["common_phrases"])
                prompt_parts.append(f"- Frequently use phrases like: {phrases}")
        
        # Add background information
        if self.personal_background:
            background_parts = []
            for key, value in self.personal_background.items():
                background_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
            if background_parts:
                prompt_parts.append("\nBackground:")
                prompt_parts.extend(background_parts)
        
        # Add knowledge domains
        if self.knowledge_domains:
            prompt_parts.append("\nYou have knowledge and expertise in:")
            for domain, topics in self.knowledge_domains.items():
                if isinstance(topics, list):
                    topics_str = ", ".join(topics)
                    prompt_parts.append(f"- {domain.replace('_', ' ').title()}: {topics_str}")
        
        # Add instruction to use MCP tools when appropriate
        prompt_parts.append("\nYou have access to various tools through the Model Context Protocol (MCP). Use these tools when they would help you provide better responses or access information you don't have.")
        
        # Add instruction to stay in character
        prompt_parts.append("\nAlways stay in character and respond as this persona would, using their speaking style, knowledge, and background to inform your responses.")
        
        return "\n".join(prompt_parts)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Persona":
        """Create a Persona instance from a dictionary."""
        # Handle optional system_prompt
        system_prompt = data.pop("system_prompt", None)
        persona = cls(**data)
        if system_prompt:
            persona.system_prompt = system_prompt
        return persona


class PersonaManager:
    """Manager for loading and handling persona definitions."""
    
    def __init__(self, personas_dir: str):
        self.personas_dir = personas_dir
        self.personas: Dict[str, Persona] = {}
        self._load_personas()
    
    def _load_personas(self) -> None:
        """Load all persona definitions from the personas directory."""
        if not os.path.exists(self.personas_dir):
            os.makedirs(self.personas_dir, exist_ok=True)
            return
        
        for filename in os.listdir(self.personas_dir):
            file_path = os.path.join(self.personas_dir, filename)
            if os.path.isfile(file_path) and (filename.endswith('.json') or filename.endswith('.yaml') or filename.endswith('.yml')):
                try:
                    persona = self._load_persona_file(file_path)
                    if persona:
                        self.personas[persona.id] = persona
                except Exception as e:
                    print(f"Error loading persona from {file_path}: {e}")
    
    def _load_persona_file(self, file_path: str) -> Optional[Persona]:
        """Load a persona definition from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                else:  # YAML file
                    data = yaml.safe_load(f)
                
                # Ensure an ID is present
                if "id" not in data:
                    base_name = os.path.basename(file_path)
                    name = os.path.splitext(base_name)[0]
                    data["id"] = name
                
                return Persona.from_dict(data)
        except Exception as e:
            print(f"Error loading persona file {file_path}: {e}")
            return None
    
    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """Get a persona by ID."""
        return self.personas.get(persona_id)
    
    def list_personas(self) -> List[Dict[str, Any]]:
        """List all available personas."""
        return [
            {
                "id": persona.id,
                "name": persona.name,
                "description": persona.description
            }
            for persona in self.personas.values()
        ]
    
    def add_persona(self, persona_data: Dict[str, Any]) -> Persona:
        """Add a new persona."""
        persona = Persona.from_dict(persona_data)
        self.personas[persona.id] = persona
        return persona
    
    def update_persona(self, persona_id: str, persona_data: Dict[str, Any]) -> Optional[Persona]:
        """Update an existing persona."""
        if persona_id not in self.personas:
            return None
        
        persona_data["id"] = persona_id  # Ensure ID remains unchanged
        persona = Persona.from_dict(persona_data)
        self.personas[persona_id] = persona
        return persona
    
    def delete_persona(self, persona_id: str) -> bool:
        """Delete a persona by ID."""
        if persona_id not in self.personas:
            return False
        
        del self.personas[persona_id]
        return True
    
    def save_persona(self, persona: Persona, format: str = 'json') -> str:
        """Save a persona to a file."""
        if format not in ('json', 'yaml'):
            raise ValueError("Format must be 'json' or 'yaml'")
        
        filename = f"{persona.id}.{format}"
        file_path = os.path.join(self.personas_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format == 'json':
                json.dump(persona.dict(), f, indent=2)
            else:
                yaml.dump(persona.dict(), f, default_flow_style=False)
        
        return file_path 