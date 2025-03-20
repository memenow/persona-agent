"""Persona Profile module.

This module defines the PersonaProfile class which encapsulates all information
about a specific persona, including personal background, language style,
knowledge domains, and behavioral patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any


@dataclass
class PersonaProfile:
    """A class representing a persona's complete profile information.

    This class stores all information needed to simulate a specific persona,
    organized into different categories for easy access and manipulation.
    
    Attributes:
        name: The name of the persona.
        personal_background: Background information about the persona.
        language_style: Information about language patterns and expression.
        knowledge_domains: Knowledge areas and expertise.
        interaction_samples: Examples of how the persona interacts.
    """

    name: str
    description: str = ""
    
    # Personal background including basic info, education, work experience, etc.
    personal_background: Dict[str, str] = field(default_factory=dict)
    
    # Language and expression characteristics
    language_style: Dict[str, str] = field(default_factory=dict)
    
    # Knowledge domains, expertise, interests, and values
    knowledge_domains: Dict[str, List[str]] = field(default_factory=dict)
    
    # Interaction samples like conversations, social media posts, articles
    interaction_samples: List[Dict[str, str]] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert the profile to a dictionary.
        
        Returns:
            A dictionary representation of the profile.
        """
        return {
            "name": self.name,
            "description": self.description,
            "personal_background": self.personal_background,
            "language_style": self.language_style,
            "knowledge_domains": self.knowledge_domains,
            "interaction_samples": self.interaction_samples,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PersonaProfile":
        """Create a profile from a dictionary.
        
        Args:
            data: Dictionary containing profile information.
            
        Returns:
            A PersonaProfile instance.
        """
        # Handle special case for knowledge_domains
        knowledge_domains = data.get("knowledge_domains", {})
        
        # If knowledge_domains is a list, convert it to dictionary format
        if isinstance(knowledge_domains, list):
            converted_domains = {}
            for domain_item in knowledge_domains:
                if isinstance(domain_item, dict):
                    name = domain_item.get("name", "domain")
                    topics = domain_item.get("topics", [])
                    converted_domains[name] = topics
            knowledge_domains = converted_domains
        
        # Handle special case for interaction_samples
        interaction_samples = data.get("interaction_samples", [])
        
        # Convert new interaction_samples format to old format
        converted_samples = []
        for sample in interaction_samples:
            if isinstance(sample, dict) and "user_query" in sample and "response" in sample:
                converted_samples.append({
                    "type": "conversation",
                    "content": f"Q: {sample['user_query']}\nA: {sample['response']}"
                })
            else:
                converted_samples.append(sample)
                
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            personal_background=data.get("personal_background", {}),
            language_style=data.get("language_style", {}),
            knowledge_domains=knowledge_domains,
            interaction_samples=converted_samples,
            metadata=data.get("metadata", {}),
        )
    
    def add_background_info(self, key: str, value: str) -> None:
        """Add or update background information.
        
        Args:
            key: Category of background information.
            value: The information to add.
        """
        self.personal_background[key] = value
    
    def add_language_style_info(self, key: str, value: str) -> None:
        """Add or update language style information.
        
        Args:
            key: Category of language style.
            value: The information to add.
        """
        self.language_style[key] = value
    
    def add_knowledge_domain(self, domain: str, items: List[str]) -> None:
        """Add or update knowledge domain information.
        
        Args:
            domain: Knowledge domain name.
            items: List of knowledge items in this domain.
        """
        self.knowledge_domains[domain] = items
    
    def add_interaction_sample(self, sample_type: str, content: str) -> None:
        """Add an interaction sample.
        
        Args:
            sample_type: Type of interaction (e.g., "conversation", "social_media").
            content: The sample content.
        """
        self.interaction_samples.append({"type": sample_type, "content": content})
