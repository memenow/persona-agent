"""Configuration Loader module.

This module provides utilities for loading and saving persona configurations
from different formats and sources.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import yaml

from persona_agent.core.persona_profile import PersonaProfile


class ConfigLoader:
    """Utility class for loading and saving persona configurations.
    
    This class provides methods to load persona configurations from various
    sources such as JSON files, YAML files, or dictionaries, and convert them
    into PersonaProfile instances.
    """
    
    def __init__(self):
        """Initialize a new ConfigLoader."""
        self.logger = logging.getLogger(__name__)
    
    def load_from_json(self, file_path: str) -> PersonaProfile:
        """Load a persona configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON file.
            
        Returns:
            A PersonaProfile instance.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded persona configuration from {file_path}")
            return PersonaProfile.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            raise
    
    def load_from_yaml(self, file_path: str) -> PersonaProfile:
        """Load a persona configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML file.
            
        Returns:
            A PersonaProfile instance.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            self.logger.info(f"Loaded persona configuration from {file_path}")
            return PersonaProfile.from_dict(data)
        except (FileNotFoundError, yaml.YAMLError) as e:
            self.logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            raise
    
    def save_to_json(self, profile: PersonaProfile, file_path: str) -> None:
        """Save a persona configuration to a JSON file.
        
        Args:
            profile: The PersonaProfile to save.
            file_path: Path where to save the configuration.
            
        Raises:
            IOError: If the file cannot be written.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            
            self.logger.info(f"Saved persona configuration to {file_path}")
        except IOError as e:
            self.logger.error(f"Error saving configuration to {file_path}: {str(e)}")
            raise
    
    def save_to_yaml(self, profile: PersonaProfile, file_path: str) -> None:
        """Save a persona configuration to a YAML file.
        
        Args:
            profile: The PersonaProfile to save.
            file_path: Path where to save the configuration.
            
        Raises:
            IOError: If the file cannot be written.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                yaml.dump(profile.to_dict(), f, default_flow_style=False)
            
            self.logger.info(f"Saved persona configuration to {file_path}")
        except IOError as e:
            self.logger.error(f"Error saving configuration to {file_path}: {str(e)}")
            raise
    
    def load_from_directory(self, directory_path: str) -> Dict[str, PersonaProfile]:
        """Load all persona configurations from a directory.
        
        This method loads all JSON and YAML files in the specified directory
        and converts them into PersonaProfile instances.
        
        Args:
            directory_path: Path to the directory containing configuration files.
            
        Returns:
            Dictionary mapping persona names to PersonaProfile instances.
        """
        profiles = {}
        
        try:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                
                if not os.path.isfile(file_path):
                    continue
                
                if filename.endswith('.json'):
                    try:
                        profile = self.load_from_json(file_path)
                        profiles[profile.name] = profile
                    except Exception as e:
                        self.logger.warning(
                            f"Skipping {file_path} due to error: {str(e)}"
                        )
                
                elif filename.endswith(('.yaml', '.yml')):
                    try:
                        profile = self.load_from_yaml(file_path)
                        profiles[profile.name] = profile
                    except Exception as e:
                        self.logger.warning(
                            f"Skipping {file_path} due to error: {str(e)}"
                        )
            
            self.logger.info(
                f"Loaded {len(profiles)} persona profiles from {directory_path}"
            )
            return profiles
        except FileNotFoundError:
            self.logger.warning(f"Directory {directory_path} not found")
            return {}
