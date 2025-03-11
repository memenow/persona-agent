"""LLM Configuration module.

This module provides utilities for loading and managing LLM configurations.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join("config", "llm_config.json")


class LLMConfigManager:
    """Manager for LLM configurations.
    
    This class provides methods for loading, validating, and accessing
    LLM configurations from a JSON file.
    
    Attributes:
        config: The loaded LLM configuration.
        model_configs: Dictionary mapping model names to configurations.
        default_model: Name of the default model.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize a new LLMConfigManager.
        
        Args:
            config_path: Path to the LLM configuration file. If None, uses the default path.
        """
        self.config = {}
        self.model_configs = {}
        self.default_model = ""
        self.api_settings = {}
        
        self.load_config(config_path or DEFAULT_CONFIG_PATH)
    
    def load_config(self, config_path: str) -> None:
        """Load LLM configuration from a file.
        
        Args:
            config_path: Path to the configuration file.
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            json.JSONDecodeError: If the configuration file contains invalid JSON.
            ValueError: If the configuration is invalid.
        """
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
            # Validate the configuration
            self._validate_config()
            
            # Extract model configurations
            self.model_configs = {
                model_config.get("name"): model_config
                for model_config in self.config.get("model_configs", [])
            }
            
            # Get the default model
            self.default_model = self.config.get("default_model", "")
            if not self.default_model and self.model_configs:
                # Use the first model as default if not specified
                self.default_model = next(iter(self.model_configs))
            
            # Get API settings
            self.api_settings = self.config.get("api_settings", {})
            
            logger.info(f"Loaded LLM configuration with {len(self.model_configs)} models")
            logger.info(f"Default model: {self.default_model}")
        except FileNotFoundError:
            logger.error(f"LLM configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in LLM configuration file: {config_path}")
            raise
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration.
        
        Raises:
            ValueError: If the configuration is invalid.
        """
        if not isinstance(self.config, dict):
            raise ValueError("LLM configuration must be a dictionary")
        
        if "model_configs" not in self.config:
            raise ValueError("LLM configuration must contain a 'model_configs' key")
        
        if not isinstance(self.config.get("model_configs"), list):
            raise ValueError("'model_configs' must be a list")
        
        for model_config in self.config.get("model_configs", []):
            if not isinstance(model_config, dict):
                raise ValueError("Each model configuration must be a dictionary")
            
            if "name" not in model_config:
                raise ValueError("Each model configuration must have a 'name' key")
            
            if "model" not in model_config:
                raise ValueError("Each model configuration must have a 'model' key")
    
    def get_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get the configuration for a specific model.
        
        Args:
            model_name: Name of the model to get. If None, uses the default model.
            
        Returns:
            The model configuration.
            
        Raises:
            ValueError: If the model doesn't exist.
        """
        model_name = model_name or self.default_model
        
        if not model_name:
            raise ValueError("No model name provided and no default model configured")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Model not found: {model_name}")
        
        return self.model_configs[model_name]
    
    def get_llm_client(self, model_name: Optional[str] = None) -> Any:
        """Get an LLM client for a specific model.
        
        Args:
            model_name: Name of the model to get a client for. If None, uses the default model.
            
        Returns:
            An LLM client for the specified model.
            
        Raises:
            ValueError: If the model doesn't exist or if the client creation fails.
        """
        model_config = self.get_model_config(model_name)
        
        # Identify the model provider based on the model name or API base URL
        provider = self._identify_provider(model_config)
        
        try:
            if provider == "openai":
                return self._create_openai_client(model_config)
            elif provider == "anthropic":
                return self._create_anthropic_client(model_config)
            else:
                raise ValueError(f"Unsupported model provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to create LLM client: {str(e)}")
            raise
    
    def _identify_provider(self, model_config: Dict[str, Any]) -> str:
        """Identify the provider for a model configuration.
        
        Args:
            model_config: The model configuration.
            
        Returns:
            The provider name.
        """
        # All models are treated as OpenAI-compatible
        return "openai"
    
    def _create_openai_client(self, model_config: Dict[str, Any]) -> Any:
        """Create an OpenAI client.
        
        Args:
            model_config: The model configuration.
            
        Returns:
            An OpenAI client.
        """
        try:
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            
            client = OpenAIChatCompletionClient(
                model=model_config.get("model", "gpt-4o"),
                api_key=model_config.get("api_key"),
                base_url=model_config.get("api_base"),
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_tokens", 4000),
                top_p=model_config.get("top_p", 1.0),
                frequency_penalty=model_config.get("frequency_penalty", 0.0),
                presence_penalty=model_config.get("presence_penalty", 0.0),
            )
            
            return client
        except ImportError:
            logger.error("Failed to import OpenAIChatCompletionClient")
            raise


# Create a global config manager for convenience
config_manager = LLMConfigManager()


def get_llm_client(model_name: Optional[str] = None) -> Any:
    """Get an LLM client for a specific model.
    
    Args:
        model_name: Name of the model to get a client for. If None, uses the default model.
        
    Returns:
        An LLM client for the specified model.
    """
    return config_manager.get_llm_client(model_name)


def get_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get the configuration for a specific model.
    
    Args:
        model_name: Name of the model to get. If None, uses the default model.
        
    Returns:
        The model configuration.
    """
    return config_manager.get_model_config(model_name)


def load_config(config_path: str) -> None:
    """Load LLM configuration from a file.
    
    Args:
        config_path: Path to the configuration file.
    """
    config_manager.load_config(config_path)
