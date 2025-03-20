"""Configuration settings for the API service.

This module provides configuration management for the Persona Agent API service,
including settings for server, authentication, CORS, and model configurations.
"""

import os
import json
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Constants for environment variable names, avoiding hardcoding
ENV_API_HOST = "API_HOST"
ENV_API_PORT = "API_PORT"
ENV_API_DEBUG = "API_DEBUG"
ENV_API_PREFIX = "API_PREFIX"
ENV_API_ENABLE_AUTH = "API_ENABLE_AUTH"
ENV_API_KEY_HEADER = "API_KEY_HEADER"
ENV_API_ALLOWED_KEYS = "API_ALLOWED_KEYS"
ENV_API_ENABLE_CORS = "API_ENABLE_CORS"
ENV_API_ALLOWED_ORIGINS = "API_ALLOWED_ORIGINS"
ENV_PERSONAS_DIR = "PERSONAS_DIR"
ENV_LLM_CONFIG_PATH = "LLM_CONFIG_PATH"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_DEFAULT_MODEL = "DEFAULT_MODEL"
ENV_MCP_CONFIG_PATH = "MCP_CONFIG_PATH"

class ApiConfig(BaseModel):
    """API service configuration.
    
    This class defines all configurable settings for the API service,
    including server settings, authentication, CORS, and model configurations.
    It uses Pydantic for validation and type checking.
    
    Attributes:
        host: Host address to bind the API server.
        port: Port number to bind the API server.
        debug: Whether to run in debug mode.
        api_prefix: Prefix for all API endpoints.
        enable_auth: Whether to enable API authentication.
        api_key_header: Header name for API key authentication.
        allowed_api_keys: List of valid API keys.
        enable_cors: Whether to enable CORS.
        allowed_origins: List of allowed origins for CORS.
        personas_dir: Directory containing persona definitions.
        llm_config_path: Path to the LLM configuration file.
        mcp_config_path: Path to the MCP service configuration file.
        default_model: Default LLM model to use for agents.
        openai_api_key: Optional OpenAI API key.
        openai_api_base: Optional custom OpenAI API base URL.
    """
    host: str = Field(default="0.0.0.0", description="Host to bind the API server")
    port: int = Field(default=8000, description="Port to bind the API server")
    debug: bool = Field(default=False, description="Run in debug mode")
    api_prefix: str = Field(default="/api/v1", description="API endpoint prefix")
    
    # Authentication settings
    enable_auth: bool = Field(default=False, description="Enable API authentication")
    api_key_header: str = Field(default="X-API-Key", description="Header name for API key")
    allowed_api_keys: List[str] = Field(default=[], description="List of allowed API keys")
    
    # CORS settings
    enable_cors: bool = Field(default=True, description="Enable CORS")
    allowed_origins: List[str] = Field(default=["*"], description="Allowed origins for CORS")
    
    # Persona settings
    personas_dir: str = Field(
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                            "examples", "personas"),
        description="Directory containing persona definitions"
    )
    
    # LLM config settings
    llm_config_path: str = Field(
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                            "config", "llm_config.json"),
        description="Path to LLM configuration file"
    )
    
    # MCP config settings
    mcp_config_path: Optional[str] = Field(
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                            "config", "mcp_config.json"),
        description="Path to MCP service configuration file"
    )
    
    # Model settings
    default_model: str = Field(default="gpt-4o", description="Default model to use for agents")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_api_base: Optional[str] = Field(default=None, description="OpenAI API base URL")


def load_config() -> ApiConfig:
    """Load configuration from environment variables and config files.
    
    This function creates an ApiConfig instance by first loading default values,
    then overriding them with environment variables, and finally with values
    from a configuration file if specified.
    
    Environment variables take precedence over config file values.
    
    Returns:
        An ApiConfig instance with the loaded configuration.
    """
    config = ApiConfig(
        host=os.environ.get(ENV_API_HOST, "0.0.0.0"),
        port=int(os.environ.get(ENV_API_PORT, "8000")),
        debug=os.environ.get(ENV_API_DEBUG, "").lower() in ("true", "1", "yes"),
        api_prefix=os.environ.get(ENV_API_PREFIX, "/api/v1"),
        enable_auth=os.environ.get(ENV_API_ENABLE_AUTH, "").lower() in ("true", "1", "yes"),
        api_key_header=os.environ.get(ENV_API_KEY_HEADER, "X-API-Key"),
        allowed_api_keys=os.environ.get(ENV_API_ALLOWED_KEYS, "").split(",") if os.environ.get(ENV_API_ALLOWED_KEYS) else [],
        enable_cors=os.environ.get(ENV_API_ENABLE_CORS, "").lower() not in ("false", "0", "no"),
        allowed_origins=os.environ.get(ENV_API_ALLOWED_ORIGINS, "*").split(","),
        personas_dir=os.environ.get(ENV_PERSONAS_DIR, ApiConfig().personas_dir),
        llm_config_path=os.environ.get(ENV_LLM_CONFIG_PATH, ApiConfig().llm_config_path),
        mcp_config_path=os.environ.get(ENV_MCP_CONFIG_PATH, ApiConfig().mcp_config_path),
        default_model=os.environ.get(ENV_DEFAULT_MODEL, ApiConfig().default_model),
    )
    
    # Load LLM configuration from file
    try:
        with open(config.llm_config_path, 'r', encoding='utf-8') as f:
            llm_config = json.load(f)
            
            # Set default model from config if not set by environment variable
            if ENV_DEFAULT_MODEL not in os.environ and "default_model" in llm_config:
                config.default_model = llm_config["default_model"]
            
            # Get API keys and settings from model_configs
            if "model_configs" in llm_config and isinstance(llm_config["model_configs"], list):
                for model_config in llm_config["model_configs"]:
                    if model_config.get("name") == config.default_model:
                        # Only use values from the config file if the environment variable is not set
                        if not config.openai_api_key:
                            config.openai_api_key = model_config.get("api_key")
                        if not config.openai_api_base:
                            config.openai_api_base = model_config.get("api_base")
                        break
    except Exception as e:
        print(f"Warning: Failed to load LLM configuration from {config.llm_config_path}: {e}")
        # Fallback to environment variables
        if not config.openai_api_key:
            config.openai_api_key = os.environ.get(ENV_OPENAI_API_KEY)

    return config
