"""Dependencies for FastAPI dependency injection."""

from functools import lru_cache

from persona_agent.api.agent_factory import AgentFactory
from persona_agent.api.config import ApiConfig, load_config
from persona_agent.api.persona_manager import PersonaManager


@lru_cache
def get_config() -> ApiConfig:
    """Get the API configuration singleton."""
    return load_config()


@lru_cache
def get_api_config() -> ApiConfig:
    """Get the API configuration singleton (legacy name)."""
    return load_config()


def get_persona_manager() -> PersonaManager:
    """Get the persona manager instance."""
    config = get_config()
    return PersonaManager(config.personas_dir)


def get_agent_factory() -> AgentFactory:
    """Get the agent factory instance."""
    config = get_config()
    return AgentFactory(llm_config_path=config.llm_config_path)
