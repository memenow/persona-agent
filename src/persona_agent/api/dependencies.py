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
def get_persona_manager() -> PersonaManager:
    """Get the persona manager singleton."""
    config = get_config()
    return PersonaManager(config.personas_dir)


@lru_cache
def get_agent_factory() -> AgentFactory:
    """Get the agent factory singleton.

    The unmocked path returns a stable instance with no shared LLM client or
    MCP manager. ``server.create_app`` overrides this dependency to inject the
    fully wired factory built during application startup.
    """
    config = get_config()
    return AgentFactory(llm_config_path=config.llm_config_path)


def clear_dependency_caches() -> None:
    """Reset all ``@lru_cache`` singletons.

    Intended for test setup/teardown so each test sees a fresh ``ApiConfig``,
    ``PersonaManager``, and ``AgentFactory``. Production code should rely on
    ``app.dependency_overrides`` instead.
    """
    get_config.cache_clear()
    get_persona_manager.cache_clear()
    get_agent_factory.cache_clear()
