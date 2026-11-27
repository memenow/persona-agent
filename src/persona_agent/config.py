"""Application settings and JSON configuration loaders."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PERSONAS_DIR = str(_PROJECT_ROOT / "examples" / "personas")
_DEFAULT_LLM_CONFIG_PATH = str(_PROJECT_ROOT / "config" / "llm_config.json")
_DEFAULT_MCP_CONFIG_PATH = str(_PROJECT_ROOT / "config" / "mcp_config.json")
_DEFAULT_MODEL = "gpt-4o-mini"

ENV_API_HOST = "API_HOST"
ENV_API_PORT = "API_PORT"
ENV_API_DEBUG = "API_DEBUG"
ENV_API_PREFIX = "API_PREFIX"
ENV_API_ENABLE_AUTH = "API_ENABLE_AUTH"
ENV_API_KEY_HEADER = "API_KEY_HEADER"
ENV_API_ALLOWED_KEYS = "API_ALLOWED_KEYS"
ENV_API_ENABLE_CORS = "API_ENABLE_CORS"
ENV_API_ALLOWED_ORIGINS = "API_ALLOWED_ORIGINS"
ENV_API_PUBLIC_BASE_URL = "API_PUBLIC_BASE_URL"
ENV_PERSONAS_DIR = "PERSONAS_DIR"
ENV_LLM_CONFIG_PATH = "LLM_CONFIG_PATH"
ENV_MCP_CONFIG_PATH = "MCP_CONFIG_PATH"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_DEFAULT_MODEL = "DEFAULT_MODEL"


class Settings(BaseModel):
    """Runtime settings for the server, auth, CORS, personas, LLM, and MCP."""

    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    api_prefix: str = "/api/v1"

    enable_auth: bool = False
    api_key_header: str = "X-API-Key"
    allowed_api_keys: list[str] = Field(default_factory=list)

    enable_cors: bool = True
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])
    public_base_url: str | None = None

    personas_dir: str = _DEFAULT_PERSONAS_DIR
    llm_config_path: str = _DEFAULT_LLM_CONFIG_PATH
    mcp_config_path: str | None = _DEFAULT_MCP_CONFIG_PATH

    default_model: str = _DEFAULT_MODEL
    openai_api_key: str | None = None
    openai_api_base: str | None = None

    def resolve_public_base_url(self) -> str:
        """Return the externally reachable base URL for A2A discovery metadata."""
        if self.public_base_url:
            return self.public_base_url.rstrip("/")

        host = self.host
        if host in ("0.0.0.0", "::", ""):
            host = "localhost"
        elif ":" in host and not host.startswith("["):
            host = f"[{host}]"

        return f"http://{host}:{self.port}"


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    lowered = raw.strip().lower()
    if default:
        return lowered not in ("false", "0", "no")
    return lowered in ("true", "1", "yes")


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    return [item for part in raw.split(",") if (item := part.strip())]


def load_llm_config(path: str | Path | None) -> dict[str, Any]:
    """Read an LLM JSON config file, returning an empty dict if missing or invalid."""
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load LLM config from %s: %s", file_path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("LLM config %s must contain a JSON object", file_path)
        return {}
    return data


def load_settings() -> Settings:
    """Load settings from environment variables with JSON config and defaults as fallback."""
    settings = Settings(
        host=os.environ.get(ENV_API_HOST, "127.0.0.1"),
        port=int(os.environ.get(ENV_API_PORT, "8000")),
        debug=_parse_bool_env(ENV_API_DEBUG, default=False),
        api_prefix=os.environ.get(ENV_API_PREFIX, "/api/v1"),
        enable_auth=_parse_bool_env(ENV_API_ENABLE_AUTH, default=False),
        api_key_header=os.environ.get(ENV_API_KEY_HEADER, "X-API-Key"),
        allowed_api_keys=_parse_csv_env(ENV_API_ALLOWED_KEYS, default=[]),
        enable_cors=_parse_bool_env(ENV_API_ENABLE_CORS, default=True),
        allowed_origins=_parse_csv_env(ENV_API_ALLOWED_ORIGINS, default=["*"]),
        public_base_url=os.environ.get(ENV_API_PUBLIC_BASE_URL) or None,
        personas_dir=os.environ.get(ENV_PERSONAS_DIR, _DEFAULT_PERSONAS_DIR),
        llm_config_path=os.environ.get(ENV_LLM_CONFIG_PATH, _DEFAULT_LLM_CONFIG_PATH),
        mcp_config_path=os.environ.get(ENV_MCP_CONFIG_PATH, _DEFAULT_MCP_CONFIG_PATH),
        default_model=os.environ.get(ENV_DEFAULT_MODEL, _DEFAULT_MODEL),
    )

    llm_config = load_llm_config(settings.llm_config_path)
    if ENV_DEFAULT_MODEL not in os.environ and isinstance(
        llm_config.get("default_model"), str
    ):
        settings.default_model = llm_config["default_model"]

    model_configs = llm_config.get("model_configs")
    if isinstance(model_configs, list):
        for entry in model_configs:
            if not isinstance(entry, dict):
                continue
            if (
                entry.get("model") == settings.default_model
                or entry.get("name") == settings.default_model
            ):
                if not settings.openai_api_key:
                    settings.openai_api_key = entry.get("api_key") or llm_config.get(
                        "api_key"
                    )
                if not settings.openai_api_base:
                    settings.openai_api_base = entry.get("api_base") or llm_config.get(
                        "api_base"
                    )
                break

    if not settings.openai_api_key:
        settings.openai_api_key = (
            llm_config.get("api_key") or os.environ.get(ENV_OPENAI_API_KEY) or None
        )
    if not settings.openai_api_base:
        settings.openai_api_base = llm_config.get("api_base") or None

    return settings
