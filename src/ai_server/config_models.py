# Static configuration models using Pydantic
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Server configuration."""

    listen_interfaces: str = "0.0.0.0"
    listen_port: int = 8000


class HistoryConfig(BaseModel):
    """History management configuration."""

    cut_after: int = Field(default=10, description="Cut history after N messages")
    summarize_after: int = Field(default=0, description="Summarize after N messages")
    use_common: bool = Field(
        default=False, description="Use common conversation history"
    )
    post_to_common: bool = Field(
        default=False, description="Post to common conversation"
    )


class LLMConfig(BaseModel):
    """LLM configuration."""

    model: str
    temperature: Optional[float] = None
    streaming: bool = False
    proxy: Optional[str] = None
    history: HistoryConfig = Field(default_factory=HistoryConfig)


class AgentConfig(BaseModel):
    """Agent configuration."""

    llms: Dict[str, LLMConfig] = Field(default_factory=dict)
    resources: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)


class APIConfig(BaseModel):
    """API configuration for external services."""

    base_url: str
    api_key: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    debug: Dict[str, bool] = Field(
        default_factory=lambda: {
            "events": False,
            "prompts": False,
            "messages_diff": False,
            "llm_init": False,
        }
    )
    loggers: Dict[str, Any] = Field(
        default_factory=lambda: {"suppress": [], "suppress_level": "WARNING"}
    )


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) configuration."""

    enabled: bool = Field(default=True)
    servers: Dict[str, Dict] = Field(default_factory=dict)


class AppConfig(BaseModel):
    """Main application configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    llm_api: Dict[str, APIConfig] = Field(default_factory=dict)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    endpoints: Dict[str, str] = Field(default_factory=lambda: {"auto": "ai"})

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        frozen = True  # Make config immutable after creation


# Global config instance
app_config = AppConfig()


def load_config_from_dict(config_dict: dict) -> AppConfig:
    """
    Load configuration from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        AppConfig instance
    """
    return AppConfig(**config_dict)


def load_config_from_files(config_files: List[str]) -> AppConfig:
    """
    Load configuration from YAML files.

    Args:
        config_files: List of configuration file paths

    Returns:
        AppConfig instance
    """
    import yaml
    from pathlib import Path

    config_data = {}

    for config_file in config_files:
        if Path(config_file).exists():
            with open(config_file, "r") as f:
                file_data = yaml.safe_load(f) or {}
                # Deep merge configurations
                config_data.update(file_data)

    return load_config_from_dict(config_data)


def update_env_values(config_dict: dict) -> dict:
    """
    Update configuration values that are set to '.env' with environment variables.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Updated configuration dictionary
    """
    import os

    def process_dict(d: dict, path: List[str] = []) -> dict:
        result = {}
        for k, v in d.items():
            current_path = path + [k]
            if isinstance(v, dict):
                result[k] = process_dict(v, current_path)
            elif v == ".env":
                # Construct environment variable name
                env_parts = (
                    current_path[-2:] if len(current_path) >= 2 else current_path
                )
                env_var = "_".join(env_parts).upper()
                env_value = os.getenv(env_var)
                if env_value is None:
                    raise ValueError(
                        f"Environment variable {env_var} not found for config path {'.'.join(current_path)}"
                    )
                result[k] = env_value
            else:
                result[k] = v
        return result

    return process_dict(config_dict)


def validate_config(config: AppConfig) -> List[str]:
    """
    Validate configuration and return list of validation errors.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages
    """
    errors = []

    # Validate server configuration
    if config.server.listen_port < 1 or config.server.listen_port > 65535:
        errors.append(f"Invalid server port: {config.server.listen_port}")

    # Validate agents have required LLMs
    for agent_name, agent_config in config.agents.items():
        if not agent_config.llms:
            errors.append(f"Agent '{agent_name}' has no LLMs configured")

        for llm_name, llm_config in agent_config.llms.items():
            if not llm_config.model:
                errors.append(
                    f"LLM '{llm_name}' in agent '{agent_name}' has no model specified"
                )

    # Validate API configurations
    for api_name, api_config in config.llm_api.items():
        if not api_config.base_url:
            errors.append(f"API '{api_name}' has no base_url configured")

    # Validate logging level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.logging.level.upper() not in valid_levels:
        errors.append(f"Invalid logging level: {config.logging.level}")

    return errors


def load_and_validate_config(config_files: List[str]) -> AppConfig:
    """
    Load and validate configuration from files.

    Args:
        config_files: List of configuration file paths

    Returns:
        Validated AppConfig instance

    Raises:
        ValueError: If configuration is invalid
    """
    # Load configuration
    config_dict = {}
    for config_file in config_files:
        config_dict = load_config_from_files([config_file]).dict()

    # Update environment variables
    config_dict = update_env_values(config_dict)

    # Create and validate configuration
    config = AppConfig(**config_dict)
    validation_errors = validate_config(config)

    if validation_errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in validation_errors
        )
        raise ValueError(error_msg)

    return config
