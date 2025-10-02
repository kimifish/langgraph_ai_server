#  -*- coding: utf-8 -*-
# pyright: basic
# pyright: reportAttributeAccessIssue=false

import logging
import os
import sys
import argparse
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from dotenv import load_dotenv
from kimiconfig import Config
from ai_server.logs.themes import common_theme, log_theme

cfg = Config(use_dataclasses=True)
install_rich_traceback(show_locals=True)

APP_NAME = "ai_server"
HOME_DIR = os.path.expanduser("~")
DEFAULT_CONFIG_DIR = os.path.join(HOME_DIR, ".config", APP_NAME)
DEFAULT_CONFIG_FILE = os.path.join(
    os.getenv("XDG_CONFIG_HOME", DEFAULT_CONFIG_DIR), "config.yaml"
)
DEFAULT_PROMPTS_FILE = os.path.join(
    os.getenv("XDG_CONFIG_HOME", DEFAULT_CONFIG_DIR), "prompts.yaml"
)

load_dotenv(os.path.join(DEFAULT_CONFIG_DIR, ".env"))

# Logging setup
console = Console(record=True, theme=common_theme)
log_console = Console(record=True, theme=log_theme)
parent_logger = logging.getLogger(APP_NAME)
log = logging.getLogger(f"{APP_NAME}.{__name__}")
rich_handler = RichHandler(
    rich_tracebacks=True,
    markup=True,
    show_path=True,
    tracebacks_show_locals=True,
    console=console,
)


def validate_server_config():
    """
    Validate server configuration.

    Returns:
        List of validation error messages
    """
    errors = []

    if not hasattr(cfg, "server"):
        errors.append("Server configuration missing")
        return errors

    # Validate listen interfaces
    if hasattr(cfg.server, "listen_interfaces"):
        # Basic validation - could be enhanced
        pass
    else:
        errors.append("Server listen_interfaces not configured")

    # Validate listen port
    if hasattr(cfg.server, "listen_port"):
        port = cfg.server.listen_port
        if not isinstance(port, int) or port < 1 or port > 65535:
            errors.append(f"Invalid server port: {port}")
    else:
        errors.append("Server listen_port not configured")

    return errors


def validate_agent_configs():
    """
    Validate agent configurations.

    Returns:
        List of validation error messages
    """
    errors = []

    if not hasattr(cfg, "agents"):
        errors.append("Agents configuration missing")
        return errors

    agent_names = [
        name for name in cfg.agents.__dict__.keys() if not name.startswith("_")
    ]

    if not agent_names:
        errors.append("No agents configured")
        return errors

    for agent_name in agent_names:
        agent = getattr(cfg.agents, agent_name)

        # Validate model
        if not hasattr(agent, "model") or not agent.model:
            errors.append(f"Agent '{agent_name}' missing model configuration")

        # Validate history config
        if hasattr(agent, "history"):
            history = agent.history
            if hasattr(history, "cut_after") and history.cut_after < 0:
                errors.append(
                    f"Agent '{agent_name}' invalid cut_after value: {history.cut_after}"
                )
            if hasattr(history, "summarize_after") and history.summarize_after < 0:
                errors.append(
                    f"Agent '{agent_name}' invalid summarize_after value: {history.summarize_after}"
                )
        else:
            errors.append(f"Agent '{agent_name}' missing history configuration")

    return errors


def validate_api_configs():
    """
    Validate API configurations.

    Returns:
        List of validation error messages
    """
    errors = []

    if not hasattr(cfg, "llm_api"):
        errors.append("LLM API configuration missing")
        return errors

    for api_name in cfg.llm_api.__dict__.keys():
        if api_name.startswith("_"):
            continue

        api_config = getattr(cfg.llm_api, api_name)

        # Validate base_url
        if not hasattr(api_config, "base_url") or not api_config.base_url:
            errors.append(f"API '{api_name}' missing base_url")

        # Note: api_key validation is handled by environment variable loading

    return errors


def validate_logging_config():
    """
    Validate logging configuration.

    Returns:
        List of validation error messages
    """
    errors = []

    if not hasattr(cfg, "logging"):
        # Logging is optional, use defaults
        return errors

    # Validate log level
    if hasattr(cfg.logging, "level"):
        level = cfg.logging.level.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level not in valid_levels:
            errors.append(f"Invalid logging level: {cfg.logging.level}")

    # Validate debug settings
    if hasattr(cfg.logging, "debug"):
        debug_config = cfg.logging.debug
        # All debug flags are optional booleans, no specific validation needed

    return errors


def validate_all_config():
    """
    Validate all configuration sections.

    Returns:
        List of all validation error messages
    """
    all_errors = []

    all_errors.extend(validate_server_config())
    all_errors.extend(validate_agent_configs())
    all_errors.extend(validate_api_configs())
    all_errors.extend(validate_logging_config())

    return all_errors


def _init_logs():
    """Initialize logging configuration.

    Checks if required config dataclasses exist before accessing them.
    Falls back to default values if they don't.
    Sets up structured logging with console output.
    """
    # Check if logging config exists
    if not hasattr(cfg, "logging"):
        log.warning("No logging configuration found, using defaults")
        _setup_default_logging()
        return

    # Check and set suppressed loggers
    if (
        hasattr(cfg.logging, "loggers")
        and hasattr(cfg.logging.loggers, "suppress")
        and hasattr(cfg.logging.loggers, "suppress_level")
    ):
        for logger_name in cfg.logging.loggers.suppress:
            logging.getLogger(logger_name).setLevel(
                getattr(logging, cfg.logging.loggers.suppress_level.upper())
            )
    else:
        log.debug("No logger suppression configuration found")

    # Check and set root logger level
    if hasattr(cfg.logging, "level"):
        try:
            level = cfg.logging.level.upper()
            parent_logger.setLevel(level)

            # Setup console handler with better formatting
            _setup_console_logging(level)

            if level == "DEBUG":
                cfg.print_config()
        except (ValueError, AttributeError) as e:
            log.warning(f"Invalid logging level configuration: {e}")
            parent_logger.setLevel(logging.INFO)
            _setup_console_logging("INFO")
    else:
        log.warning("No logging level configuration found, using INFO")
        parent_logger.setLevel(logging.INFO)
        _setup_console_logging("INFO")


def _setup_default_logging():
    """Setup default logging configuration."""
    parent_logger.setLevel(logging.INFO)
    _setup_console_logging("INFO")


def _setup_console_logging(level: str):
    """Setup console logging with structured format."""
    # Remove existing handlers to avoid duplicates
    for handler in parent_logger.handlers[:]:
        parent_logger.removeHandler(handler)

    # Create console handler
    console_handler = RichHandler(
        rich_tracebacks=(
            getattr(cfg.logging, "rich_tracebacks", True)
            if hasattr(cfg, "logging")
            else True
        ),
        show_time=(
            getattr(cfg.logging, "show_time", True) if hasattr(cfg, "logging") else True
        ),
        show_path=(
            getattr(cfg.logging, "show_path", False)
            if hasattr(cfg, "logging")
            else False
        ),
        markup=(
            getattr(cfg.logging, "markup", True) if hasattr(cfg, "logging") else True
        ),
    )

    # Set formatter
    formatter = logging.Formatter(
        fmt=(
            getattr(cfg.logging, "format", "%(message)s")
            if hasattr(cfg, "logging")
            else "%(message)s"
        ),
        datefmt=(
            getattr(cfg.logging, "date_format", "%X")
            if hasattr(cfg, "logging")
            else "%X"
        ),
    )
    console_handler.setFormatter(formatter)

    # Add handler to parent logger
    parent_logger.addHandler(console_handler)


def _parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace and unknown arguments list
    """
    log.info("Parsing args")
    parser = argparse.ArgumentParser(
        description="AI Server" f"Default values are read from {DEFAULT_CONFIG_DIR}"
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default=DEFAULT_CONFIG_FILE,
        help="Configuration file location.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        dest="prompts_file",
        default=DEFAULT_PROMPTS_FILE,
        help="Prompts file location.",
    )

    args, unknown = parser.parse_known_args()
    return args, unknown


def _update_env_values():
    """Update all config values that are set to '.env' with corresponding environment variables.

    Traverses the config structure and updates any '.env' values with their corresponding
    environment variable values. Environment variable names are constructed from the last
    two levels of the config hierarchy (e.g., llm_api.aihubmix.api_key -> AIHUBMIX_API_KEY)
    """

    def _process_dict(d, path=[]):
        if not isinstance(d, dict):
            return

        for k, v in d.items():
            current_path = path + [k]

            if isinstance(v, dict):
                _process_dict(v, current_path)
            elif v == ".env":
                # Take last two parts of the path for env var name
                env_parts = (
                    current_path[-2:] if len(current_path) >= 2 else current_path
                )
                env_var = "_".join(env_parts).upper()
                env_value = os.getenv(env_var)

                if env_value is None:
                    log.warning(
                        f"Environment variable {env_var} not found for config path: {'.'.join(current_path)}"
                    )
                    continue

                # Update config with environment variable value
                cfg.update(".".join(current_path), env_value)
                log.debug(
                    f"Updated config {'.'.join(current_path)} from environment variable {env_var}"
                )

    # Process config starting from root data dictionary
    _process_dict(cfg.data)


# Load config and compile patterns
args, unknown = _parse_args()
cfg.load_files([args.config_file, args.prompts_file])
cfg.load_args(unknown)

# Load and validate configuration
validation_errors = validate_all_config()
if validation_errors:
    log.error("Configuration validation failed:")
    for error in validation_errors:
        log.error(f"  - {error}")
    sys.exit(1)

# Update all .env values with environment variables
_update_env_values()

# Ensure all agents have proxy field
# _ensure_model_proxies()

cfg.update("runtime.console", console)
cfg.update("runtime.log_console", log_console)


_init_logs()

if __name__ == "__main__":
    # cfg.print_config()
    sys.exit(0)
