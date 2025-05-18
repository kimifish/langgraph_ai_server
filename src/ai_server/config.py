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

APP_NAME = 'ai_server'
HOME_DIR = os.path.expanduser("~")
DEFAULT_CONFIG_DIR = os.path.join(HOME_DIR, ".config", APP_NAME)
DEFAULT_CONFIG_FILE = os.path.join(
    os.getenv("XDG_CONFIG_HOME", DEFAULT_CONFIG_DIR), 
    "config.yaml")
DEFAULT_PROMPTS_FILE = os.path.join(
    os.getenv("XDG_CONFIG_HOME", DEFAULT_CONFIG_DIR), 
    "prompts.yaml")

load_dotenv(os.path.join(DEFAULT_CONFIG_DIR, ".env"))

# Logging setup
console = Console(record=True, theme=common_theme)
log_console = Console(record=True, theme=log_theme)
logging.basicConfig(
    level=logging.NOTSET,
    format="%(message)s",
    datefmt="%X",
    handlers=[RichHandler(console=log_console, markup=True)],
)
parent_logger = logging.getLogger(APP_NAME)
log = logging.getLogger(f'{APP_NAME}.{__name__}')
rich_handler = RichHandler(rich_tracebacks=True,
                           markup=True,
                           show_path=True,
                           tracebacks_show_locals=True,
                           console=console)


def _init_logs():
    """Initialize logging configuration.
    
    Checks if required config dataclasses exist before accessing them.
    Falls back to default values if they don't.
    """
    # Check if logging config exists
    if not hasattr(cfg, 'logging'):
        log.warning("No logging configuration found, using defaults")
        return

    # Check and set suppressed loggers
    if (hasattr(cfg.logging, 'loggers') and 
        hasattr(cfg.logging.loggers, 'suppress') and 
        hasattr(cfg.logging.loggers, 'suppress_level')):
        for logger_name in cfg.logging.loggers.suppress:
            logging.getLogger(logger_name).setLevel(
                getattr(logging, cfg.logging.loggers.suppress_level.upper())
            )
    else:
        log.debug("No logger suppression configuration found")

    # Check and set root logger level
    if hasattr(cfg.logging, 'level'):
        try:
            parent_logger.setLevel(cfg.logging.level.upper())
            if cfg.logging.level.upper() == "DEBUG":
                cfg.print_config()
        except (ValueError, AttributeError) as e:
            log.warning(f"Invalid logging level configuration: {e}")
            parent_logger.setLevel(logging.INFO)
    else:
        log.warning("No logging level configuration found, using INFO")
        parent_logger.setLevel(logging.INFO)


def _parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace and unknown arguments list
    """
    log.info("Parsing args")
    parser = argparse.ArgumentParser(
        description='AI Server'
        f'Default values are read from {DEFAULT_CONFIG_DIR}'
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
            elif v == '.env':
                # Take last two parts of the path for env var name
                env_parts = current_path[-2:] if len(current_path) >= 2 else current_path
                env_var = '_'.join(env_parts).upper()
                env_value = os.getenv(env_var)
                
                if env_value is None:
                    log.warning(f"Environment variable {env_var} not found for config path: {'.'.join(current_path)}")
                    continue
                    
                # Update config with environment variable value
                cfg.update('.'.join(current_path), env_value)
                log.debug(f"Updated config {'.'.join(current_path)} from environment variable {env_var}")

    # Process config starting from root data dictionary
    _process_dict(cfg.data)


# Load config and compile patterns
args, unknown = _parse_args()
cfg.load_files([args.config_file, args.prompts_file])
cfg.load_args(unknown)

cfg.validate_config([
    'server.listen_interfaces',
    'server.listen_port',

    'agents.%.model',
    'agents.%.history',

    ('agents.%.resources', []),
    ('agents.%.streaming', False),
    ('agents.%.temperature', None),
    ('agents.%.proxy', ""),
    ('agents.%.history.cut_after', 10),
    ('agents.%.history.summarize_after', 0),
    ('agents.%.history.use_common', False),
    ('agents.%.history.post_to_common', False),

    ('endpoints.auto', 'ai'),

    'llm_api.%.base_url',
    ('llm_api.%.api_key', '.env'),

    ('logging.debug.events', False),
    ('logging.debug.prompts', False),
    ('logging.debug.messages_diff', False),
    ('logging.debug.llm_init', False),
])

# Update all .env values with environment variables
_update_env_values()

# Ensure all agents have proxy field
# _ensure_model_proxies()

cfg.update('runtime.console', console)
cfg.update("runtime.log_console", log_console)

_init_logs()

if __name__ == '__main__':
    # cfg.print_config()
    sys.exit(0)
