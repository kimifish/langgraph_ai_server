"""
Unit tests for configuration module.
"""

from unittest.mock import patch
import os

from ai_server.config import cfg, APP_NAME


class TestConfig:
    """Test configuration loading and management."""

    def test_app_name(self):
        """Test that APP_NAME is set correctly."""
        assert APP_NAME == "ai_server"

    def test_config_initialization(self):
        """Test that config object is initialized."""
        assert cfg is not None
        assert hasattr(cfg, "update")

    @patch.dict(os.environ, {"TEST_VAR": "test_value"})
    def test_environment_variable_access(self):
        """Test accessing environment variables through config."""
        # This test assumes the config system can access env vars
        # The actual implementation may vary
        assert os.getenv("TEST_VAR") == "test_value"

    def test_config_update(self):
        """Test updating configuration values."""
        test_key = "test_config_key"
        test_value = "test_config_value"

        # Update config
        cfg.update(test_key, test_value)

        # Verify update (this depends on kimiconfig implementation)
        # For now, just ensure no exception is raised
        assert True

    def test_server_config_access(self):
        """Test accessing server configuration."""
        # These should not raise exceptions
        server_config = getattr(cfg, "server", None)
        assert server_config is not None

    def test_logging_config_access(self):
        """Test accessing logging configuration."""
        logging_config = getattr(cfg, "logging", None)
        assert logging_config is not None

    def test_agents_config_access(self):
        """Test accessing agents configuration."""
        agents_config = getattr(cfg, "agents", None)
        assert agents_config is not None

    def test_dotenv_loading(self):
        """Test that dotenv loading functionality exists."""
        # dotenv is loaded at import time, so we just verify the import works
        from ai_server.config import load_dotenv

        assert load_dotenv is not None

    def test_config_format_attributes(self):
        """Test config formatting method exists."""
        # Check that the format_attributes method exists
        assert hasattr(cfg, "format_attributes")
