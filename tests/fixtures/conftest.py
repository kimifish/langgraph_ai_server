"""
Shared fixtures and configuration for AI Server tests.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import tempfile
from pathlib import Path
from langchain_core.runnables import RunnableConfig

from ai_server.models.userconfs import UserConf


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_config_dir():
    """Create a temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = MagicMock()
    config.agents = MagicMock()
    config.agents.__dict__ = {
        "common": MagicMock(),
        "smarthome": MagicMock(),
        "music_assistant": MagicMock(),
    }
    config.agents.common.model = "gpt-4"
    config.agents.common.temperature = 0.7
    config.agents.smarthome.model = "claude-3"
    config.agents.music_assistant.model = "gpt-3.5-turbo"

    config.logging = MagicMock()
    config.logging.level = "DEBUG"
    config.logging.debug = MagicMock()
    config.logging.debug.llm_init = False
    config.logging.debug.prompts = False
    config.logging.debug.events = True

    config.server = MagicMock()
    config.server.listen_interfaces = "127.0.0.1"
    config.server.listen_port = 8000

    return config


@pytest.fixture
def sample_user_conf():
    """Sample user configuration for testing."""
    return UserConf(
        thread_id=RunnableConfig(configurable={"thread_id": "test_thread_123"}),
        user="test_user",
        location="test_location",
        additional_instructions="Test instructions",
        llm_to_use="common",
        last_used_llm="common",
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    response = MagicMock()
    response.content = "Test response from LLM"
    response.tool_calls = None
    return response


@pytest.fixture
def mock_async_llm():
    """Mock async LLM for testing."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Mocked response"))
    return llm


@pytest.fixture
def mock_graph():
    """Mock LangGraph for testing."""
    graph = MagicMock()
    graph.astream = AsyncMock()
    graph.get_state = MagicMock()
    return graph


@pytest.fixture
def mock_mcp_client():
    """Mock MCP client for testing."""
    client = MagicMock()
    client.server_name_to_tools = {}
    return client


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global configuration between tests."""
    # For now, just yield without resetting
    # TODO: Implement proper config reset when needed
    yield


@pytest.fixture
def test_config_file(temp_config_dir):
    """Create a test configuration file."""
    config_content = """
server:
  listen_interfaces: 127.0.0.1
  listen_port: 8000

agents:
  common:
    model: gpt-4
    temperature: 0.7
  smarthome:
    model: claude-3
    temperature: 0.5

logging:
  level: DEBUG
"""
    config_file = temp_config_dir / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
