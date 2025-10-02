"""
Unit tests for service layer modules.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from ai_server.services.conversation_service import (
    ConversationService,
)
from ai_server.services.llm_service import LLMService
from ai_server.services.tool_service import ToolService
from ai_server.services.user_service import UserService
from ai_server.models.userconfs import UserConf


@pytest.fixture
def mock_graph():
    """Mock graph for testing."""
    return MagicMock()


@pytest.fixture
def mock_userconf():
    """Mock user configuration."""
    return UserConf(
        thread_id=RunnableConfig(configurable={"thread_id": "test_thread"}),
        user="test_user",
        location="test_location",
        additional_instructions="test_instructions",
        llm_to_use="test_agent",
        last_used_llm="test_agent",
    )


class TestConversationService:
    """Test ConversationService class."""

    def test_init(self, mock_graph):
        """Test ConversationService initialization."""
        service = ConversationService(mock_graph)
        assert service.graph == mock_graph

    def test_prepare_messages(self):
        """Test message preparation."""
        service = ConversationService(MagicMock())
        messages = service.prepare_messages("Hello world")
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello world"

    def test_prepare_graph_init_values(self, mock_userconf):
        """Test graph initial values preparation."""
        service = ConversationService(MagicMock())
        messages = [HumanMessage(content="Test")]
        result = service.prepare_graph_init_values(messages, mock_userconf)

        assert result["user"] == "test_user"
        assert result["location"] == "test_location"
        assert result["additional_instructions"] == "test_instructions"
        assert result["llm_to_use"] == "test_agent"
        assert result["last_used_llm"] == "test_agent"
        assert result["thread_id"] == mock_userconf.thread_id
        assert result["messages"]["test_agent"] == messages

    @pytest.mark.asyncio
    async def test_execute_graph_stream(self, mock_graph, mock_userconf):
        """Test graph stream execution."""
        service = ConversationService(mock_graph)
        init_values = {"test": "value"}
        config = RunnableConfig(configurable={"thread_id": "test_123"})

        # Mock async stream to return async iterator
        async def mock_astream(*args, **kwargs):
            yield {"event": "test"}

        mock_graph.astream = mock_astream

        events = []
        async for event in service.execute_graph_stream(init_values, config):
            events.append(event)

        assert len(events) == 1
        assert events[0] == {"event": "test"}

    def test_process_stream_event(self, mock_userconf):
        """Test stream event processing."""
        service = ConversationService(MagicMock())
        event = {
            "messages": {"test_agent": [AIMessage(content="Response")]},
            "llm_to_use": "test_agent",
        }
        result = service.process_stream_event(event, mock_userconf)

        assert result["next_state"] == "END"
        assert result["answer"] == "Response"


class TestLLMService:
    """Test LLMService class."""

    def test_init(self):
        """Test LLMService initialization."""
        service = LLMService()
        assert service._llm_cache == {}

    def test_get_llm_cached(self):
        """Test getting cached LLM."""
        service = LLMService()
        mock_llm = MagicMock()
        service._llm_cache["gpt-4_0.7_False_None"] = mock_llm

        result = service.get_llm("gpt-4", temperature=0.7)
        assert result == mock_llm

    def test_get_llm_new(self):
        """Test getting new LLM."""
        service = LLMService()
        mock_llm = MagicMock()

        with patch.object(service, "_create_llm") as mock_create:
            mock_create.return_value = mock_llm

            result = service.get_llm("gpt-4", temperature=0.7, streaming=True)

            assert result == mock_llm
            mock_create.assert_called_once_with("gpt-4", 0.7, True, None)
            assert "gpt-4_0.7_True_None" in service._llm_cache

    def test_get_llm_none(self):
        """Test getting None for unsupported model."""
        service = LLMService()

        with patch.object(service, "_create_llm") as mock_create:
            mock_create.return_value = None

            result = service.get_llm("unsupported-model")
            assert result is None


class TestToolService:
    """Test ToolService class."""

    def test_init(self):
        """Test ToolService initialization."""
        service = ToolService()
        assert service._tool_registry == {}
        assert service._mcp_client is None

    def test_register_tools(self):
        """Test tool registration."""
        service = ToolService()
        tools = [MagicMock(), MagicMock()]

        service.register_tools("test_agent", tools)

        assert "test_agent" in service._tool_registry
        assert len(service._tool_registry["test_agent"]) == 2

    def test_register_tools_extend(self):
        """Test extending existing tool registry."""
        service = ToolService()
        tools1 = [MagicMock()]
        tools2 = [MagicMock(), MagicMock()]

        service.register_tools("test_agent", tools1)
        service.register_tools("test_agent", tools2)

        assert len(service._tool_registry["test_agent"]) == 3

    def test_get_tools_existing(self):
        """Test getting tools for existing agent."""
        service = ToolService()
        tools = [MagicMock(), MagicMock()]
        service._tool_registry["test_agent"] = tools

        result = service.get_tools("test_agent")
        assert result == tools

    def test_get_tools_nonexistent(self):
        """Test getting tools for nonexistent agent."""
        service = ToolService()
        result = service.get_tools("nonexistent")
        assert result == []

    def test_initialize_static_tools(self):
        """Test static tools initialization."""
        service = ToolService()
        # Currently just logs that it's not implemented
        service.initialize_static_tools()
        # No assertions needed for now


class TestUserService:
    """Test UserService class."""

    def test_init(self):
        """Test UserService initialization."""
        service = UserService()
        assert isinstance(service._user_confs, object)  # UserConfs instance

    def test_get_user_config_existing(self):
        """Test getting existing user config."""
        service = UserService()

        # Mock the UserConfs.get method
        with patch.object(service._user_confs, "get") as mock_get:
            mock_config = MagicMock()
            mock_get.return_value = mock_config

            result = service.get_user_config("test_thread")
            assert result == mock_config
            mock_get.assert_called_once_with("test_thread")

    def test_get_user_config_nonexistent(self):
        """Test getting nonexistent user config."""
        service = UserService()

        with patch.object(service._user_confs, "get") as mock_get:
            mock_get.return_value = None

            result = service.get_user_config("nonexistent")
            assert result is None

    def test_create_user_config(self):
        """Test user config creation."""
        service = UserService()

        with patch.object(service._user_confs, "add") as mock_add:
            mock_config = MagicMock()
            mock_add.return_value = mock_config

            result = service.create_user_config(
                thread_id="test_thread",
                user="test_user",
                location="test_location",
                additional_instructions="test_instructions",
                llm_to_use="test_agent",
            )

            assert result == mock_config
            mock_add.assert_called_once()

    def test_get_all_user_configs(self):
        """Test getting all user configs."""
        service = UserService()

        with patch.object(service._user_confs, "get_all") as mock_get_all:
            mock_configs = {"config1": "value1", "config2": "value2"}
            mock_get_all.return_value = mock_configs

            result = service.get_all_user_configs()
            assert result == mock_configs
            mock_get_all.assert_called_once()
