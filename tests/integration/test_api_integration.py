"""
Integration tests for API endpoints with mocked services.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from ai_server.api.routes import router
from ai_server.models.userconfs import UserConfs, UserConf
from ai_server.services.conversation_service import ConversationService


@pytest.fixture
def test_app():
    """Create test FastAPI app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)

    # Mock the config runtime to have user_confs
    with patch("ai_server.api.routes.cfg") as mock_cfg:
        mock_cfg.runtime.user_confs = UserConfs()
        yield app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test basic health check returns ok status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "phase" in data


class TestChatEndpoint:
    """Test chat endpoint with mocked services."""

    @patch("ai_server.api.routes.cfg")
    def test_chat_endpoint_initialization_message(self, mock_cfg, client):
        """Test chat endpoint returns initialization message when user_confs not ready."""
        # Mock cfg to not have user_confs
        mock_cfg.runtime = MagicMock()
        del mock_cfg.runtime.user_confs  # Simulate not initialized

        response = client.get("/ai?message=Hello")

        assert response.status_code == 200
        data = response.json()
        assert "Service initializing" in data["answer"]
        assert "error" in data

    @patch("ai_server.api.routes.cfg")
    def test_chat_endpoint_with_user_confs(self, mock_cfg, client):
        """Test chat endpoint with user configurations available."""
        # Setup mocked user_confs
        user_confs = UserConfs()
        mock_cfg.runtime.user_confs = user_confs
        mock_cfg.agents = MagicMock()  # Mock agents for LLM validation

        response = client.get("/ai?message=Hello&user=test_user&thread_id=test_123")

        assert response.status_code == 200
        data = response.json()
        assert "Service temporarily unavailable" in data["answer"]
        assert "refactoring" in data["error"]
        assert data["thread_id"] == "test_123"
        assert data["llm_used"] == "ai"

    @patch("ai_server.api.routes.cfg")
    def test_chat_endpoint_creates_user_conf(self, mock_cfg, client):
        """Test that chat endpoint creates user configuration when none exists."""
        user_confs = UserConfs()
        mock_cfg.runtime.user_confs = user_confs
        mock_cfg.agents = MagicMock()

        response = client.get(
            "/common?message=Hello&user=new_user&location=test_location"
        )

        assert response.status_code == 200
        data = response.json()

        # Check that user conf was created
        created_conf = user_confs.get(data["thread_id"])
        assert created_conf is not None
        assert created_conf.user == "new_user"
        assert created_conf.location == "test_location"
        assert created_conf.llm_to_use == "common"

    @patch("ai_server.api.routes.cfg")
    def test_chat_endpoint_with_additional_instructions(self, mock_cfg, client):
        """Test chat endpoint with additional instructions parameter."""
        user_confs = UserConfs()
        mock_cfg.runtime.user_confs = user_confs
        mock_cfg.agents = MagicMock()

        instructions = "Please be helpful and concise"
        response = client.get(
            f"/ai?message=Hello&additional_instructions={instructions}"
        )

        assert response.status_code == 200
        data = response.json()

        # Check that instructions were stored
        created_conf = user_confs.get(data["thread_id"])
        assert created_conf is not None
        assert created_conf.additional_instructions == instructions

    @patch("ai_server.api.routes.cfg")
    def test_chat_endpoint_with_llm_to_use(self, mock_cfg, client):
        """Test chat endpoint with specific LLM selection."""
        user_confs = UserConfs()
        mock_cfg.runtime.user_confs = user_confs
        mock_cfg.agents = MagicMock()

        response = client.get("/ai?message=Hello&llm_to_use=gpt-4")

        assert response.status_code == 200
        data = response.json()

        # Check that LLM was set correctly
        created_conf = user_confs.get(data["thread_id"])
        assert created_conf is not None
        assert created_conf.llm_to_use == "gpt-4"

    def test_chat_endpoint_invalid_method(self, client):
        """Test that POST method is not allowed on chat endpoint."""
        response = client.post("/ai", json={"message": "Hello"})

        assert response.status_code == 405  # Method Not Allowed

    @patch("ai_server.api.routes.cfg")
    def test_chat_endpoint_error_handling(self, mock_cfg, client):
        """Test error handling in chat endpoint."""
        # Mock cfg to raise an exception
        mock_cfg.runtime.user_confs = MagicMock()
        mock_cfg.runtime.user_confs.add.side_effect = Exception("Test error")

        response = client.get("/ai?message=Hello")

        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]


class TestConversationServiceIntegration:
    """Test ConversationService integration (when implemented)."""

    @patch(
        "ai_server.services.conversation_service.ConversationService.execute_graph_stream"
    )
    @pytest.mark.asyncio
    async def test_conversation_service_with_mock_graph(self, mock_stream):
        """Test conversation service with mocked graph execution."""
        # This test will be expanded when the full conversation flow is implemented
        # from ai_server.graph import graph  # Mock this when available

        # Mock graph
        mock_graph = MagicMock()
        service = ConversationService(mock_graph)

        # Mock stream events
        mock_event = {
            "messages": {"common": [MagicMock(content="AI response")]},
            "llm_to_use": "common",
            "path": [["start", "end"]],
        }
        mock_stream.return_value = [mock_event]

        # Mock user conf
        from langchain_core.runnables import RunnableConfig

        user_conf = UserConf(
            thread_id=RunnableConfig(configurable={"thread_id": "test_123"}),
            user="test_user",
        )

        # This will need to be updated when the actual conversation flow is connected
        # result = await service.execute_conversation("Hello", user_conf)
        # assert "answer" in result
        # assert result["thread_id"] == "test_123"

        # For now, just test the preparation methods
        messages = service.prepare_messages("Hello")
        assert len(messages) == 1
        assert messages[0].content == "Hello"

        init_values = service.prepare_graph_init_values(messages, user_conf)
        assert init_values["user"] == "test_user"
        assert init_values["thread_id"] == user_conf.thread_id
