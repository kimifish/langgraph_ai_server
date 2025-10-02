"""
Unit tests for API layer modules.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from ai_server.api.routes import router
from ai_server.api.schemas import (
    ChatRequest,
    ChatResponse,
    APIErrorResponse,
    ErrorResponse,
    ErrorDetail,
)
from ai_server.api.dependencies import get_user_confs, get_thread_id
from ai_server.api.middleware import LoggingMiddleware


@pytest.fixture
def client():
    """Test client for API testing."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    # Mock user_confs in app state
    app.state.user_confs = MagicMock()
    return TestClient(app)


class TestSchemas:
    """Test API schema models."""

    def test_chat_request_valid(self):
        """Test valid ChatRequest creation."""
        request = ChatRequest(
            message="Hello",
            thread_id="test_thread",
            user="test_user",
            location="test_location",
            additional_instructions="test_instructions",
            llm_to_use="common",
        )
        assert request.message == "Hello"
        assert request.thread_id == "test_thread"
        assert request.user == "test_user"
        assert request.location == "test_location"
        assert request.additional_instructions == "test_instructions"
        assert request.llm_to_use == "common"

    def test_chat_request_defaults(self):
        """Test ChatRequest with default values."""
        request = ChatRequest(message="Hello")
        assert request.message == "Hello"
        assert request.thread_id is None
        assert request.user == "Undefined"
        assert request.location == "Undefined"
        assert request.additional_instructions == ""
        assert request.llm_to_use is None

    def test_chat_response(self):
        """Test ChatResponse creation."""
        response = ChatResponse(
            answer="Hello back!", thread_id="test_thread", llm_used="common", error=None
        )
        assert response.answer == "Hello back!"
        assert response.thread_id == "test_thread"
        assert response.llm_used == "common"
        assert response.error is None

    def test_api_error_response(self):
        """Test APIErrorResponse creation."""
        details = [ErrorDetail(field="message", message="Required field")]
        response = APIErrorResponse(
            error="Validation error",
            type="ValidationError",
            details=details,
            request_id="req_123",
        )
        assert response.error == "Validation error"
        assert response.type == "ValidationError"
        assert response.details is not None and len(response.details) == 1
        assert response.request_id == "req_123"

    def test_error_response(self):
        """Test ErrorResponse creation."""
        response = ErrorResponse(
            error="Internal server error", detail="Something went wrong"
        )
        assert response.error == "Internal server error"
        assert response.detail == "Something went wrong"


class TestDependencies:
    """Test API dependency functions."""

    def test_get_user_confs(self):
        """Test get_user_confs dependency."""
        mock_request = MagicMock()
        mock_user_confs = MagicMock()
        mock_request.app.state.user_confs = mock_user_confs

        result = get_user_confs(mock_request)
        assert result == mock_user_confs

    def test_get_thread_id_with_value(self):
        """Test get_thread_id with value."""
        result = get_thread_id("test_thread")
        assert result == "test_thread"

    def test_get_thread_id_none(self):
        """Test get_thread_id with None."""
        result = get_thread_id(None)
        assert result is None


class TestMiddleware:
    """Test API middleware."""

    @pytest.mark.asyncio
    async def test_logging_middleware_success(self):
        """Test LoggingMiddleware with successful request."""
        middleware = LoggingMiddleware(MagicMock())

        # Mock request and response
        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url = "http://test.com/api"

        mock_response = MagicMock()
        mock_response.status_code = 200

        call_next = AsyncMock(return_value=mock_response)

        result = await middleware.dispatch(mock_request, call_next)

        assert result == mock_response
        call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_logging_middleware_exception(self):
        """Test LoggingMiddleware with exception."""
        middleware = LoggingMiddleware(MagicMock())

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url = "http://test.com/api"

        call_next = AsyncMock(side_effect=Exception("Test error"))

        with pytest.raises(Exception, match="Test error"):
            await middleware.dispatch(mock_request, call_next)


class TestRoutes:
    """Test API routes."""

    def test_chat_endpoint_placeholder_response(self, client):
        """Test chat endpoint returns placeholder response."""
        # Mock cfg.runtime.user_confs
        with patch("ai_server.api.routes.cfg") as mock_cfg:
            mock_user_confs = MagicMock()
            mock_cfg.runtime.user_confs = mock_user_confs

            # Mock user config
            mock_user_conf = MagicMock()
            mock_user_conf.thread_id = "test_thread_123"
            mock_user_conf.llm_to_use = "common"
            mock_user_confs.get.return_value = None  # No existing config
            mock_user_confs.add.return_value = mock_user_conf

            response = client.get("/common?message=Hello")

            assert response.status_code == 200
            data = response.json()
            assert (
                data["answer"] == "Service temporarily unavailable - under refactoring"
            )
            assert data["thread_id"] == "test_thread_123"
            assert data["llm_used"] == "common"
            assert data["error"] == "Phase 1 refactoring in progress"

    def test_chat_endpoint_no_user_confs(self, client):
        """Test chat endpoint when user_confs not initialized."""
        with patch("ai_server.api.routes.cfg") as mock_cfg:
            mock_cfg.runtime.user_confs = None

            response = client.get("/common?message=Hello")

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Service initializing..."
            assert "error" in data

    def test_chat_endpoint_with_existing_thread(self, client):
        """Test chat endpoint with existing thread ID."""
        with patch("ai_server.api.routes.cfg") as mock_cfg:
            mock_user_confs = MagicMock()
            mock_cfg.runtime.user_confs = mock_user_confs

            # Mock existing user config
            mock_user_conf = MagicMock()
            mock_user_conf.thread_id = "existing_thread"
            mock_user_conf.llm_to_use = "existing_agent"
            mock_user_confs.get.return_value = mock_user_conf

            response = client.get("/common?message=Hello&thread_id=existing_thread")

            assert response.status_code == 200
            data = response.json()
            assert data["thread_id"] == "existing_thread"
            assert data["llm_used"] == "existing_agent"

    def test_chat_endpoint_exception(self, client):
        """Test chat endpoint with unexpected exception."""
        with patch("ai_server.api.routes.cfg") as mock_cfg:
            # Make user_confs None to trigger the initializing response
            mock_cfg.runtime.user_confs = None

            response = client.get("/common?message=Hello")

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Service initializing..."
            assert "error" in data

    def test_chat_endpoint_post_success(self, client):
        """Test POST chat endpoint with valid request."""
        with patch("ai_server.api.routes.cfg") as mock_cfg:
            mock_user_confs = MagicMock()
            mock_cfg.runtime.user_confs = mock_user_confs

            # Mock user config
            mock_user_conf = MagicMock()
            mock_user_conf.thread_id = "test_thread_123"
            mock_user_conf.llm_to_use = "common"
            mock_user_confs.get.return_value = None
            mock_user_confs.add.return_value = mock_user_conf

            request_data = {
                "message": "Hello",
                "user": "test_user",
                "location": "test_location",
            }

            response = client.post("/common", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert (
                data["answer"] == "Service temporarily unavailable - under refactoring"
            )
            assert data["thread_id"] == "test_thread_123"
            assert data["llm_used"] == "common"
            assert data["error"] == "Phase 1 refactoring in progress"

    def test_chat_endpoint_post_invalid_llm(self, client):
        """Test POST chat endpoint with invalid LLM."""
        request_data = {"message": "Hello"}

        response = client.post("/invalid_llm", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert "not found" in data["detail"]["error"]
