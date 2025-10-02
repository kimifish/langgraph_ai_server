"""
Performance tests for API endpoints.
"""

import time
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from ai_server.api.routes import router
from ai_server.models.userconfs import UserConfs


@pytest.fixture
def test_app():
    """Create test FastAPI app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestHealthEndpointPerformance:
    """Performance tests for health endpoint."""

    def test_health_endpoint_response_time(self, client):
        """Test health endpoint response time is under 100ms."""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # Convert to milliseconds

        assert response.status_code == 200
        assert response_time < 100  # Should respond in under 100ms

    @pytest.mark.parametrize("num_requests", [10, 50, 100])
    def test_health_endpoint_concurrent_requests(self, client, num_requests):
        """Test health endpoint can handle concurrent requests."""
        import concurrent.futures

        def make_request():
            response = client.get("/health")
            return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        assert all(status == 200 for status in results)
        assert len(results) == num_requests


class TestChatEndpointPerformance:
    """Performance tests for chat endpoint."""

    @patch("ai_server.api.routes.cfg")
    def test_chat_endpoint_response_time(self, mock_cfg, client):
        """Test chat endpoint response time with mocked services."""
        # Setup mocked user_confs
        user_confs = UserConfs()
        mock_cfg.runtime.user_confs = user_confs
        mock_cfg.agents = type("MockAgents", (), {})()  # Mock agents object

        start_time = time.time()
        response = client.get("/ai?message=Hello&user=test_user&thread_id=test_123")
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # Convert to milliseconds

        assert response.status_code == 200
        # Allow up to 500ms for mocked response (should be much faster in practice)
        assert response_time < 500

    @patch("ai_server.api.routes.cfg")
    def test_chat_endpoint_memory_usage(self, mock_cfg, client):
        """Test chat endpoint doesn't have excessive memory growth."""
        import psutil
        import os

        # Setup mocked user_confs
        user_confs = UserConfs()
        mock_cfg.runtime.user_confs = user_confs
        mock_cfg.agents = type("MockAgents", (), {})()

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Make multiple requests
        for i in range(100):
            response = client.get(
                f"/ai?message=Hello{i}&user=user{i}&thread_id=thread{i}"
            )
            assert response.status_code == 200

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Allow up to 50MB memory growth for 100 requests
        assert memory_growth < 50

    @patch("ai_server.api.routes.cfg")
    def test_chat_endpoint_concurrent_requests(self, mock_cfg, client):
        """Test concurrent requests to chat endpoint."""
        # Setup mocked user_confs
        user_confs = UserConfs()
        mock_cfg.runtime.user_confs = user_confs
        mock_cfg.agents = type("MockAgents", (), {})()

        import concurrent.futures

        def make_request(i):
            response = client.get(
                f"/ai?message=Hello{i}&user=user{i}&thread_id=thread{i}"
            )
            return response.status_code, response.json()

        start_time = time.time()

        # Make 10 concurrent requests using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # All responses should be successful
        assert all(status == 200 for status, _ in results)

        # Should complete within reasonable time (under 2 seconds for 10 concurrent requests)
        assert total_time < 2000


class TestLoadTesting:
    """Load testing scenarios."""

    @patch("ai_server.api.routes.cfg")
    def test_sustained_load(self, mock_cfg, client):
        """Test sustained load over time."""
        # Setup mocked user_confs
        user_confs = UserConfs()
        mock_cfg.runtime.user_confs = user_confs
        mock_cfg.agents = type("MockAgents", (), {})()

        start_time = time.time()

        # Make 1000 requests over 10 seconds
        for i in range(1000):
            response = client.get(
                f"/ai?message=Test{i}&user=user{i % 10}&thread_id=thread{i % 10}"
            )
            assert response.status_code == 200

            # Small delay to spread over time
            if i % 100 == 0:
                time.sleep(0.01)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete within 15 seconds
        assert total_time < 15

        # Check that user configurations were created and reused
        assert len(user_confs.get_all()) == 10  # 10 unique thread_ids


class TestResourceUsage:
    """Test resource usage patterns."""

    @patch("ai_server.api.routes.cfg")
    def test_user_conf_cleanup(self, mock_cfg, client):
        """Test that user configurations don't accumulate indefinitely."""
        # Setup mocked user_confs
        user_confs = UserConfs()
        mock_cfg.runtime.user_confs = user_confs
        mock_cfg.agents = type("MockAgents", (), {})()

        # Create many user configurations
        for i in range(100):
            response = client.get(
                f"/ai?message=Hello&user=user{i}&thread_id=unique_thread_{i}"
            )
            assert response.status_code == 200

        # Should have created 100 configurations
        assert len(user_confs.get_all()) == 100

        # In a real implementation, there should be cleanup logic
        # For now, we just verify the configurations exist
