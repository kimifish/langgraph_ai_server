#!/usr/bin/env python3
"""Test script for Phase 1 changes."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_api_schemas():
    """Test API schemas."""
    try:
        from ai_server.api.schemas import ChatRequest

        request = ChatRequest(message="Hello", user="test")
        print("✓ API schemas work")
        return True
    except Exception as e:
        print(f"✗ API schemas failed: {e}")
        return False


def test_services():
    """Test service classes."""
    try:
        from ai_server.services.llm_service import LLMService
        from ai_server.services.tool_service import ToolService
        from ai_server.services.user_service import UserService

        # Test service instantiation
        llm_service = LLMService()
        tool_service = ToolService()
        user_service = UserService()

        print("✓ Services instantiate correctly")
        return True
    except Exception as e:
        print(f"✗ Services failed: {e}")
        return False


def test_config_models():
    """Test configuration models."""
    try:
        from ai_server.config_models import AppConfig

        config = AppConfig()
        print(f"✓ Config models work - server port: {config.server.listen_port}")
        return True
    except Exception as e:
        print(f"✗ Config models failed: {e}")
        return False


def test_conversation_state():
    """Test conversation state model."""
    try:
        from ai_server.models.conversation_state import ConversationState

        state = ConversationState(messages={})
        state.set_llm_to_use("test_llm")
        print("✓ Conversation state works")
        return True
    except Exception as e:
        print(f"✗ Conversation state failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Phase 1 implementation...")

    tests = [
        test_api_schemas,
        test_services,
        test_config_models,
        test_conversation_state,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Phase 1 foundation is solid!")
    else:
        print("⚠️  Some tests failed - review implementation")
