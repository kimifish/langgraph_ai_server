"""
Unit tests for LLM module.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from ai_server.llms import _get_llm, LLMNode


@pytest.fixture
def mock_agent_config():
    """Mock agent configuration."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.temperature = 0.7
    config.streaming = False
    config.proxy = ""
    config.resources = []
    config.history = MagicMock()
    config.history.post_to_common = False
    return config


@pytest.fixture
def mock_cfg_with_agent(mock_agent_config):
    """Mock cfg with test_agent."""
    with patch("ai_server.llms.cfg") as mock_cfg:
        mock_cfg.agents.test_agent = mock_agent_config
        mock_cfg.prompts.test_agent = "Test prompt"
        mock_cfg.runtime.mood = "happy"
        mock_cfg.runtime.tools = MagicMock()
        mock_cfg.runtime.resources = []
        mock_cfg.logging.debug.llm_init = False
        mock_cfg.logging.debug.prompts = False
        mock_cfg.logging.debug.messages_diff = False
        yield mock_cfg


class TestGetLLM:
    """Test LLM factory function."""

    @patch("langchain_openai.ChatOpenAI")
    def test_get_openai_llm(self, mock_chat_openai):
        """Test creating OpenAI LLM."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        result = _get_llm(
            model="gpt-4o", temperature=0.7, streaming=False, proxy="", tools=None
        )

        assert result == mock_instance
        mock_chat_openai.assert_called_once()

    def test_get_deepseek_llm(self):
        """Test creating DeepSeek LLM."""
        mock_module = MagicMock()
        mock_chat_deepseek = MagicMock()
        mock_module.ChatDeepSeek = mock_chat_deepseek
        mock_instance = MagicMock()
        mock_chat_deepseek.return_value = mock_instance

        with patch.dict("sys.modules", {"langchain_deepseek": mock_module}):
            result = _get_llm(
                model="deepseek-chat",
                temperature=0.5,
                streaming=True,
                proxy="test_proxy",
                tools=None,
            )

            assert result == mock_instance
            mock_chat_deepseek.assert_called_once()

    def test_get_unknown_llm(self):
        """Test handling unknown LLM model."""
        result = _get_llm(
            model="unknown-model",
            temperature=0.7,
            streaming=False,
            proxy="",
            tools=None,
        )

        # Should return None for unknown models
        assert result is None


class TestLLMNode:
    """Test LLMNode class."""

    def test_llm_node_initialization(self, mock_cfg_with_agent):
        """Test LLMNode initialization."""
        with patch("ai_server.llms._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm

            node = LLMNode("test_agent")

            assert node.name == "test_agent"
            assert node.llm == mock_llm
            mock_get_llm.assert_called_once()

    def test_prepare_prompt_template(self, mock_cfg_with_agent):
        """Test prompt template preparation."""
        with patch("ai_server.llms._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm

            node = LLMNode("test_agent")
            template = node.prepare_prompt_template()

            # Check that template has the expected structure
            assert hasattr(template, "invoke")

    def test_prepare_prompt_substitutions(self, mock_cfg_with_agent):
        """Test prompt substitutions preparation."""
        with (
            patch("ai_server.llms._get_llm") as mock_get_llm,
            patch("ai_server.llms.datetime") as mock_datetime,
        ):
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            mock_datetime.now.return_value.strftime.return_value = "2024-01-01"

            node = LLMNode("test_agent")

            state = {
                "user": "test_user",
                "location": "test_location",
                "additional_instructions": "test_instructions",
                "messages": {"test_agent": [HumanMessage(content="Hello")]},
                "summary": {"test_agent": "test_summary"},
            }

            substitutions = node.prepare_prompt_substitutions(state)

            assert substitutions["mood"] == "happy"
            assert substitutions["username"] == "test_user"
            assert substitutions["location"] == "test_location"
            assert substitutions["additional_instructions"] == "test_instructions"
            assert substitutions["summary"] == "test_summary"
            assert "conversation" in substitutions
            assert "mcp_resources" in substitutions

    @pytest.mark.asyncio
    async def test_execute_llm_call(self, mock_cfg_with_agent):
        """Test LLM call execution."""
        with patch("ai_server.llms._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            node = LLMNode("test_agent")
            prompt = MagicMock()

            result = await node.execute_llm_call(prompt)

            assert result == mock_response
            mock_llm.ainvoke.assert_called_once_with(prompt)

    def test_prepare_return_values(self, mock_cfg_with_agent):
        """Test return values preparation."""
        with patch("ai_server.llms._get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm

            node = LLMNode("test_agent")

            answer = [AIMessage(content="Test response")]
            state = {
                "messages": {"test_agent": [HumanMessage(content="Hello")]},
                "llm_to_use": "test_agent",
            }

            result = node.prepare_return_values(answer, state)

            assert "messages" in result
            assert "path" in result
            assert result["path"] == "test_agent"

    @pytest.mark.asyncio
    async def test_llm_node_call_with_valid_llm(self, mock_cfg_with_agent):
        """Test LLMNode.__call__ with valid LLM."""
        with patch("ai_server.llms._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            node = LLMNode("test_agent")

            state = {
                "messages": {"test_agent": [HumanMessage(content="Hello")]},
                "user": "test_user",
                "location": "test_location",
                "additional_instructions": "",
                "llm_to_use": "test_agent",
                "last_used_llm": "test_agent",
                "thread_id": RunnableConfig(configurable={"thread_id": "test_123"}),
                "path": [],
                "summary": {},
                "last_messages": {},
            }

            result = await node.__call__(state)

            assert isinstance(result, dict)
            assert "messages" in result
            assert "path" in result

    @pytest.mark.asyncio
    async def test_llm_node_call_with_none_llm(self, mock_cfg_with_agent):
        """Test LLMNode.__call__ with None LLM."""
        with patch("ai_server.llms._get_llm") as mock_get_llm:
            mock_get_llm.return_value = None

            node = LLMNode("test_agent")

            state = {
                "messages": {"test_agent": [HumanMessage(content="Hello")]},
                "user": "test_user",
                "location": "test_location",
                "additional_instructions": "",
                "llm_to_use": "test_agent",
                "last_used_llm": "test_agent",
                "thread_id": RunnableConfig(configurable={"thread_id": "test_123"}),
                "path": [],
                "summary": {},
                "last_messages": {},
            }

            result = await node.__call__(state)

            assert result == {"path": "test_agent"}
