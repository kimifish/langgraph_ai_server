"""
Unit tests for graph module.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from langgraph.graph import END

from ai_server.graph import (
    StartNode,
    FinalNode,
    init_graph,
    route_llms,
    route_tools,
    route_shorten_history,
    _update_path_in_state,
)
from ai_server.models.state import State


def create_test_state(**kwargs) -> State:
    """Create a minimal valid State for testing."""
    default_state: State = {
        "messages": {},
        "user": "test_user",
        "location": "test_location",
        "additional_instructions": "",
        "llm_to_use": "test_agent",
        "last_used_llm": "test_agent",
        "last_messages": {},
        "thread_id": "test_thread",
        "path": [["start"]],  # Initialize with start path
        "summary": {},
    }
    default_state.update(kwargs)  # type: ignore
    return default_state


@pytest.fixture
def mock_cfg_with_agents():
    """Mock cfg with agents configuration."""
    with patch("ai_server.graph.cfg") as mock_cfg:
        # Mock agents
        mock_agent = MagicMock()
        mock_agent.history.post_to_common = False
        mock_agent.history.use_common = False
        mock_agent.history.cut_after = 10
        mock_agent.history.summarize_after = 20

        mock_cfg.agents.__dict__ = {
            "test_agent": mock_agent,
            "_define": MagicMock(),
            "_summarize": MagicMock(),
        }

        # Mock runtime
        mock_cfg.runtime.memory = MagicMock()
        mock_cfg.runtime.tools = MagicMock()
        mock_cfg.runtime.tools.test_agent = []
        mock_cfg.runtime.console = MagicMock()

        # Mock logging
        mock_cfg.logging = MagicMock()

        yield mock_cfg


@pytest.fixture
def mock_cfg_summarize_only():
    """Mock cfg with agents configuration for summarize testing (no cutting)."""
    with patch("ai_server.graph.cfg") as mock_cfg:
        # Mock agents
        mock_agent = MagicMock()
        mock_agent.history.post_to_common = False
        mock_agent.history.use_common = False
        mock_agent.history.cut_after = 0  # Disable cutting
        mock_agent.history.summarize_after = 15

        mock_cfg.agents.__dict__ = {
            "test_agent": mock_agent,
            "_define": MagicMock(),
            "_summarize": MagicMock(),
        }

        # Mock runtime
        mock_cfg.runtime.memory = MagicMock()
        mock_cfg.runtime.tools = MagicMock()
        mock_cfg.runtime.tools.test_agent = []
        mock_cfg.runtime.console = MagicMock()

        # Mock logging
        mock_cfg.logging = MagicMock()

        yield mock_cfg


class TestStartNode:
    """Test StartNode class."""

    def test_start_node_call(self):
        """Test StartNode.__call__ returns correct path."""
        node = StartNode()
        state = create_test_state()
        result = node(state)
        assert result == {"path": ["start"]}


class TestFinalNode:
    """Test FinalNode class."""

    def test_final_node_call_with_messages(self):
        """Test FinalNode.__call__ with messages."""
        with patch("ai_server.graph.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = "2024-01-01"

            node = FinalNode()
            state = create_test_state(
                messages={"test_agent": [AIMessage(content="Test response")]}
            )

            result = node(state)
            assert result["path"] == "final"
            assert "last_answer_time" in result

    def test_final_node_call_no_messages(self):
        """Test FinalNode.__call__ with no messages raises error."""
        node = FinalNode()
        state = create_test_state(messages={})

        with pytest.raises(ValueError, match="No messages found in inputs"):
            node(state)


class TestModToolNode:
    """Test ModToolNode class."""

    @pytest.mark.skip(reason="Complex tool mocking required")
    @pytest.mark.asyncio
    async def test_mod_tool_node_ainvoke(self, mock_cfg_with_agents):
        """Test ModToolNode.ainvoke processes tool calls correctly."""
        # TODO: Implement proper tool mocking
        pass


class TestRoutingFunctions:
    """Test routing functions."""

    def test_route_llms_with_known_llm(self, mock_cfg_with_agents):
        """Test route_llms with known LLM."""
        state = create_test_state(llm_to_use="test_agent")
        result = route_llms(state)
        assert result == "test_agent_llm"

    def test_route_llms_with_unknown_llm(self, mock_cfg_with_agents):
        """Test route_llms with unknown LLM."""
        state = create_test_state(llm_to_use="unknown_agent")
        result = route_llms(state)
        assert result == "define_llm"

    def test_route_llms_no_llm(self, mock_cfg_with_agents):
        """Test route_llms with no LLM specified."""
        state = create_test_state(llm_to_use="")
        result = route_llms(state)
        assert result == "define_llm"

    def test_route_tools_with_tool_calls(self):
        """Test route_tools with tool calls."""
        from langchain_core.messages import ToolCall

        tool_call = ToolCall(id="1", name="test", args={})
        state = create_test_state(
            messages={"test_agent": [AIMessage(content="Test", tool_calls=[tool_call])]}
        )
        result = route_tools(state)
        assert result == "tools"

    def test_route_tools_without_tool_calls(self):
        """Test route_tools without tool calls."""
        state = create_test_state(messages={"test_agent": [AIMessage(content="Test")]})
        result = route_tools(state)
        assert result == "final"

    def test_route_tools_no_messages(self):
        """Test route_tools with no messages raises error."""
        state = create_test_state(messages={})
        with pytest.raises(ValueError, match="No messages found in input state"):
            route_tools(state)

    def test_route_shorten_history_cut(self, mock_cfg_with_agents):
        """Test route_shorten_history returns cut_conversation."""
        state = create_test_state(
            messages={
                "test_agent": [AIMessage(content="Test")] * 15  # More than cut_after=10
            }
        )
        result = route_shorten_history(state)
        assert result == "cut_conversation"

    def test_route_shorten_history_summarize(self, mock_cfg_summarize_only):
        """Test route_shorten_history returns summarize_conversation."""
        state = create_test_state(
            messages={
                "test_agent": [AIMessage(content="Test")]
                * 16  # More than summarize_after=15
            }
        )
        result = route_shorten_history(state)
        assert result == "summarize_conversation"

    def test_route_shorten_history_end(self, mock_cfg_with_agents):
        """Test route_shorten_history returns END."""
        state = create_test_state(
            messages={
                "test_agent": [AIMessage(content="Test")] * 5  # Less than thresholds
            }
        )
        result = route_shorten_history(state)
        assert result == END


class TestUpdatePathInState:
    """Test _update_path_in_state function."""

    def test_update_path_in_state(self):
        """Test _update_path_in_state updates path correctly."""
        state = create_test_state(path=[["start"]])
        _update_path_in_state(state, "test_point")
        assert state["path"] == [["start", "test_point"]]

    def test_update_path_in_state_empty_path(self):
        """Test _update_path_in_state with empty path list."""
        state = create_test_state(path=[])
        # This should fail as expected since path is empty
        with pytest.raises(IndexError):
            _update_path_in_state(state, "test_point")


class TestInitGraph:
    """Test init_graph function."""

    def test_init_graph(self, mock_cfg_with_agents):
        """Test init_graph creates graph correctly."""
        with (
            patch("ai_server.graph.StateGraph") as mock_state_graph,
            patch("ai_server.graph.LLMNode") as mock_llm_node,
            patch("ai_server.graph.ModToolNode") as mock_tool_node,
            patch("ai_server.graph.define_llm") as mock_define_llm,
            patch("ai_server.graph.summarize_conversation") as mock_summarize,
            patch("ai_server.graph.cut_conversation") as mock_cut,
        ):
            # Mock the graph builder
            mock_graph = MagicMock()
            mock_state_graph.return_value = mock_graph

            # Mock the compiled graph
            mock_compiled_graph = MagicMock()
            mock_graph.compile.return_value = mock_compiled_graph

            init_graph()

            # Verify key nodes were added
            assert mock_graph.add_node.called
            assert mock_graph.add_edge.called

            # Verify graph was compiled
            mock_graph.compile.assert_called_once()
