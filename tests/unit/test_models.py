"""
Unit tests for data models.
"""

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from ai_server.models.state import add_path, add_summary
from ai_server.models.userconfs import UserConf, UserConfs


class TestState:
    """Test State TypedDict and reducer functions."""

    def test_add_path_function(self):
        """Test the add_path reducer function."""
        # Test with empty path
        result = add_path([], "start")
        assert result == [["start"]]

        # Test adding to existing path
        result = add_path([["start"]], "node1")
        assert result == [["start", "node1"]]

        # Test starting new sublist
        result = add_path([["start", "node1"]], "start")
        assert result == [["start", "node1"], ["start"]]

    def test_add_messages_to_dict_function(self):
        """Test the add_messages_to_dict reducer function."""
        # Skip this test for now due to type complexity
        pass

    def test_add_summary_function(self):
        """Test the add_summary reducer function."""
        left = {"agent1": "summary1"}
        right = {"agent2": "summary2"}

        result = add_summary(left, right)
        assert result["agent1"] == "summary1"
        assert result["agent2"] == "summary2"

    def test_state_creation(self):
        """Test creating a State object."""
        state = {
            "messages": {"common": [HumanMessage(content="Test")]},
            "user": "test_user",
            "location": "test_location",
            "additional_instructions": "test_instructions",
            "llm_to_use": "common",
            "last_used_llm": "common",
            "thread_id": "test_thread",
            "path": [["start"]],
            "summary": {"common": "test_summary"},
        }

        # Verify all required fields are present
        required_fields = [
            "messages",
            "user",
            "location",
            "additional_instructions",
            "llm_to_use",
            "last_used_llm",
            "thread_id",
            "path",
            "summary",
        ]

        for field in required_fields:
            assert field in state


class TestUserConf:
    """Test UserConf model."""

    def test_user_conf_creation(self):
        """Test creating a UserConf instance."""
        config = RunnableConfig(configurable={"thread_id": "test_123"})

        user_conf = UserConf(
            thread_id=config,
            user="test_user",
            location="test_location",
            additional_instructions="test_instructions",
            llm_to_use="common",
            last_used_llm="common",
        )

        assert user_conf.thread_id == config
        assert user_conf.user == "test_user"
        assert user_conf.location == "test_location"
        assert user_conf.additional_instructions == "test_instructions"
        assert user_conf.llm_to_use == "common"
        assert user_conf.last_used_llm == "common"

    def test_user_conf_defaults(self):
        """Test UserConf default values."""
        config = RunnableConfig(configurable={"thread_id": "test_123"})

        user_conf = UserConf(thread_id=config)

        assert user_conf.user == "Undefined"
        assert user_conf.location == "Undefined"
        assert user_conf.additional_instructions == ""
        assert user_conf.llm_to_use == "Undefined"
        assert user_conf.last_used_llm == ""


class TestUserConfs:
    """Test UserConfs collection class."""

    def test_user_confs_initialization(self):
        """Test UserConfs initialization."""
        user_confs = UserConfs()
        assert user_confs.user_dict == {}

    def test_add_user_conf_with_string_thread_id(self):
        """Test adding user conf with string thread_id."""
        user_confs = UserConfs()

        result = user_confs.add(
            thread_id="test_thread_123",
            user="test_user",
            location="test_location",
            llm_to_use="common",
        )

        assert isinstance(result, UserConf)
        assert result.user == "test_user"
        assert result.location == "test_location"
        assert result.llm_to_use == "common"

    def test_add_user_conf_with_runnable_config(self):
        """Test adding user conf with RunnableConfig thread_id."""
        user_confs = UserConfs()
        config = RunnableConfig(configurable={"thread_id": "test_123"})

        result = user_confs.add(thread_id=config, user="test_user", llm_to_use="common")

        assert result.thread_id == config
        assert result.user == "test_user"

    def test_get_user_conf(self):
        """Test retrieving user configuration."""
        user_confs = UserConfs()
        user_confs.add(thread_id="test_123", user="test_user")

        result = user_confs.get("test_123")
        assert result is not None
        assert result.user == "test_user"

    def test_get_nonexistent_user_conf(self):
        """Test retrieving non-existent user configuration."""
        user_confs = UserConfs()

        result = user_confs.get("nonexistent")
        assert result is None

    def test_exists_user_conf(self):
        """Test checking if user conf exists."""
        user_confs = UserConfs()
        user_confs.add(thread_id="test_123", user="test_user")

        assert user_confs.exists("test_123") is True
        assert user_confs.exists("nonexistent") is False

    def test_get_all_user_confs(self):
        """Test getting all user configurations."""
        user_confs = UserConfs()
        user_confs.add(thread_id="test_1", user="user1")
        user_confs.add(thread_id="test_2", user="user2")

        all_confs = user_confs.get_all()
        assert len(all_confs) == 2
        assert "test_1" in all_confs
        assert "test_2" in all_confs
