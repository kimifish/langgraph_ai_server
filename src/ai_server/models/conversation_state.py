# Simplified conversation state model
from typing import Dict, List
from pydantic import BaseModel
from langchain_core.messages import BaseMessage


class ConversationState(BaseModel):
    """Simplified state model for conversation management."""

    # Messages kept in dict like { "common": [HumanMessage, ], "school_tutor": [AIMessage, ], }
    messages: Dict[str, List[BaseMessage]]

    # User information
    user: str = "Undefined"
    location: str = "Undefined"
    additional_instructions: str = ""

    # LLM routing
    llm_to_use: str = "Undefined"
    last_used_llm: str = ""

    # Conversation management
    thread_id: str = ""
    path: List[List[str]] = []
    summary: Dict[str, str] = {}

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True  # Allow BaseMessage types

    def add_to_path(self, step: str):
        """Add a step to the current path."""
        if not self.path:
            self.path = [[]]
        if isinstance(step, str):
            self.path[-1].append(step)
        elif isinstance(step, list):
            (
                self.path.extend(step)
                if isinstance(step[0], list)
                else self.path[-1].extend(step)
            )

    def start_new_path(self):
        """Start a new path segment."""
        self.path.append([])

    def get_current_messages(self) -> List[BaseMessage]:
        """Get messages for the current LLM."""
        return self.messages.get(self.llm_to_use, [])

    def update_messages(self, llm_name: str, new_messages: List[BaseMessage]):
        """Update messages for a specific LLM."""
        self.messages[llm_name] = new_messages

    def add_messages(self, llm_name: str, new_messages: List[BaseMessage]):
        """Add messages to a specific LLM's conversation."""
        if llm_name not in self.messages:
            self.messages[llm_name] = []
        self.messages[llm_name].extend(new_messages)

    def get_llm_messages(self, llm_name: str) -> List[BaseMessage]:
        """Get messages for a specific LLM."""
        return self.messages.get(llm_name, [])

    def set_llm_to_use(self, llm_name: str):
        """Set the current LLM to use and update last used."""
        if self.llm_to_use and self.llm_to_use != llm_name:
            self.last_used_llm = self.llm_to_use
        self.llm_to_use = llm_name

    def should_cut_history(self, max_length: int = 10) -> bool:
        """Check if conversation should be cut based on length."""
        current_messages = self.get_current_messages()
        return len(current_messages) > max_length

    def should_summarize_history(self, max_length: int = 0) -> bool:
        """Check if conversation should be summarized based on length."""
        if max_length <= 0:
            return False
        current_messages = self.get_current_messages()
        return len(current_messages) > max_length

    def cut_history(self, keep_last: int = 10):
        """Cut conversation history to keep only the last N messages."""
        current_messages = self.get_current_messages()
        if len(current_messages) > keep_last:
            self.messages[self.llm_to_use] = current_messages[-keep_last:]

    def add_summary(self, key: str, summary: str):
        """Add a summary for a conversation aspect."""
        self.summary[key] = summary

    def get_summary(self, key: str) -> str:
        """Get a summary for a conversation aspect."""
        return self.summary.get(key, "")
