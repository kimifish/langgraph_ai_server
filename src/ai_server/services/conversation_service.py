# Conversation service for managing AI conversations
import logging
import datetime
from typing import Dict, Optional
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from ..config import APP_NAME
from ..models.userconfs import UserConf


# Local exception definitions for now
class ConversationError(Exception):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details if details is not None else {}


class ServiceUnavailableError(Exception):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details if details is not None else {}


log = logging.getLogger(f"{APP_NAME}.services.conversation")


class ConversationService:
    """Service for managing AI conversations and graph execution."""

    def __init__(self, graph: Runnable):
        self.graph = graph

    def prepare_messages(self, user_input: str) -> list:
        """
        Prepare messages for graph execution.

        Args:
            user_input: The user's input message

        Returns:
            List of prepared messages
        """
        messages = []
        messages.append(HumanMessage(content=user_input))
        return messages

    def prepare_graph_init_values(self, messages: list, userconf: UserConf) -> Dict:
        """
        Prepare initial values for graph execution.

        Args:
            messages: Prepared messages
            userconf: User configuration

        Returns:
            Dictionary of initial graph values
        """
        return {
            "messages": {userconf.llm_to_use: messages},
            "user": userconf.user,
            "location": userconf.location,
            "additional_instructions": userconf.additional_instructions,
            "llm_to_use": userconf.llm_to_use,
            "last_used_llm": userconf.last_used_llm,
            "thread_id": userconf.thread_id,
        }

    async def execute_graph_stream(self, init_values: Dict, config):
        """
        Execute graph streaming and yield events.

        Args:
            init_values: Initial graph values
            config: Runnable configuration with thread_id

        Yields:
            Graph execution events
        """
        async for event in self.graph.astream(
            init_values,
            config,
            stream_mode="values",
        ):
            yield event

    def process_stream_event(self, event: Dict, userconf: UserConf) -> Dict:
        """
        Process a single stream event.

        Args:
            event: Graph event
            userconf: User configuration

        Returns:
            Processing result
        """
        # TODO: Add event debugging when logging config is available
        # TODO: Get next state from graph when method is available
        next_state = "END"  # Placeholder until graph state access is implemented

        result = {"next_state": next_state}

        if next_state == "END":
            result["answer"] = event["messages"][event["llm_to_use"]][-1].content
            userconf.last_used_llm = event["llm_to_use"]
            log.debug(
                f"Conversation completed with path: {event.get('path', 'unknown')}"
            )

        return result

    def extract_final_answer(self, event: Dict, userconf: UserConf) -> str:
        """
        Extract final answer from completed graph execution.

        Args:
            event: Final graph event
            userconf: User configuration

        Returns:
            Final answer text
        """
        try:
            return event["messages"][event["llm_to_use"]][-1].content
        except (KeyError, IndexError) as e:
            raise ConversationError(
                f"Failed to extract answer from graph result: {str(e)}"
            )

    def handle_execution_errors(self, error: Exception) -> ConversationError:
        """
        Handle and standardize execution errors.

        Args:
            error: Original exception

        Returns:
            Standardized ConversationError
        """
        if isinstance(error, ConversationError):
            return error
        return ConversationError(f"Conversation execution failed: {str(error)}")

    async def execute_conversation(
        self, user_input: str, userconf: UserConf
    ) -> Dict[str, str]:
        """
        Execute a conversation turn through the AI graph.

        Processes user input by streaming it through the configured
        LangGraph workflow, handling state transitions and error recovery.

        Args:
            user_input: The user's message to process
            userconf: User configuration containing session data

        Returns:
            Dictionary containing:
            - answer: The AI's response text
            - thread_id: Conversation thread identifier
            - llm_used: Name of the LLM that processed the request
            - error: Error message if execution failed (optional)
        """
        log.debug("Starting conversation execution")

        # Prepare messages and initial values
        messages = self.prepare_messages(user_input)
        init_values = self.prepare_graph_init_values(messages, userconf)

        answer = {
            "answer": "",
            "error": "",
            "thread_id": userconf.thread_id,
            "llm_used": userconf.llm_to_use,
        }

        try:
            # Execute graph streaming
            final_event = None
            from langchain_core.runnables import RunnableConfig

            config = RunnableConfig(configurable={"thread_id": userconf.thread_id})
            async for event in self.execute_graph_stream(init_values, config):
                result = self.process_stream_event(event, userconf)
                if result["next_state"] == "END":
                    final_event = event
                    answer["answer"] = result.get("answer", "")
                    break

            # Ensure we have a final answer
            if not answer["answer"] and final_event:
                answer["answer"] = self.extract_final_answer(final_event, userconf)

        except Exception as e:
            error = self.handle_execution_errors(e)
            log.error(f"Conversation error: {error.message}", extra=error.details)
            answer["error"] = error.message
            answer["answer"] = ""

        # Update timestamp
        userconf.last_answer_time = datetime.datetime.now()
        return answer
