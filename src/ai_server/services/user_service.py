# User service for managing user sessions and configurations
import logging
from typing import Optional, Dict

from ..config import APP_NAME
from ..models.userconfs import UserConfs, UserConf


# Local exception definitions for now
class ValidationError(Exception):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details if details is not None else {}


log = logging.getLogger(f"{APP_NAME}.services.user")


class UserService:
    """Service for managing user sessions and configurations."""

    def __init__(self):
        self._user_confs = UserConfs()

    def get_user_config(self, thread_id: str) -> Optional[UserConf]:
        """
        Get user configuration by thread ID.

        Args:
            thread_id: Thread identifier

        Returns:
            User configuration or None if not found
        """
        return self._user_confs.get(thread_id)

    def create_user_config(
        self,
        thread_id: Optional[str] = None,
        user: str = "Undefined",
        location: str = "Undefined",
        additional_instructions: str = "",
        llm_to_use: str = "Undefined",
    ) -> UserConf:
        """
        Create a new user configuration.

        Args:
            thread_id: Thread identifier
            user: User identifier
            location: User location
            additional_instructions: Additional instructions
            llm_to_use: LLM to use

        Returns:
            Created user configuration
        """
        if not thread_id:
            thread_id = f"{user}_{hash(user + str(id(self)))}"

        userconf = self._user_confs.add(
            thread_id=thread_id,
            user=user,
            location=location,
            additional_instructions=additional_instructions,
            llm_to_use=llm_to_use,
        )

        log.info(f"Created user config for {user} with thread {thread_id}")
        return userconf

    def get_or_create_user_config(
        self,
        thread_id: Optional[str] = None,
        user: str = "Undefined",
        location: str = "Undefined",
        additional_instructions: str = "",
        llm_to_use: str = "Undefined",
    ) -> UserConf:
        """
        Get existing user config or create new one.

        Args:
            thread_id: Thread identifier
            user: User identifier
            location: User location
            additional_instructions: Additional instructions
            llm_to_use: LLM to use

        Returns:
            User configuration
        """
        if thread_id:
            existing = self.get_user_config(thread_id)
            if existing:
                return existing

        return self.create_user_config(
            thread_id=thread_id,
            user=user,
            location=location,
            additional_instructions=additional_instructions,
            llm_to_use=llm_to_use,
        )

    def get_all_user_configs(self) -> dict[str, UserConf]:
        """
        Get all user configurations.

        Returns:
            Dictionary of all user configurations
        """
        return self._user_confs.get_all()

    @property
    def user_confs(self) -> UserConfs:
        """Get the underlying UserConfs instance."""
        return self._user_confs
