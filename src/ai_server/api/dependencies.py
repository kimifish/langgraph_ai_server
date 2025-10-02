# API dependencies for dependency injection
from typing import Optional
from fastapi import Request
from ..models.userconfs import UserConfs


def get_user_confs(request: Request) -> UserConfs:
    """Get user configurations from app state."""
    return request.app.state.user_confs


def get_thread_id(thread_id: Optional[str] = None) -> Optional[str]:
    """Extract and validate thread ID."""
    return thread_id
