# Custom exception hierarchy for AI Server
from typing import Optional, Dict, Any


class AIServerError(Exception):
    """Base exception for AI Server errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AIServerError):
    """Configuration-related errors."""

    pass


class ValidationError(AIServerError):
    """Data validation errors."""

    pass


class LLMError(AIServerError):
    """LLM-related errors."""

    pass


class ConversationError(AIServerError):
    """Conversation processing errors."""

    pass


class ToolError(AIServerError):
    """Tool execution errors."""

    pass


class APIError(AIServerError):
    """API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.status_code = status_code


class ServiceUnavailableError(APIError):
    """Service temporarily unavailable."""

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, status_code=503, details=details)


class BadRequestError(APIError):
    """Bad request error."""

    def __init__(
        self, message: str = "Bad request", details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=400, details=details)


class NotFoundError(APIError):
    """Resource not found error."""

    def __init__(
        self,
        message: str = "Resource not found",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, status_code=404, details=details)


class InternalServerError(APIError):
    """Internal server error."""

    def __init__(
        self,
        message: str = "Internal server error",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, status_code=500, details=details)
