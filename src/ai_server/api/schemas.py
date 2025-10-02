# API schemas for request/response validation
from typing import Optional, List, Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request model for chat endpoints."""

    message: str
    thread_id: Optional[str] = None
    user: str = "Undefined"
    location: str = "Undefined"
    additional_instructions: str = ""
    llm_to_use: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""

    answer: str
    thread_id: str
    llm_used: str
    error: Optional[str] = None


class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: str
    message: str
    value: Optional[Any] = None


class APIErrorResponse(BaseModel):
    """Standardized API error response."""

    error: str
    type: str  # Error class name
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Legacy error response model for backward compatibility."""

    error: str
    detail: Optional[str] = None
