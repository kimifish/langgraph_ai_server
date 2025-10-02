"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Model for validating chat API requests."""

    message: str = Field(..., description="Message to send to AI")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")
    user: str = Field("Undefined", description="User identifier")
    location: str = Field("Undefined", description="User location")
    additional_instructions: str = Field(
        "", description="Additional instructions for AI"
    )
    llm_to_use: Optional[str] = Field(None, description="Override LLM for this request")


class ChatResponse(BaseModel):
    """Model for structured chat API responses."""

    answer: str = Field(..., description="AI response")
    thread_id: str = Field(..., description="Conversation thread ID")
    llm_used: str = Field(..., description="LLM that was used")
    error: Optional[str] = Field(None, description="Error message if any")


class HealthResponse(BaseModel):
    """Model for health check responses."""

    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="Server version")
    phase: str = Field(..., description="Development phase")


class DetailedHealthResponse(HealthResponse):
    """Extended health check with system information."""

    system: dict = Field(..., description="System information")
    user_configurations: dict = Field(..., description="User configuration status")
    llm_services: dict = Field(..., description="LLM service status")


class MetricsResponse(BaseModel):
    """Model for metrics responses."""

    timestamp: str = Field(..., description="Current timestamp")
    process: dict = Field(..., description="Process metrics")
    system: dict = Field(..., description="System metrics")
    conversations: dict = Field(..., description="Conversation metrics")
