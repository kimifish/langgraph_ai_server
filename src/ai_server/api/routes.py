# API routes for AI Server
import logging
from typing import Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Path, Body

from ..config import cfg, APP_NAME
from .schemas import ChatRequest, ChatResponse, APIErrorResponse, ErrorDetail

log = logging.getLogger(f"{APP_NAME}.api.routes")

router = APIRouter()


# Health check endpoint - must be before /{llm} route
@router.get(
    "/health",
    summary="Basic Health Check",
    description="Simple health check endpoint that returns basic server status",
)
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint.

    Returns a simple status response indicating if the server is running.

    **Returns:**
    - `status`: Always "ok" if server is responding
    - `phase`: Current development phase of the server

    **Example response:**
    ```json
    {
        "status": "ok",
        "phase": "refactoring"
    }
    ```
    """
    return {"status": "ok", "phase": "refactoring"}


@router.get(
    "/health/detailed",
    summary="Detailed Health Check",
    description="Comprehensive health check with system information, user configurations, and service status",
)
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with comprehensive system information.

    Provides detailed information about system resources, user configurations,
    and service status for monitoring and debugging purposes.

    **Returns:**
    - `status`: Overall health status
    - `timestamp`: Current server time in ISO format
    - `version`: Server version
    - `phase`: Development phase
    - `system`: System information (platform, Python version, CPU, memory)
    - `user_configurations`: User session information
    - `llm_services`: LLM service availability status

    **Example response:**
    ```json
    {
        "status": "ok",
        "timestamp": "2025-09-25T22:26:58.612931",
        "version": "0.3.0",
        "phase": "refactoring",
        "system": {
            "platform": "Linux",
            "python_version": "3.10.12",
            "cpu_count": 8,
            "memory_total": 17179869184,
            "memory_available": 8589934592
        },
        "user_configurations": {
            "total": 5,
            "status": "ok"
        },
        "llm_services": {
            "status": "mocked",
            "available_llms": ["common", "gpt-4", "claude"]
        }
    }
    ```
    """
    import datetime
    import psutil
    import platform

    # Basic system info
    system_info = {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "0.3.0",
        "phase": "refactoring",
        "system": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        },
    }

    # Check user configurations
    user_confs = getattr(cfg.runtime, "user_confs", None)
    if user_confs:
        system_info["user_configurations"] = {
            "total": len(user_confs.get_all()),
            "status": "ok",
        }
    else:
        system_info["user_configurations"] = {"status": "initializing"}

    # Check agents/LLMs (placeholder)
    system_info["llm_services"] = {
        "status": "mocked",
        "available_llms": (
            ["common", "gpt-4", "claude"] if hasattr(cfg, "agents") else []
        ),
    }

    return system_info


@router.get(
    "/metrics",
    summary="System Metrics",
    description="Real-time system and process metrics for monitoring server performance",
)
async def metrics_endpoint() -> Dict[str, Any]:
    """
    System metrics endpoint for monitoring server performance.

    Provides real-time metrics about the server process, system resources,
    and conversation statistics for monitoring and alerting purposes.

    **Returns:**
    - `timestamp`: Current time in ISO format
    - `process`: Process-specific metrics (PID, CPU, memory, threads, uptime)
    - `system`: System-wide metrics (CPU, memory, disk usage)
    - `conversations`: Conversation statistics (total users, active conversations)

    **Example response:**
    ```json
    {
        "timestamp": "2025-09-25T22:26:58.612931",
        "process": {
            "pid": 12345,
            "cpu_percent": 15.2,
            "memory_mb": 256.7,
            "threads": 8,
            "uptime_seconds": 3600.5
        },
        "system": {
            "cpu_percent": 45.3,
            "memory_percent": 67.8,
            "disk_usage": 23.4
        },
        "conversations": {
            "total_users": 5,
            "active_conversations": 2
        }
    }
    ```
    """
    import datetime
    import psutil
    import os

    process = psutil.Process(os.getpid())

    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "process": {
            "pid": os.getpid(),
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "threads": process.num_threads(),
            "uptime_seconds": (
                datetime.datetime.now()
                - datetime.datetime.fromtimestamp(process.create_time())
            ).total_seconds(),
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
        },
    }

    # Add conversation metrics if available
    user_confs = getattr(cfg.runtime, "user_confs", None)
    if user_confs:
        all_confs = user_confs.get_all()
        metrics["conversations"] = {
            "total_users": len(all_confs),
            "active_conversations": len(
                [
                    c
                    for c in all_confs.values()
                    if c.last_answer_time
                    and (datetime.datetime.now() - c.last_answer_time).seconds < 3600
                ]
            ),  # Active in last hour
        }

    return metrics


async def _process_chat_request(llm: str, request: ChatRequest) -> ChatResponse:
    """Shared logic for processing chat requests."""
    # Validate LLM
    available_llms = [
        "common",
        "smarthome",
        "music_assistant",
        "music_machine",
        "shell_assistant",
        "code_assistant",
        "school_tutor",
        "smarthome_machine",
    ]
    if llm not in available_llms:
        raise HTTPException(
            status_code=404,
            detail=APIErrorResponse(
                error=f"LLM agent '{llm}' not found",
                type="BadRequestError",
                details=[
                    ErrorDetail(
                        field="llm",
                        message=f"Must be one of: {', '.join(available_llms)}",
                        value=llm,
                    )
                ],
                request_id=f"req_{hash(str(request))}",
            ).model_dump(),
        )

    try:
        # Get user configurations from app state (will be set during app startup)
        user_confs = getattr(cfg.runtime, "user_confs", None)
        if not user_confs:
            return ChatResponse(
                answer="Service initializing...",
                thread_id=request.thread_id or "temp",
                llm_used=request.llm_to_use or llm,
                error="User configurations not initialized",
            )

        # Validate LLM
        if request.llm_to_use and not hasattr(cfg, "agents"):
            request.llm_to_use = None

        # Get or create user configuration
        userconf = user_confs.get(request.thread_id) if request.thread_id else None

        if not userconf:
            userconf = user_confs.add(
                thread_id=request.thread_id
                or f"{request.user}_{hash(request.user + str(id(request.message)))}",
                user=request.user,
                location=request.location,
                additional_instructions=request.additional_instructions,
                llm_to_use=request.llm_to_use or llm,
            )

        # Log the request
        log.debug(f"New message to /{llm}: {request.message[:100]}...")
        log.info(f"New message from {request.user} to /{llm}")

        # TODO: Integrate with ConversationService once created
        # For now, return a placeholder response
        return ChatResponse(
            answer="Service temporarily unavailable - under refactoring",
            thread_id=(
                userconf.thread_id["configurable"]["thread_id"]
                if isinstance(userconf.thread_id, dict)
                else str(userconf.thread_id)
            ),
            llm_used=userconf.llm_to_use,
            error="Phase 1 refactoring in progress",
        )

    except Exception as e:
        log.error(f"Unexpected error in chat request processing: {e}")
        return ChatResponse(
            answer="",
            thread_id=request.thread_id or "temp",
            llm_used=request.llm_to_use or llm,
            error=f"Internal server error: {str(e)}",
        )


# Local exception definitions for now
class APIError(Exception):
    def __init__(
        self, message: str, status_code: int = 500, details: Optional[Dict] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details if details is not None else {}


class BadRequestError(APIError):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=400, details=details)


class InternalServerError(APIError):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=500, details=details)


@router.get(
    "/{llm}",
    summary="AI Chat Endpoint (GET)",
    description="Main endpoint for AI conversations with configurable LLM selection - GET method with query parameters. Note: POST method is recommended for new implementations.",
)
async def chat_endpoint_get(
    llm: str = Path(
        ..., description="LLM identifier to use (e.g., 'common', 'gpt-4', 'claude')"
    ),
    message: str = Query("Hello, world!", description="Message to send to the AI"),
    thread_id: Optional[str] = Query(
        None,
        description="Thread ID for conversation continuity. If not provided, a new conversation will be created",
    ),
    user: str = Query("Undefined", description="User identifier for personalization"),
    location: str = Query(
        "Undefined", description="User location for context-aware responses"
    ),
    additional_instructions: str = Query(
        "", description="Additional system instructions for the AI"
    ),
    llm_to_use: Optional[str] = Query(
        None, description="Override the LLM to use for this specific request"
    ),
) -> ChatResponse:
    """
    Main chat endpoint for AI conversations (GET method).

    This endpoint allows you to have conversations with various AI models.
    Supports conversation threading, user context, and configurable AI behavior.

    **Path Parameters:**
    - `llm`: The LLM identifier to route the conversation to

    **Query Parameters:**
    - `message`: The message to send to the AI (required)
    - `thread_id`: Unique identifier for conversation thread (optional, auto-generated if not provided)
    - `user`: User identifier for personalization (default: "Undefined")
    - `location`: User location for context (default: "Undefined")
    - `additional_instructions`: Extra instructions for the AI (default: "")
    - `llm_to_use`: Override the default LLM for this request (optional)

    **Returns:**
    ChatResponse object with AI answer and metadata

    **Examples:**

    **Basic chat:**
    ```
    GET /common?message=Hello, how are you?
    ```

    **Threaded conversation:**
    ```
    GET /gpt-4?message=Tell me about Python&thread_id=user123_session1&user=john
    ```

    **With custom instructions:**
    ```
    GET /claude?message=Explain quantum physics&additional_instructions=Use simple analogies
    ```

    **Response example:**
    ```json
    {
        "answer": "Hello! I'm doing well, thank you for asking. How can I help you today?",
        "thread_id": "john_123456789",
        "llm_used": "common",
        "error": null
    }
    ```
    """
    # Create ChatRequest from query parameters
    request = ChatRequest(
        message=message,
        thread_id=thread_id,
        user=user,
        location=location,
        additional_instructions=additional_instructions,
        llm_to_use=llm_to_use,
    )

    return await _process_chat_request(llm, request)


@router.post(
    "/{llm}",
    summary="AI Chat Endpoint (POST)",
    description="Main endpoint for AI conversations with configurable LLM selection - POST method with JSON body",
)
async def chat_endpoint_post(
    llm: str = Path(
        ..., description="LLM identifier to use (e.g., 'common', 'gpt-4', 'claude')"
    ),
    request: ChatRequest = Body(...),
) -> ChatResponse:
    """
    Main chat endpoint for AI conversations (POST method).

    This endpoint allows you to have conversations with various AI models.
    Supports conversation threading, user context, and configurable AI behavior.

    **Path Parameters:**
    - `llm`: The LLM identifier to route the conversation to

    **Request Body:**
    JSON object with chat parameters (see ChatRequest schema)

    **Returns:**
    ChatResponse object with AI answer and metadata

    **Example:**
    ```json
    {
        "message": "Hello, how are you?",
        "user": "john_doe",
        "location": "living_room",
        "thread_id": "conv_123"
    }
    ```
    """
    return await _process_chat_request(llm, request)
