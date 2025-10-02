# API middleware for logging, validation, and CORS
import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import APP_NAME

log = logging.getLogger(f"{APP_NAME}.api.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        log.info(f"Request: {request.method} {request.url}")

        try:
            response = await call_next(request)

            # Log response
            process_time = time.time() - start_time
            log.info(f"Response: {response.status_code} in {process_time:.2f}s")

            return response

        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            log.error(f"Error in {process_time:.2f}s: {str(e)}")
            raise


def get_cors_middleware_kwargs() -> dict:
    """Get CORS middleware configuration for FastAPI app."""
    return {
        "allow_origins": ["*"],  # Configure based on your needs
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


def get_logging_middleware_class():
    """Get logging middleware class for app.add_middleware()."""
    return LoggingMiddleware
