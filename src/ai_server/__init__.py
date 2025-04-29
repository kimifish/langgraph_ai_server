"""
AI Server using LangChain and FastAPI.

This package provides a server implementation that uses LangChain for AI model integration
and FastAPI for API endpoints.
"""

from importlib.metadata import version, PackageNotFoundError

# Core components
from .main import app, main
from .models.state import State, add_path
from .llms import LLMNode, define_llm
from .graph import init_graph
from .llm_tools import init_tools

try:
    __version__ = version("ai_server")
except PackageNotFoundError:
    __version__ = "0.3.0"  # fallback version

__all__ = [
    "app",
    "main",
    "State",
    "add_path",
    "LLMNode",
    "define_llm",
    "init_graph",
    "init_tools",
]