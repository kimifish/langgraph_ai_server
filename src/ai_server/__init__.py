"""
AI Server using LangChain and FastAPI.

This package provides a server implementation that uses LangChain for AI model integration
and FastAPI for API endpoints.
"""

from importlib.metadata import version, PackageNotFoundError

# Core components
from .main import app, main
from .state import State, add_path
from .llms import LLMNode, define_llm
from .graph import _init_graph
from .tools import _init_tools

try:
    __version__ = version("ai_server")
except PackageNotFoundError:
    __version__ = "0.2.2"  # fallback version

__all__ = [
    "app",
    "main",
    "State",
    "add_path",
    "LLMNode",
    "define_llm",
    "_init_graph",
    "_init_tools",
]