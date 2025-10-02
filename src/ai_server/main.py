# pyright: basic
# pyright: reportAttributeAccessIssue=false

import logging
import sys
import anyio
import asyncio
import time
import datetime
import uvicorn
from typing import Dict, Optional
from dataclasses import asdict
from fastapi import FastAPI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langchain_core.runnables.base import Runnable
from rich.pretty import pretty_repr
from openai import BadRequestError, PermissionDeniedError

from kimiUtils.killer import GracefulKiller
from ai_server.config import cfg, APP_NAME
from ai_server.schemas import ChatRequest
from ai_server.llm_tools import init_tools, init_mcp_tools, init_mcp_resources
from ai_server.llms import init_agents
from ai_server.graph import init_graph
from ai_server.logs.utils import log_diff, sep_line
from ai_server.models.userconfs import UserConfs, UserConf
from ai_server.history import init_memory

log = logging.getLogger(f"{APP_NAME}.{__name__}")

# Initialization of GracefulKiller for proper application termination
from ai_server.llms import cleanup_http_clients

killer = GracefulKiller(kill_targets=[cleanup_http_clients])

app = FastAPI(
    title="AI Server",
    version="0.3.0",
    description="Spin up a simple API server using LangChain's Runnable interfaces",
)


async def get_graph_answer(
    graph: Runnable, user_input: str, userconf: UserConf
) -> Dict[str, str]:
    sep_line("START")

    messages = []
    messages.append(HumanMessage(content=user_input))

    answer = dict()
    try:
        graph_init_values = {
            "messages": {userconf.llm_to_use: messages},
            "user": userconf.user,
            "location": userconf.location,
            "additional_instructions": userconf.additional_instructions,
            "llm_to_use": userconf.llm_to_use,
            "last_used_llm": userconf.last_used_llm,
            "thread_id": userconf.thread_id,
        }

        async for event in graph.astream(
            graph_init_values,
            userconf.thread_id,
            stream_mode="values",
        ):
            if cfg.logging.debug.events:
                log_diff(userconf.last_event, event)
                userconf.last_event = event

            next_state = graph.get_state(userconf.thread_id).next
            next_state = next_state[-1] if next_state else "END"
            sep_line(next_state)
            if next_state == "END":
                answer["answer"] = event["messages"][event["llm_to_use"]][-1].content
                userconf.last_used_llm = event[
                    "llm_to_use"
                ]  # Value will be used in future requests for more relevant llm autodetection
                log.debug(pretty_repr(event["path"]))

    except BadRequestError as e:
        log.error(
            f"Bad Request while executing graph: \nLast state: {pretty_repr(event)}. \nError: {e}"
        )
        answer["error"] = "Bad request"
        answer["answer"] = ""
    except PermissionDeniedError as e:
        log.error(
            f"Permission denied while executing graph: \nLast state: {pretty_repr(event)}. \nError: {e}"
        )
        answer["error"] = "Permission denied"
        answer["answer"] = ""
    except IndexError as e:
        log.error(
            f"Something wrong with answer forming: \nLast state: {pretty_repr(event)}. \nError: {e}"
        )
        answer["error"] = "Malformed answer"
        answer["answer"] = ""
    except NotImplementedError as e:
        log.error(
            f"Unrecognized error (graph chain couldn't reach END): \nLast state: {pretty_repr(event)}. \nError: {e}"
        )
        answer["error"] = "Unknown error"
        answer["answer"] = ""
    finally:
        answer["thread_id"] = event["thread_id"]["configurable"]["thread_id"]
        answer["llm_used"] = event["llm_to_use"]

    userconf.last_answer_time = datetime.datetime.now()
    return answer


async def handle_chat_request(llm: str, request: ChatRequest) -> Dict[str, str]:
    """Common function for handling chat requests from both GET and POST endpoints."""
    user_confs: UserConfs = cfg.runtime.user_confs
    userconf = user_confs.get(request.thread_id) if request.thread_id else None

    llm_to_use = request.llm_to_use or ("Undefined" if llm == "ai" else llm)
    if llm_to_use not in cfg.agents.__dict__.keys():
        llm_to_use = "Undefined"

    if not userconf:
        userconf = user_confs.add(
            thread_id=request.thread_id or f"{request.user}_{int(time.time())}",
            user=request.user,
            location=request.location,
            additional_instructions=request.additional_instructions,
            llm_to_use=llm_to_use,
        )

    log.debug(f"New message to /{llm} endpoint: {pretty_repr(request.dict())}")
    log.info(
        f"New message from {request.user} to /{llm}: [light_sky_blue3]{request.message}[/]"
    )
    answer = await get_graph_answer(
        graph=cfg.runtime.graph,
        user_input=request.message,
        userconf=userconf,
    )
    log.info(f"AI ({answer['llm_used']}) answer: [tan]{answer['answer']}[/]")
    return answer


# Function to create endpoints dynamically
def _create_endpoint(llm: str):
    # GET endpoint (existing, for backward compatibility)
    @app.get(f"/{llm}")
    async def read_item_get(
        message: str = "Hello, world!",
        thread_id: Optional[str] = "",
        user: str = "Undefined",
        location: str = "Undefined",
        additional_instructions: str = "",
        llm_to_use: str = "Undefined" if llm == "ai" else llm,
    ) -> Dict[str, str]:
        # Convert GET parameters to ChatRequest model
        request = ChatRequest(
            message=message,
            thread_id=thread_id or None,
            user=user,
            location=location,
            additional_instructions=additional_instructions,
            llm_to_use=llm_to_use if llm_to_use != "Undefined" else None,
        )
        return await handle_chat_request(llm, request)

    # POST endpoint (new, recommended)
    @app.post(f"/{llm}")
    async def read_item_post(request: ChatRequest) -> Dict[str, str]:
        return await handle_chat_request(llm, request)

    return read_item_get


def get_uvicorn() -> uvicorn.Server:
    uv_config = uvicorn.Config(
        app,
        host=cfg.server.listen_interfaces,
        port=cfg.server.listen_port,
        log_level=cfg.logging.level.lower(),
    )
    uv_server = uvicorn.Server(uv_config)
    return uv_server


async def initialize_mcp_client_safely() -> Optional[MultiServerMCPClient]:
    """Пытается инициализировать MCP клиент с таймаутом."""
    if not cfg.mcp.enabled:
        log.info("MCP disabled in configuration")
        return None

    try:
        log.info("Attempting to initialize MCP client...")
        async with asyncio.timeout(10):  # 10 секунд таймаут
            # Use the servers dict directly, not asdict
            client = MultiServerMCPClient(asdict(cfg.mcp))
            await client.__aenter__()
            log.info("MCP client initialized successfully")
            return client
    except (asyncio.TimeoutError, Exception) as e:
        log.warning(f"MCP initialization failed: {e}")
        return None

    try:
        log.info("Attempting to initialize MCP client...")
        async with asyncio.timeout(10):  # 10 секунд таймаут
            mcp_config = (
                asdict(cfg.mcp.servers)
                if hasattr(cfg.mcp, "servers")
                else asdict(cfg.mcp)
            )
            client = MultiServerMCPClient(mcp_config)
            await client.__aenter__()
            log.info("MCP client initialized successfully")
            return client
    except (asyncio.TimeoutError, Exception) as e:
        log.warning(f"MCP initialization failed: {e}. Continuing without MCP.")
        return None


async def main():
    """
    Main application entry point.

    Orchestrates the complete application initialization sequence:
    1. AI agents setup
    2. Tool initialization
    3. MCP component setup (optional)
    4. Memory system initialization
    5. Graph compilation
    6. Runtime configuration
    7. API endpoint creation
    8. Web server startup
    """
    # Phase 1: Try to initialize MCP client (optional)
    mcp_client = await initialize_mcp_client_safely()

    # Phase 2: Initialize AI agents
    initialize_agents()

    # Phase 3: Initialize static tools
    initialize_static_tools()

    # Phase 4: Initialize MCP components (only if MCP is available)
    if mcp_client:
        try:
            await initialize_mcp_components(mcp_client)
        except Exception as e:
            log.error(f"Failed to initialize MCP components: {e}")
            mcp_client = None
    else:
        log.info("Skipping MCP components initialization")

    # Phase 5: Initialize memory system
    initialize_memory_system()

    # Phase 6: Initialize conversation graph
    initialize_graph_system()

    # Phase 7: Configure runtime settings
    configure_runtime_settings()

    # Phase 8: Create API endpoints
    endpoints = create_api_endpoints()

    # Phase 9: Start web server
    log.debug(cfg.format_attributes())
    await start_web_server()


def initialize_agents():
    """
    Initialize routing and summarizing LLMs.

    This function sets up the core AI agents that handle
    conversation routing and summarization.
    """
    log.info("Initializing AI agents...")
    init_agents()
    log.info("AI agents initialized successfully")


def initialize_static_tools():
    """
    Initialize static tools that don't require external connections.

    These are tools that are available locally without needing
    MCP clients or external services.
    """
    log.info("Initializing static tools...")
    init_tools()
    log.info("Static tools initialized successfully")


async def initialize_mcp_components(mcp_client):
    """
    Initialize MCP (Model Context Protocol) tools and resources.

    Args:
        mcp_client: Initialized MCP client instance
    """
    log.info("Initializing MCP tools...")
    await init_mcp_tools(mcp_client)
    await init_mcp_resources(mcp_client)
    log.info("MCP components initialized successfully")


def initialize_memory_system():
    """
    Initialize the conversation memory system.

    Sets up persistent storage for conversation history
    and state management.
    """
    log.info("Initializing memory system...")
    init_memory()
    # TODO: mysql saver doesn't work with async. Some methods must be implemented there.
    log.info("Memory system initialized successfully")


def initialize_graph_system():
    """
    Initialize and compile the conversation graph.

    Sets up the LangGraph workflow with all nodes and edges
    for conversation processing.
    """
    log.info("Initializing conversation graph...")
    init_graph()
    log.info("Conversation graph initialized successfully")


def configure_runtime_settings():
    """
    Configure runtime settings and global state.

    Sets up application-wide configuration values
    and initializes core services.
    """
    log.info("Configuring runtime settings...")

    cfg.update("runtime.mood", "slightly depressed")

    # Initialize user configurations manager
    cfg.update("runtime.user_confs", UserConfs())

    log.info("Runtime settings configured successfully")


def create_api_endpoints():
    """
    Create and register API endpoints for all configured LLMs.

    Returns:
        List of created endpoint functions
    """
    log.info("Creating API endpoints...")

    endpoints = []
    for llm in [*cfg.agents.__dict__.keys(), cfg.endpoints.auto]:
        if llm.startswith("_"):
            continue
        endpoints.append(_create_endpoint(llm))

    log.info(f"Created {len(endpoints)} API endpoints")
    return endpoints


async def start_web_server():
    """
    Start the FastAPI web server using Uvicorn.

    This is the main server loop that handles HTTP requests.
    """
    log.info("Starting web server...")
    server = get_uvicorn()
    await server.serve()


def draw_graph(graph: Runnable):
    """
    Draw ASCII representation of the conversation graph.

    Args:
        graph: The compiled LangGraph instance
    """
    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Error drawing graph: {e}")


def run():
    """
    Entry point for the application. Parses command line arguments,
    initializes the configuration, and runs the main application.
    """
    try:
        # Launching the main function of the application
        anyio.run(main)
    except NotImplementedError as e:
        log.error(f"Unexpected error: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Unexpected error during startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
