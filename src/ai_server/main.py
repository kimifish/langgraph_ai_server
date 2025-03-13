# pyright: basic
# pyright: reportAttributeAccessIssue=false

import argparse
import logging
import os
import sys
import time
import datetime
import asyncio
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import Runnable
from langchain_core.runnables import RunnableConfig
from kimiconfig import Config
from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from rich.traceback import install as install_rich_traceback
from openai import BadRequestError, PermissionDeniedError

from kimiUtils.killer import GracefulKiller
from tools import _init_tools
from llms import _init_models
from graph import _init_graph
from utils import log_diff, common_theme, log_theme, sep_line

HOME_DIR = os.path.expanduser("~")
DEFAULT_CONFIG_FILE = os.path.join(os.getenv("XDG_CONFIG_HOME", os.path.join(HOME_DIR, ".config")), "ai_server", "config.yaml")
PROMPTS_FILE = os.path.join(os.getenv("XDG_CONFIG_HOME", os.path.join(HOME_DIR, ".config")), "ai_server", "prompts.yaml")

load_dotenv()

console = Console(record=True, theme=common_theme)
log_console = Console(record=True, theme=log_theme)

# Logging setup
logging.basicConfig(
    level=logging.NOTSET,
    format="%(message)s",
    datefmt="%X",
    handlers=[RichHandler(console=log_console, markup=True)],
)
parent_logger = logging.getLogger("ai_server")

for logger_name in [
    "uvicorn",
    "httpx",
    "markdown_it",
    "httpcore",
    "openai._base_client",
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

log = logging.getLogger("ai_server.main")
install_rich_traceback(show_locals=True)

# Configuration initialization 
cfg = Config(use_dataclasses=True)
cfg.update("runtime.console", console)
cfg.update("runtime.log_console", log_console)

# Initialization of GracefulKiller for proper application termination
killer = GracefulKiller(kill_targets=[cfg.shutdown])

app = FastAPI(
    title="AI Server",
    version="0.2.2",
    description="Spin up a simple API server using LangChain's Runnable interfaces",
)


class UserConf(BaseModel):
    thread_id: RunnableConfig
    user: str = "Undefined"
    location: str = "Undefined"
    additional_instructions: str = ""
    llm_to_use: str = "Undefined"
    last_used_llm: str = ""
    last_event: Dict = {}
    last_answer_time: datetime.datetime = datetime.datetime.now()


class UserConfs:
    def __init__(self):
        self.user_dict: dict[str, UserConf] = {}

    def add(self, **kwargs) -> UserConf:
        thread_id = kwargs['thread_id']
        if isinstance(thread_id, str):
            kwargs['thread_id'] = RunnableConfig(configurable={"thread_id": thread_id})
        conf = UserConf(**kwargs)
        self.user_dict[thread_id] = conf
        log.debug(self)
        return conf

    def get(self, thread_id: str) -> Optional[UserConf]:
        return self.user_dict.get(thread_id)

    def get_all(self) -> dict[str, UserConf]:
        return self.user_dict

    def exists(self, thread_id: str) -> bool:
        return thread_id in self.user_dict

    def __str__(self) -> str:
        result = "UserConfs:\n"
        for k, v in self.user_dict.items():
            result += f"{k}: {pretty_repr(v)}\n"
        return result


def _init_memory():
    cfg.update("runtime.memory", MemorySaver())  # TODO: Change to DB 


def get_graph_answer(graph: Runnable, user_input: str, userconf: UserConf) -> Dict[str, str]:
    sep_line("START")

    messages = []
    messages.append(HumanMessage(content=user_input))


    answer = dict()
    try:
        for event in graph.stream(
            {
                "messages": {userconf.llm_to_use: messages},
                "user": userconf.user,
                "location": userconf.location,
                "additional_instructions": userconf.additional_instructions,
                "llm_to_use": userconf.llm_to_use,
                "last_used_llm": userconf.last_used_llm,
                "thread_id": userconf.thread_id,
            },
            userconf.thread_id,
            stream_mode="values",
        ):
            log_diff(userconf.last_event, event)
            userconf.last_event = event
            next_state = graph.get_state(userconf.thread_id).next
            next_state = next_state[-1] if next_state else "END"
            sep_line(next_state)
            if next_state == 'END':
                answer["answer"] = event["messages"][event["llm_to_use"]][-1].content
                userconf.last_used_llm = event["llm_to_use"]  # Value will be used in future requests for more relevant llm autodetection
                log.debug(pretty_repr(event["path"]))

    except BadRequestError as e:
        log.error(f"Bad Request while executing graph: \nLast state: {pretty_repr(event)}. \nError: {e}")
        answer["error"] = "Bad request"
        answer["answer"] = ""
    except PermissionDeniedError as e:
        log.error(f"Permission denied while executing graph: \nLast state: {pretty_repr(event)}. \nError: {e}")
        answer["error"] = "Permission denied"
        answer["answer"] = ""
    except IndexError as e:
        log.error(f"Something wrong with answer forming: \nLast state: {pretty_repr(event)}. \nError: {e}")
        answer["error"] = "Malformed answer"
        answer["answer"] = ""
    except Exception as e:
        log.error(f"Unrecognized error (graph chain couldn't reach END): \nLast state: {pretty_repr(event)}. \nError: {e}")
        answer["error"] = "Unknown error"
        answer["answer"] = ""
    finally:
        answer["thread_id"] = event["thread_id"]["configurable"]["thread_id"]
        answer["llm_used"] = event["llm_to_use"]

    # if not answer:
    #     log.error(f"Unrecognized error (graph chain couldn't reach END): \nLast state: {pretty_repr(event)}.")
    #     answer["error"] = "Unrecognized error"
    #     answer["answer"] = ""

    userconf.last_answer_time = datetime.datetime.now()
    return answer


# Function to create endpoints dynamically
def _create_endpoint(llm: str):
    @app.get(f"/{llm}")
    def read_item(
        message: str = "Hello, world!",
        thread_id: Optional[str] = "",
        user: str = "Undefined",
        location: str = "Undefined",
        additional_instructions: str = "",
        llm_to_use: str = "Undefined" if llm == "ai" else llm,
    ) -> Dict[str, str]:
        user_confs: UserConfs = cfg.runtime.user_confs
        userconf = user_confs.get(thread_id) if thread_id else None

        if llm_to_use not in cfg.models.__dict__.keys():
            llm_to_use = "Undefined"

        if not userconf:
            userconf = user_confs.add(
                thread_id=thread_id or f"{user}_{int(time.time())}",
                user=user,
                location=location,
                additional_instructions=additional_instructions,
                llm_to_use=llm_to_use,
            )
        
        log.debug(
            f"New message to /{llm} endpoint: "
            + pretty_repr(
                {
                    "messages": message,
                    "thread_id": thread_id,
                    "user": user,
                    "location": location,
                    "additional_instructions": additional_instructions,
                    "llm_to_use": llm_to_use,
                }
            )
        )
        log.info(f"New message from {user} to /{llm}: [light_sky_blue3]{message}[/]")
        answer = get_graph_answer(
            graph=cfg.runtime.graph,
            user_input=message,
            userconf=userconf,
        )
        log.info(f"AI ({answer['llm_used']}) answer: [tan]{answer['answer']}[/]")
        return answer
    return read_item


def main():
    _init_logs()
    _init_models()
    _init_tools()
    _init_memory()
    _init_graph()
    cfg.update("runtime.mood", "slightly depressed")  # TODO: Nothing here yet.
    cfg.update("runtime.user_confs", UserConfs())

    import uvicorn

    # Creating API endpoints
    # endpoints = [await _create_endpoint(llm) for llm in [*cfg.models.__dict__.keys(), cfg.endpoints.auto] if not llm.startswith('_') else continue]


    endpoints = list()
    for llm in [*cfg.models.__dict__.keys(), cfg.endpoints.auto]:
        if llm.startswith('_'):
            continue
        endpoints.append(_create_endpoint(llm))

    uvicorn.run(
        app,
        host=cfg.server.listen_interfaces,
        port=cfg.server.listen_port,
        # workers=1,
        # log_config=log_config,
        # lifespan=lifespan,
        # ssl_keyfile="key.pem",        # HTTPS
        # ssl_certfile="cert.pem",
        # reload=True,                  # auto reload if code changed
    )


def draw_graph(graph: Runnable):
    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Error drawing graph: {e}")


def _init_logs():
    parent_logger.setLevel(cfg.logging.level)


def _parse_args():
    parser = argparse.ArgumentParser(prog="ai_server", description="AI Server")
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default=DEFAULT_CONFIG_FILE,
        help="Configuration file location.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        dest="prompts_file",
        default=PROMPTS_FILE,
        help="Prompts file location.",
    )
    return parser.parse_known_args()


def init_config(files: List[str], unknown_args: List[str]):
    """
    Initializes the configuration by loading configuration files and passed arguments.

    Args:
        files (List[str]): List of config files.
        unknown_args (List[str]): List of arguments (unknown for argparse).
    """
    cfg.load_files(files)
    cfg.load_args(unknown_args)


if __name__ == "__main__":
    """
    Entry point for the application. Parses command line arguments,
    initializes the configuration, and runs the main application.
    """
    try:
        # Parsing known and unknown arguments
        arguments, unknown_args = _parse_args()
        
        # Initialization of the configuration with the provided files and arguments
        init_config(
            files=[arguments.config_file, arguments.prompts_file],
            unknown_args=unknown_args
        )

        # Launching the main function of the application
        sys.exit(main())
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        sys.exit(1)
