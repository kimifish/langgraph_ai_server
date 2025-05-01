# pyright: basic
# pyright: reportAttributeAccessIssue=false

import logging
import sys
import time
import datetime
from typing import Dict, Optional

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.graph.graph import Runnable
from rich.pretty import pretty_repr
from openai import BadRequestError, PermissionDeniedError

from kimiUtils.killer import GracefulKiller
from ai_server.config import cfg, APP_NAME
from ai_server.llm_tools import init_tools
from ai_server.llms import init_models
from ai_server.graph import init_graph
from ai_server.logs.utils import log_diff, sep_line
from ai_server.models.userconfs import UserConfs, UserConf
from ai_server.history import init_memory

log = logging.getLogger(f'{APP_NAME}.{__name__}')

# Initialization of GracefulKiller for proper application termination
killer = GracefulKiller(kill_targets=[cfg.shutdown])

app = FastAPI(
    title="AI Server",
    version="0.3.0",
    description="Spin up a simple API server using LangChain's Runnable interfaces",
)


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
    init_models()
    init_tools()
    init_memory("mariadb")
    init_graph()
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


if __name__ == "__main__":
    """
    Entry point for the application. Parses command line arguments,
    initializes the configuration, and runs the main application.
    """
    try:
        # Launching the main function of the application
        sys.exit(main())
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        sys.exit(1)
