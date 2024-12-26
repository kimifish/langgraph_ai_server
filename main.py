# pyright: basic 
# pyright: ignore[reportAttributeAccessIssue]

import logging
import sys
import time
import os
import argparse
from typing import Annotated, Literal, Callable, List
from rich.markdown import Markdown
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.pretty import pretty_repr
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from kimiconfig import Config
from kimiUtils.killer import GracefulKiller
from tools import _init_tools
from llms import _init_models
from graph import _init_graph
from fastapi import FastAPI, UploadFile, File


DEFAULT_CONFIG_FILE = f'{os.getenv("HOME")}/.config/ai_server/config.yaml'
PROMPTS_FILE = f'{os.getenv("HOME")}/.config/ai_server/prompts.yaml'

load_dotenv()

logging.basicConfig(level='NOTSET',
                    format='%(message)s',
                    datefmt='[%X]',
                    handlers=[RichHandler()],
                    )
parent_logger = logging.getLogger('ai_server')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)
log = logging.getLogger('ai_server.main')
install_rich_traceback(show_locals=True)

cfg = Config(use_dataclasses=True)
console = Console(record=True)
cfg.update('runtime.console', console)

killer = GracefulKiller(kill_targets=[
    cfg.shutdown,
    ]
)

app = FastAPI(
    title="AI Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)


class UserConfs:
    def __init__(self):
        self.user_dict = {}

    def add(
            self, 
            thread_id: str = "", 
            user: str = "Undefined", 
            location: str = "Undefined", 
            additional_instructions: str = "", 
            llm_to_use = "",
            callback: Callable = console.print
    ):
        if thread_id == "":
            thread_id = user + '_' + str(round(time.time()))

        self.user_dict[thread_id] = {
            "thread_id": thread_id,
            "user": user, 
            "location": location, 
            "additional_instructions": additional_instructions, 
            "llm_to_use": llm_to_use,
            "callback": callback
        }
        log.debug(self)
        return self.user_dict[thread_id]

    def get_lgconf(self, thread_id: str):
        return {"configurable": {"thread_id": thread_id}}

    def get_(self, thread_id: str):
        return self.user_dict[thread_id]

    def get_all(self):
        return self.user_dict

    def exists(self, thread_id):
        return thread_id in self.user_dict

    def __str__(self):
        result = "UserConfs:\n"
        for k, v in self.user_dict.items():
            result += f"{k}: {v}\n"
        return result


def _init_memory():
    cfg.update('runtime.memory', MemorySaver())  # TODO: Добавить в БД


def print_answer(graph, user_input: str, config):
    answer = get_graph_answer(graph, user_input, config)
    console.print("[yellow4]bot: [/yellow4]", end='')
    console.print(Markdown(answer))


def get_graph_answer(graph, user_input: str, config):
    lgconf = {'configurable': {'thread_id': config['thread_id']}}

    for event in graph.stream(
        {
            'messages': [ ("system", "You are powerful assistant!"), ("user", user_input), ],
            'user': config['user'],
            'location': config['location'],
            'additional_instructions': config['additional_instructions'],
            'llm_to_use': config['llm_to_use'],
            'thread_id': config['thread_id'],
        }, 
        lgconf, 
        stream_mode="values"
    ):
        log.debug(f'-------↓↓↓-------{event.get("path", ["-"])[-1]}-------↓↓↓--------')
        log.debug(pretty_repr(event))
        # console.input(f'-----------------{event.get("path", ["-"])[-1]}-----------------')

        # Checking stream reached final node:
        # if event.get('path', ['-'])[-1] == 'final':
        # if "messages" in event and isinstance(event["messages"][-1], AIMessage):
    log.info(f"answer: {event['messages'][-1].content}")
    return event["messages"][-1].content

    return "Oops. You shouldn't see this."


# For test purposes only
def local_loop():
    while not killer.kill_now:
        try:
            user_input = console.input("[#44ff33] @ : ")
            # user_input = "hi"
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            print_answer(cfg.runtime.graph, user_input, cfg.runtime.user_confs.add(user='local_terminal'))  # pyright: ignore[reportAttributeAccessIssue]
        except KeyboardInterrupt:
            break
    cfg.shutdown()


@app.get("/test")
def test_endpoint(
    message: str = "Hello, world!",
    thread_id: str = "",
    user = "Undefined",
    location = "Undefined",
    additional_instructions = "",
    llm_to_use = "",
    ):
    user_confs: UserConfs = cfg.runtime.user_confs
    if user_confs.exists(thread_id):
        config = user_confs.get_(thread_id)
    else:
        config = user_confs.add(
                thread_id=thread_id,
                user=user,
                location=location,
                additional_instructions=additional_instructions,
                llm_to_use=llm_to_use,
        )
    answer = get_graph_answer(
            graph=cfg.runtime.graph,  # pyright: ignore[reportAttributeAccessIssue]
            user_input=message, 
            config=config,
        )
    return {"answer": answer}

def main():
    _init_logs()
    _init_models()
    _init_tools()
    _init_memory()
    _init_graph()
    cfg.update('runtime.mood', 'slightly depressed')  # TODO: Пока тут ничего.
    cfg.update('runtime.user_confs', UserConfs())
    cfg.runtime.user_confs.add(
        user="local_terminal", 
        location="F2_LR", 
        additional_instructions="Используй Markdown для форматирования ответа.", 
    )

    # from threading import Thread
    # t = Thread(target=local_loop)
    # t.start()

    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # workers=1,
        # log_config=log_config,
        # lifespan=lifespan,
        # ssl_keyfile="key.pem",        # если нужен HTTPS
        # ssl_certfile="cert.pem",
        # reload=True,                  # для автоматической перезагрузки при изменении кода
    )
    # t.join()


def draw_graph(graph):
    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Error drawing graph: {e}")
        return

def _init_logs():
    parent_logger.setLevel(cfg.logging.level, )  # pyright: ignore[reportAttributeAccessIssue]

# def _init_mqtt():
#     mqtt = MQTT(connect_on_init=False, host=cfg.mqtt['server'], port=cfg.mqtt['port'], client_id="i3-commander")
#     cfg.runtime_mqtt = {}
#     cfg.runtime_mqtt['client'] = mqtt
#     return mqtt

def _parse_args():
    parser = argparse.ArgumentParser(
        prog='ai_server',
        description='ai_server') 
    # parser.add_argument('-s', '--socket',
    #                     dest='socket_path',
    #                     default='',
    #                     help='Sets socket path.'
    #                     )
    parser.add_argument('-c', '--config',
                        dest='config_file',
                        default=DEFAULT_CONFIG_FILE,
                        help='Configuration file location.'
                        )
    parser.add_argument('-p', '--prompts',
                        dest='prompts_file',
                        default=PROMPTS_FILE,
                        help='Prompts file location.'
                        )
    return parser.parse_known_args()


def _init_config(files: List[str], unknown_args: List[str]):
    cfg.load_files(files)
    cfg.load_args(unknown_args)


if __name__ == '__main__':
    arguments, unknown_args = _parse_args()
    _init_config([arguments.config_file, arguments.prompts_file], unknown_args, )
    sys.exit(main())

