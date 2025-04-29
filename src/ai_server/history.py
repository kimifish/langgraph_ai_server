# pyright: basic
# pyright: reportAttributeAccessIssue=false

from rich.pretty import pretty_repr
import logging
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from ai_server.config import cfg, APP_NAME
from ai_server.logs.utils import sep_line

log = logging.getLogger(f'{APP_NAME}.{__name__}')


def init_memory():
    cfg.update("runtime.memory", MemorySaver())  # TODO: Change to DB 

