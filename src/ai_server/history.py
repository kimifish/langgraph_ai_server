# pyright: basic
# pyright: reportAttributeAccessIssue=false

from rich.pretty import pretty_repr
import logging
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import CheckpointTuple

import pymysql
from ai_server.config import cfg, APP_NAME
from ai_server.logs.utils import sep_line


log = logging.getLogger(f'{APP_NAME}.{__name__}')


def init_memory(memory_type: str = 'memory'):
    if memory_type == 'memory':
        cfg.update("runtime.memory", MemorySaver())
    elif memory_type == 'mariadb':
        from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
        conn = pymysql.connect(
            host=cfg.database.host,
            port=cfg.database.port,
            user=cfg.database.user,
            password=cfg.database.password,
            db=cfg.database.database,
        )
        saver = PyMySQLSaver(conn)
        saver.setup()
        cfg.update("runtime.memory", saver)
    else:
        raise ValueError(f"Неизвестный тип памяти: {memory_type}")

