# pyright: basic
# pyright: reportAttributeAccessIssue=false

import logging
import datetime
from typing import Dict, Optional
from rich.pretty import pretty_repr
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from ai_server.config import cfg, APP_NAME

log = logging.getLogger(f'{APP_NAME}.{__name__}')


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
