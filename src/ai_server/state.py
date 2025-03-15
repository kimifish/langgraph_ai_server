# pyright: basic
# pyright: reportAttributeAccessIssue=false

import logging
from rich.traceback import install as install_rich_traceback
from typing import Annotated, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, Messages
from kimiconfig import Config

cfg = Config()
log = logging.getLogger('ai_server.state')
install_rich_traceback(show_locals=True)

Messages_dict = Dict[str, Messages]


def add_path(left: str|list, right: str|list) -> Annotated:
    """Keeps path as list of sublists. Starts new sublist if 'start' passed as right value.

    Args:
        left (str | list): List up till now
        right (str | list): Value to add

    Returns:
        Annotated: New paths list with value added.
    """
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    merged = left.copy()
    if right[0] == 'start':
        merged.append(list())
    for m in right:
        merged[-1].append(m)
    return merged

def add_messages_to_dict(left: Messages_dict, right: Messages_dict) -> Messages_dict:
    for k, v in right.items():
        # log.debug(f'{k=}, {v=}')
        left[k] = add_messages(left=left.get(k, []), right = v if isinstance(v, list) else [v] )
    return left

def add_summary(left: dict, right: dict) -> dict:
    for k, v in right.items():
        # log.debug(f'{k=}, {v=}')
        left[k] = v if isinstance(v, str) else ''
    return left


class State(TypedDict):
    # Messages kept in dict like 
    # { "common": [HumanMessage, ],
    #   "school_tutor": [AIMessage, ],
    # }
    messages: Annotated[ dict, add_messages_to_dict]
    user: str
    location: str
    additional_instructions: str
    llm_to_use: str
    last_used_llm: str
    thread_id: str
    path: Annotated[list, add_path]
    summary: Annotated[ dict, add_summary]

