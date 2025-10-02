# pyright: basic
# pyright: reportAttributeAccessIssue=false

import logging
from typing import Annotated, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, Messages

from ai_server.config import APP_NAME

log = logging.getLogger(f"{APP_NAME}.{__name__}")

MAX_PATHS = 10
Messages_dict = Dict[str, Messages]


def add_path(left: str | list, right: str | list) -> Annotated:
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
    if len(merged) > MAX_PATHS:
        merged = merged[-MAX_PATHS:]
    if right[0] == "start":
        merged.append(list())
    for m in right:
        merged[-1].append(m)
    return merged


def add_messages_to_dict(left: Messages_dict, right: Messages_dict) -> Messages_dict:
    for k, v in right.items():
        left[k] = add_messages(
            left=left.get(k, []), right=v if isinstance(v, list) else [v]
        )
    return left


def add_summary(left: dict, right: dict) -> dict:
    for k, v in right.items():
        left[k] = v if isinstance(v, str) else ""
    return left


class State(TypedDict):
    # Messages kept in dict like
    # { "common": [HumanMessage, ],
    #   "school_tutor": [AIMessage, ],
    # }
    messages: Annotated[dict, add_messages_to_dict]
    user: str
    location: str
    additional_instructions: str
    llm_to_use: str
    last_used_llm: str
    last_messages: Dict
    thread_id: str
    path: Annotated[list, add_path]
    summary: Annotated[dict, add_summary]
