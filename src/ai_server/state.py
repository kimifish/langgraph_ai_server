
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages



def add_path(left: str|list, right: str|list) -> Annotated:
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    merged = left.copy()
    for m in right:
        merged.append(m)
    return merged


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user: str
    location: str
    additional_instructions: str
    llm_to_use: str
    thread_id: str
    path: Annotated[list, add_path]

