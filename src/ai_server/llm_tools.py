# pyright: basic
# pyright: reportAttributeAccessIssue=false

from datetime import datetime
from dataclasses import asdict
from langchain_core.documents.base import Blob
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mcp_adapters.client import MultiServerMCPClient
import logging
from langchain_core.tools import tool
import urllib3
from ai_server.tools.caldav import add_calendar_event, get_calendar_events
from ai_server.tools.music import (
    get_music_by_tags,
    play_playlist,
    mpd_control,
    get_metadata_list,
)

from ai_server.config import cfg, APP_NAME


log = logging.getLogger(f"{APP_NAME}.{__name__}")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@tool(parse_docstring=True)
def get_current_datetime() -> str:
    """
    Возвращает текущие дату и время.

    Returns:
        Строка с текущей датой и временем в формате "DD.MM.YYYY HH:MM"
    """
    current = datetime.now()
    return current.strftime("%d.%m.%Y %H:%M")


def init_tools():
    tavily_search = TavilySearchResults(max_results=2)

    # Define tool lists for different assistants
    music_tools_list = [
        get_music_by_tags,
        get_metadata_list,
        # create_playlist,
        play_playlist,
        mpd_control,
        get_current_datetime,
    ]

    smarthome_tools_list = [
        # send_command,
        # get_items,
        add_calendar_event,
        get_calendar_events,
        get_current_datetime,
    ]

    cfg.update(
        "runtime.tools",
        {
            "common": [tavily_search],
            "smarthome": smarthome_tools_list,
            "smarthome_machine": smarthome_tools_list,
            "school_tutor": [tavily_search],
            "shell_assistant": [tavily_search],
            "code_assistant": [tavily_search],
            "music_assistant": music_tools_list,
            "music_machine": [tavily_search],
        },
    )


async def init_mcp_tools(client: MultiServerMCPClient):
    tools_list: dict = asdict(cfg.runtime.tools)

    for name, mcp_list in client.server_name_to_tools.items():
        if name in tools_list.keys() and isinstance(tools_list[name], list):
            tools_list[name] += mcp_list
        else:
            tools_list[name] = mcp_list

    cfg.update("runtime.tools", tools_list)


async def init_mcp_resources(client: MultiServerMCPClient):
    resources: list[Blob] = []
    for name, _ in client.server_name_to_tools.items():
        resources += await client.get_resources(name)

    cfg.update("runtime.resources", resources)


async def init_mcp_prompts(client: MultiServerMCPClient):
    pass
