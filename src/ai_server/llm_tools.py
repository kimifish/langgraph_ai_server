# pyright: basic
# pyright: reportAttributeAccessIssue=false

from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
import logging
from langchain_core.tools import tool
import urllib3
from ai_server.tools.caldav import add_calendar_event, get_calendar_events
from ai_server.tools.kimihome import get_items, send_command
from ai_server.tools.music import get_music_by_tags, play_playlist, mpd_control

from ai_server.config import cfg, APP_NAME
log = logging.getLogger(f'{APP_NAME}.{__name__}')

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
        # create_playlist,
        play_playlist,    # Added new MPD tools
        mpd_control,      # Added new MPD tools
        get_current_datetime,
    ]
    
    smarthome_tools_list = [
        send_command,
        get_items,
        add_calendar_event,
        get_calendar_events,
        get_current_datetime,
    ]
    
    cfg.update('runtime.tools', {
        'common': [tavily_search],
        'smarthome': smarthome_tools_list,
        'smarthome_machine': smarthome_tools_list,
        'school_tutor': [tavily_search],
        'shell_assistant': [tavily_search],
        'code_assistant': [tavily_search],
        'music_assistant': music_tools_list,
        'music_machine': [tavily_search],
    })


if __name__ == '__main__':
    from ai_server.config import cfg
    cfg.print_config()
    from pprint import pprint
    pprint(send_command.args_schema.schema())  # pyright: ignore
    pprint(get_items.args_schema.schema())  # pyright: ignore
    pprint(add_calendar_event.args_schema.schema())  # pyright: ignore
    pprint(get_calendar_events.args_schema.schema())  # pyright: ignore
    pprint(get_current_datetime.args_schema.schema())  # pyright: ignore
    pprint(get_music_by_tags.args_schema.schema())  # pyright: ignore
