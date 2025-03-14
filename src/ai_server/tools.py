# pyright: basic
# pyright: reportAttributeAccessIssue=false

from datetime import datetime, timedelta
from langchain_community.tools.tavily_search import TavilySearchResults
import caldav
import os
import requests
import logging
from langchain_core.tools import tool
from kimiconfig import Config
import urllib3
from chromadb import HttpClient
from typing import List, Optional, Callable
import time
from mpd import MPDClient

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger('ai_server.' + __name__)
cfg = Config(use_dataclasses=True)


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CalendarManager(metaclass=Singleton):
    def __init__(self):
        self.url = cfg.calendar.url
        self.username = cfg.calendar.username
        # self.password = cfg.calendar.password
        self.password = os.getenv("CALDAV_PASSWORD")
        
        log.info(f'Initializing calendar manager with url: {self.url}, username: {self.username}, password: {self.password}')
        try:
            self.client = caldav.DAVClient(
                url=self.url,
                username=self.username,
                password=self.password,
                ssl_verify_cert=False
            )
            self.principal = self.client.principal()
            self.calendar = self.principal.calendar(name=cfg.calendar.name)   # или другое имя календаря
        except Exception as e:
            log.error(f"Ошибка инициализации календаря: {e}")
            raise


def get_calendar_manager():
    return CalendarManager()


@tool(parse_docstring=True)
def get_items(categories: str) -> list:
    """
    Возвращает json-список устройств(item), их описаний(label) и текущих состояний(state).

    Args:
        categories: A comma-separated list of tags to filter items by.
        Possible values are 'Light' - for illumination,
                         'Sensors' - for doors, windows state,
                         'Weather' - for weather,
                         'Battery' - for battery state,
                         'Terminals' - for computer terminals state,
                         'TV' - for TV state,
                         'HVAC' - for managing heating, watering, temperature and humidity sensors.
    """
    log.debug(f'Executing get_items with command "{categories}"')
    
    final_list = list()
    for tag in categories.split(','):
        payload = {'tags': f'Ai_{tag}',
                   'metadata': '.*',
                   'recursive': 'false',
                   'fields': 'type,name,label,state',
                   'staticDataOnly': 'false',
                   }
        headers = {'accept': 'application/json'}
        
        resp = requests.get(cfg.openhab.api_url,
                          headers=headers,
                          params=payload,
                          ).json()
        if len(resp) > 0:
            final_list.extend(resp)
            
    return final_list

@tool(parse_docstring=True)
def send_command(item: str, command: str) -> bool:
    """
    Executes smart home command, sending new state to item. True if successful, False if not.

    Args:
        item: Name of the item to send command to. e.g. "F2_LR_Light_1"
        command: The command to send. 
            For switches - [ON|OFF]. 
            For strings ending with LBM - [ON|OFF|AUTO]. 
            For type Color - values in HSB, e.g. "41,100,3"
            For players - [PLAY|PAUSE|NEXT|PREVIOUS|REWIND|FASTFORWARD]
            For strings ending with Application - ["YouTube"|"HDMI-3"] (приставка)
            For dimmers - values in percent, e.g. "50"
    """
    log.info(f'Executing {command} on {item}.')
    headers = {'accept': 'application/json',
               'Content-Type': 'text/plain',
               }
    x = requests.post(f'{cfg.openhab.api_url}/{item}',
                      headers=headers,
                      data=str(command).strip()
                      )
    log.info(x.text)
    # return str(x.status_code == 200)
    return x.status_code == 200

@tool(parse_docstring=True)
def add_calendar_event(event_data: str) -> str:
    """
    Добавляет новое событие в календарь.

    Args:
        event_data: Строка в формате "название|дата|время|продолжительность|описание"
            название - название события
            дата - дата в формате DD.MM.YYYY
            время - время начала в формате HH.MM
            продолжительность - в минутах
            описание - описание события (опционально)
            
            Пример - "День рождения Феди|11.02.2024|12.00|180|Не забыть купить подарок"

    Returns:
        Строка с подтверждением или описанием ошибки
    """
    try:
        # Разбираем входные данные
        parts = event_data.split('|')
        if len(parts) < 4:
            return "Ошибка: недостаточно данных. Нужно указать название, дату, время и продолжительность."
        
        title = parts[0].strip()
        date_str = parts[1].strip()
        time_str = parts[2].strip()
        duration = int(parts[3].strip())
        description = parts[4].strip() if len(parts) > 4 else ""

        # Преобразуем строки в datetime
        start_dt = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H.%M")
        end_dt = start_dt + timedelta(minutes=duration)
        calendar_manager = get_calendar_manager()

        # Создаем событие
        event = calendar_manager.calendar.save_event(
            dtstart=start_dt,
            dtend=end_dt,
            summary=title,
            description=description
        )

        log.info(f"Добавлено событие: {title} на {start_dt}")
        return f"Событие успешно добавлено в календарь: {title} на {date_str} {time_str}"

    except ValueError as e:
        log.error(f"Ошибка формата данных: {e}")
        return f"Ошибка формата данных: проверьте правильность ввода даты, времени и продолжительности"
    except Exception as e:
        log.error(f"Ошибка при добавлении события: {e}")
        return f"Ошибка при добавлении события в календарь: {str(e)}"

@tool(parse_docstring=True)
def get_calendar_events(date_str: str) -> str:
    """
    Возвращает список событий в календаре на указанную дату.

    Args:
        date_str: Дата в формате DD.MM.YYYY, например "15.03.2024"
    
    Returns:
        Строка со списком событий или сообщение об отсутствии событий
    """
    try:
        # Преобразуем строку в datetime
        target_date = datetime.strptime(date_str, "%d.%m.%Y")
        
        # Устанавливаем временной интервал на весь день
        start = target_date.replace(hour=0, minute=0, second=0)
        end = target_date.replace(hour=23, minute=59, second=59)
        
        # Получаем события
        calendar_manager = get_calendar_manager()
        events = calendar_manager.calendar.search(
            start=start,
            end=end,
            event=True
        )
        
        if not events:
            return f"На {date_str} событий не найдено"
        
        # Форматируем результат
        result = [f"События на {date_str}:"]
        for event in events:
            event_data = event.instance.vevent
            summary = event_data.summary.value
            start_time = event_data.dtstart.value.strftime("%H:%M")
            end_time = event_data.dtend.value.strftime("%H:%M")
            description = getattr(event_data, 'description', None)
            description_text = f" - {description.value}" if description else ""
            
            result.append(f"- {start_time}-{end_time}: {summary}{description_text}")
        
        return "\n".join(result)
        
    except ValueError:
        return "Ошибка: неверный формат даты. Используйте формат DD.MM.YYYY"
    except Exception as e:
        log.error(f"Ошибка при получении событий: {e}")
        return f"Ошибка при получении событий из календаря: {str(e)}"

@tool(parse_docstring=True)
def get_current_datetime() -> str:
    """
    Возвращает текущие дату и время.

    Returns:
        Строка с текущей датой и временем в формате "DD.MM.YYYY HH:MM"
    """
    current = datetime.now()
    return current.strftime("%d.%m.%Y %H:%M")

@tool(parse_docstring=True)
def get_music_by_tags(tags: str) -> List[str]:
    """
    Queries ChromaDB for music files matching the given tags.

    Args:
        tags: Comma-separated list of music tags/descriptions.
              Example - "rock, guitar solo, energetic" or "ambient, calm, meditation"
    
    Returns:
        List of matching music filenames, sorted by relevance
    """
    try:
        client = HttpClient(
            host=cfg.music.chroma_host,
            port=cfg.music.chroma_port,
        )
        
        query_tags = [tag.strip().lower() for tag in tags.split(',')]
        results = client.query(
            query_texts=query_tags,
            n_results=10,
            include=['documents']
        )
        
        if results and hasattr(results, 'documents'):
            # Flatten the list of lists into a single list
            return [doc for sublist in results.documents for doc in sublist]
        return []

    except Exception as e:
        log.error(f"Error querying music database: {e}")
        return []
    finally:
        if 'client' in locals():
            client.close()

@tool(parse_docstring=True)
def create_playlist(songs: List[str], name: str = "") -> str:
    """
    Creates a playlist from the given songs.

    Args:
        songs: List of song filenames to add to playlist
        name: Optional playlist name. If empty, generates timestamp-based name.
    
    Returns:
        Playlist name or error message
    """
    try:
        playlist_name = name or f"playlist_{int(time.time())}"
        # Add your playlist creation logic here
        return f"Created playlist '{playlist_name}' with {len(songs)} songs"
    except Exception as e:
        log.error(f"Error creating playlist: {e}")
        return f"Failed to create playlist: {str(e)}"

@tool(parse_docstring=True)
def play_playlist(playlist_name: str) -> str:
    """
    Plays a playlist on the MPD server.

    Args:
        playlist_name: Name of the playlist to play

    Returns:
        Status message indicating success or failure
    """
    try:
        client = MPDClient()
        client.connect(cfg.music.mpd.host, cfg.music.mpd.port)
        
        # Handle password if configured
        if hasattr(cfg.music.mpd, 'password'):
            if cfg.music.mpd.password == '.env':
                password = os.getenv('MPD_PASSWORD')
                if password:
                    client.password(password)
            else:
                client.password(cfg.music.mpd.password)

        # Clear current playlist
        client.clear()
        
        # Load and play the playlist
        client.load(playlist_name)
        client.play()
        
        # Get current song info
        current = client.currentsong()
        client.close()
        client.disconnect()
        
        return f"Playing playlist '{playlist_name}'. Now playing: {current.get('title', 'Unknown')}"

    except Exception as e:
        log.error(f"Error playing playlist on MPD: {e}")
        return f"Failed to play playlist: {str(e)}"

@tool(parse_docstring=True)
def mpd_control(command: str) -> str:
    """
    Controls MPD playback.

    Args:
        command: One of: play, pause, stop, next, previous, shuffle, clear
    
    Returns:
        Status message
    """
    try:
        client = MPDClient()
        client.connect(cfg.music.mpd.host, cfg.music.mpd.port)
        
        # Handle password if configured
        if hasattr(cfg.music.mpd, 'password'):
            if cfg.music.mpd.password == '.env':
                password = os.getenv('MPD_PASSWORD')
                if password:
                    client.password(password)
            else:
                client.password(cfg.music.mpd.password)

        command = command.lower()
        if command == 'play':
            client.play()
            status = "Playing"
        elif command == 'pause':
            client.pause()
            status = "Paused"
        elif command == 'stop':
            client.stop()
            status = "Stopped"
        elif command == 'next':
            client.next()
            status = "Skipped to next track"
        elif command == 'previous':
            client.previous()
            status = "Returned to previous track"
        elif command == 'shuffle':
            client.shuffle()
            status = "Playlist shuffled"
        elif command == 'clear':
            client.clear()
            status = "Playlist cleared"
        else:
            status = f"Unknown command: {command}"

        # Get current song info if playing
        if command in ['play', 'next', 'previous']:
            current = client.currentsong()
            if current:
                status += f". Now playing: {current.get('title', 'Unknown')}"

        client.close()
        client.disconnect()
        return status

    except Exception as e:
        log.error(f"Error controlling MPD: {e}")
        return f"Failed to execute command: {str(e)}"

def _init_tools():
    tavily_search = TavilySearchResults(max_results=2)
    
    # Define tool lists for different assistants
    music_tools_list = [
        get_music_by_tags,
        create_playlist,
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
    })



if __name__ == '__main__':
    cfg = Config(file='./config.yaml')
    cfg.print_config()
    from pprint import pprint
    pprint(send_command.args_schema.schema())  # pyright: ignore
    pprint(get_items.args_schema.schema())  # pyright: ignore
    pprint(add_calendar_event.args_schema.schema())  # pyright: ignore
    pprint(get_calendar_events.args_schema.schema())  # pyright: ignore
    pprint(get_current_datetime.args_schema.schema())  # pyright: ignore
    pprint(get_music_by_tags.args_schema.schema())  # pyright: ignore
