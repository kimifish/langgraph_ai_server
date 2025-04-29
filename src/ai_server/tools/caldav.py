# pyright: basic
# pyright: reportAttributeAccessIssue=false

from datetime import datetime, timedelta
import caldav
import os
import logging
from langchain_core.tools import tool
import urllib3

from ai_server.config import cfg, APP_NAME
log = logging.getLogger(f'{APP_NAME}.{__name__}')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
        self.password = os.getenv("CALENDAR_PASSWORD")
        
        log.info(f'Initializing calendar manager with url: {self.url}, username: {self.username}, password: {self.password}')
        try:
            self.client = caldav.DAVClient(
                url=self.url,
                username=self.username,
                password=self.password,
                ssl_verify_cert=False
            )
            self.principal = self.client.principal()
            self.calendar = self.principal.calendar(name=cfg.calendar.name)   # or another calendar name
        except Exception as e:
            log.error(f"Calendar initialization error: {e}")
            raise


def get_calendar_manager():
    return CalendarManager()


@tool(parse_docstring=True)
def add_calendar_event(event_data: str) -> str:
    """
    Adds a new event to the calendar.

    Args:
        event_data: String in format "title|date|time|duration|description"
            title - event title
            date - date in DD.MM.YYYY format
            time - start time in HH.MM format
            duration - in minutes
            description - event description (optional)
            
            Example - "День рождения Феди|11.02.2024|12.00|180|Не забыть купить подарок"

    Returns:
        Confirmation string or error description
    """
    try:
        # Parse input data
        parts = event_data.split('|')
        if len(parts) < 4:
            return "Error: insufficient data. Title, date, time and duration are required."
        
        title = parts[0].strip()
        date_str = parts[1].strip()
        time_str = parts[2].strip()
        duration = int(parts[3].strip())
        description = parts[4].strip() if len(parts) > 4 else ""

        # Convert strings to datetime
        start_dt = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H.%M")
        end_dt = start_dt + timedelta(minutes=duration)
        calendar_manager = get_calendar_manager()

        # Create event
        event = calendar_manager.calendar.save_event(
            dtstart=start_dt,
            dtend=end_dt,
            summary=title,
            description=description
        )

        log.info(f"Added event: {title} at {start_dt}")
        return f"Event successfully added to calendar: {title} on {date_str} {time_str}"

    except ValueError as e:
        log.error(f"Data format error: {e}")
        return f"Data format error: please check date, time and duration format"
    except Exception as e:
        log.error(f"Error adding event: {e}")
        return f"Error adding event to calendar: {str(e)}"


@tool(parse_docstring=True)
def get_calendar_events(date_str: str) -> str:
    """
    Returns list of calendar events for the specified date.

    Args:
        date_str: Date in DD.MM.YYYY format, example "15.03.2024"
    
    Returns:
        String with list of events or message about no events
    """
    try:
        # Convert string to datetime
        target_date = datetime.strptime(date_str, "%d.%m.%Y")
        
        # Set time interval for whole day
        start = target_date.replace(hour=0, minute=0, second=0)
        end = target_date.replace(hour=23, minute=59, second=59)
        
        # Get events
        calendar_manager = get_calendar_manager()
        events = calendar_manager.calendar.search(
            start=start,
            end=end,
            event=True
        )
        
        if not events:
            return f"No events found for {date_str}"
        
        # Format result
        result = [f"Events for {date_str}:"]
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
        return "Error: invalid date format. Use DD.MM.YYYY format"
    except Exception as e:
        log.error(f"Error getting events: {e}")
        return f"Error retrieving events from calendar: {str(e)}"
