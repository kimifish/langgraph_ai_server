
# pyright: basic
# pyright: reportAttributeAccessIssue=false

import requests
import logging
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ai_server.config import cfg, APP_NAME
log = logging.getLogger(f'{APP_NAME}.{__name__}')


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
