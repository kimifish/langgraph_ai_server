# pyright: basic
# pyright: reportAttributeAccessIssue=false

import requests
import logging
from langchain_core.tools import tool

from ai_server.config import cfg, APP_NAME

log = logging.getLogger(f"{APP_NAME}.{__name__}")


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
    log.info(f"Executing {command} on {item}.")
    headers = {
        "accept": "application/json",
        "Content-Type": "text/plain",
    }
    x = requests.post(
        f"{cfg.openhab.api_url}/{item}", headers=headers, data=str(command).strip()
    )
    log.info(x.text)
    # return str(x.status_code == 200)
    return x.status_code == 200
