import logging
from rich.pretty import pretty_repr
from state import State

log = logging.getLogger('ai_server.utils')

def _log_state(state: State, tabs: int = 0):
    log.debug(pretty_repr(state))

