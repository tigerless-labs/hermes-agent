"""A background turn: one a plugin injects to run without anything of it reaching the platform.

``PluginContext.inject_message(origin=..., background=True)`` marks the injected event; the turn
runs and is recorded like any other, and the gateway sends none of it (reply, streaming, progress,
typing, notices). What the person should see is the plugin's to send. Only a trusted injected
event can be background: human content never is.
"""

from contextvars import ContextVar
from typing import Any, Mapping, Optional

BACKGROUND_TURN = "hermes_background_turn"
_handling_background: ContextVar[bool] = ContextVar("hermes_handling_background_turn", default=False)


def background_event(event: Any) -> bool:
    return bool(getattr(event, "internal", False)) and (getattr(event, "metadata", None) or {}).get(
        BACKGROUND_TURN) is True


def background_metadata(event: Any) -> dict:
    """The display metadata that carries the mark from the event to its turn (and a queued follow-up)."""
    return {BACKGROUND_TURN: True} if background_event(event) else {}


def background_turn(display_metadata: Optional[Mapping[str, Any]]) -> bool:
    return (display_metadata or {}).get(BACKGROUND_TURN) is True


def enter_turn(event: Any) -> None:
    """Record, for the task handling ``event``, whether its turn is background; the notices sent
    before the agent runs (session reset, compression) have no event in reach and read this."""
    _handling_background.set(background_event(event))


def handling_background_turn() -> bool:
    return _handling_background.get()
