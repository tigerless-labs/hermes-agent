"""A tool can hold its turn for a person's answer: the turn ends after the current tool round,
with no further model call and no reply, the way a run that returns approval interruptions does.

A plugin asks through ``PluginContext.hold_turn`` from inside a tool handler, naming the session
the handler was called for. The round's other calls still run. The turn then closes on a silence
marker and its result carries ``held``, which lets a turn that must be answered end without words.
A request made outside a running turn is dropped when the session's next turn starts, and a
subagent's turn is never held: its result still has to reach its parent.
"""
from __future__ import annotations

import threading
from typing import Any

EXIT_REASON = "held_for_person"
HELD_REPLY = "[SILENT]"
RESULT_KEY = "held"

_lock = threading.Lock()
_requested: set[str] = set()


def request(session_id: str) -> None:
    with _lock:
        _requested.add(session_id)


def forget(session_id: Any) -> None:
    with _lock:
        _requested.discard(session_id)


def take(agent: Any) -> bool:
    """Whether the round that just ran asked to hold this agent's turn; the request is used up."""
    session_id = getattr(agent, "session_id", None)
    with _lock:
        asked = session_id in _requested
        _requested.discard(session_id)
    return asked and not getattr(agent, "_parent_session_id", None)


def held(agent_result: Any) -> bool:
    return isinstance(agent_result, dict) and agent_result.get(RESULT_KEY) is True
