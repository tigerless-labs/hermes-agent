"""``gateway.wake_as_starter``: a background wake runs as the person who started the work.

Finished background work (a delegated subagent, a background command with a completion notice, a loop wakeup)
re-enters its conversation as a synthetic turn. Upstream takes that turn's whole source from the session's stored
origin, or from the session's last cached source: in a shared thread every wake ran as the thread's opener, whoever
dispatched the work, and a gateway that maps the turn's user to credentials lent the opener's to it. With the switch
on the wake keeps the resolved place and takes its person from the work: the user an async delegation or loop event
recorded at dispatch, else the user whose turn started the background process. Work that names nobody wakes as no
one, never as the origin. The stored origin is never rewritten.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

ASYNC_DELEGATION = "async_delegation"


def wake_as_starter() -> bool:
    from gateway.run import _load_gateway_config

    section = _load_gateway_config().get("gateway")
    return isinstance(section, dict) and section.get("wake_as_starter") is True


def _named(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _process_starter(evt: dict) -> tuple[Optional[str], Optional[str]]:
    session_id = _named(evt.get("session_id"))
    if session_id is None or evt.get("type") == ASYNC_DELEGATION:
        return None, None
    try:
        from tools.process_registry import process_registry
        session = process_registry.get(session_id)
    except Exception as error:
        logger.warning("No starter for background process %s (%s); its wake runs as no one", session_id, error)
        return None, None
    return _named(getattr(session, "watcher_user_id", None)), _named(getattr(session, "watcher_user_name", None))


def starter_of(evt: dict) -> tuple[Optional[str], Optional[str]]:
    user_id, user_name = _named(evt.get("user_id")), _named(evt.get("user_name"))
    if user_id is None:
        user_id, user_name = _process_starter(evt)
    return (user_id, user_name) if user_id is not None else (None, None)


def wakes_as_starter(build: Callable[[Any, dict], Any]) -> Callable[[Any, dict], Any]:
    @functools.wraps(build)
    def as_starter(runner: Any, evt: dict) -> Any:
        source = build(runner, evt)
        if source is None or not wake_as_starter():
            return source
        from gateway.session_identity import replace_source
        user_id, user_name = starter_of(evt if isinstance(evt, dict) else {})
        return replace_source(source, user_id=user_id, user_name=user_name, user_id_alt=None)

    return as_starter
