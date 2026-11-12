"""Host side of ``PluginContext.hold_turn``: a plugin whose own profile config allows it asks the
agent loop to end the running turn of a session after its current tool round (``agent.turn_hold``)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hermes_cli.config import load_config_readonly
from hermes_cli.plugins_loader import _plugin_home_scope
from hermes_cli.plugins_state import _plugin_settings_entry

logger = logging.getLogger("hermes_cli.plugins")

ALLOW_TURN_HOLD = "allow_turn_hold"


def turn_hold_allowed(plugin_id: str, plugin_home: Path) -> bool:
    """Whether the plugin's own profile config sets ``allow_turn_hold: true`` (never the calling
    thread's ambient home); an unreadable config denies."""
    try:
        with _plugin_home_scope(plugin_home):
            cfg = load_config_readonly() or {}
    except Exception:
        return False
    return (_plugin_settings_entry(cfg, plugin_id) or {}).get(ALLOW_TURN_HOLD) is True


def hold_plugin_turn(ctx: Any, session_id: Any) -> bool:
    if not isinstance(session_id, str) or not session_id:
        logger.warning("hold_turn: session_id must be the id the tool handler was called with")
        return False
    if not turn_hold_allowed(ctx.plugin_id, ctx._manager.home_path):
        logger.warning("hold_turn: denied for plugin %s; set plugins.entries.%s.%s: true to allow it",
                       ctx.plugin_id, ctx.plugin_id, ALLOW_TURN_HOLD)
        return False
    from agent import turn_hold
    turn_hold.request(session_id)
    return True
