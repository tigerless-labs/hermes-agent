"""Host routing for ``PluginContext.steer_session``: the live messaging gateway — the owner of the
published gateway message host — gives the running turn of the session at a plugin-supplied ``origin``,
in the plugin's own profile, the plugin's words as a mid-turn message, only when that turn acts for the
origin's own sender."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hermes_cli.config import load_config_readonly
from hermes_cli.plugins_loader import _plugin_home_scope
from hermes_cli.plugins_state import _plugin_settings_entry

logger = logging.getLogger("hermes_cli.plugins")

ALLOW_GATEWAY_STEER = "allow_gateway_steer"


def gateway_steer_allowed(plugin_id: str, plugin_home: Path) -> bool:
    """Whether the plugin's own profile config sets ``allow_gateway_steer: true`` (never the calling
    thread's ambient home); an unreadable config denies."""
    try:
        with _plugin_home_scope(plugin_home):
            cfg = load_config_readonly() or {}
    except Exception:
        return False
    return (_plugin_settings_entry(cfg, plugin_id) or {}).get(ALLOW_GATEWAY_STEER) is True


def steer_plugin_session(ctx: Any, content: Any, origin: Any) -> bool:
    manager = ctx._manager
    if not isinstance(content, str) or not content.strip():
        logger.warning("steer_session: content must be non-empty text")
        return False
    if not isinstance(origin, Mapping):
        logger.warning("steer_session: origin must be a mapping in the SessionSource.to_dict() shape")
        return False
    if not gateway_steer_allowed(ctx.plugin_id, manager.home_path):
        logger.warning("steer_session: denied for plugin %s; set plugins.entries.%s.%s: true to allow it",
                       ctx.plugin_id, ctx.plugin_id, ALLOW_GATEWAY_STEER)
        return False
    host = manager._gateway_message_injector
    steer = getattr(host[0], "_steer_plugin_session", None) if host is not None else None
    if not callable(steer):
        logger.warning("steer_session: no live gateway is available")
        return False
    try:
        return bool(steer(origin=dict(origin), content=content, plugin_home=manager.home_path,
                          plugin_id=ctx.plugin_id))
    except Exception:
        logger.warning("steer_session: gateway steer failed for plugin %s", ctx.plugin_id, exc_info=True)
        return False
