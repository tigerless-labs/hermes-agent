"""Host routing for ``PluginContext.interrupt_session``: the live messaging gateway — the owner of the
published gateway message host — stops the running turn of the session at a plugin-supplied
``origin``, in the plugin's own profile, the way ``/stop`` does."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hermes_cli.config import load_config_readonly
from hermes_cli.plugins_loader import _plugin_home_scope
from hermes_cli.plugins_state import _plugin_settings_entry

logger = logging.getLogger("hermes_cli.plugins")

ALLOW_GATEWAY_INTERRUPT = "allow_gateway_interrupt"


def gateway_interrupt_allowed(plugin_id: str, plugin_home: Path) -> bool:
    """Whether the plugin's own profile config sets ``allow_gateway_interrupt: true`` (never the
    calling thread's ambient home); an unreadable config denies."""
    try:
        with _plugin_home_scope(plugin_home):
            cfg = load_config_readonly() or {}
    except Exception:
        return False
    return (_plugin_settings_entry(cfg, plugin_id) or {}).get(ALLOW_GATEWAY_INTERRUPT) is True


def interrupt_plugin_session(ctx: Any, origin: Any) -> bool:
    manager = ctx._manager
    if not isinstance(origin, Mapping):
        logger.warning("interrupt_session: origin must be a mapping in the SessionSource.to_dict() shape")
        return False
    if not gateway_interrupt_allowed(ctx.plugin_id, manager.home_path):
        logger.warning("interrupt_session: denied for plugin %s; set plugins.entries.%s.%s: true to allow it",
                       ctx.plugin_id, ctx.plugin_id, ALLOW_GATEWAY_INTERRUPT)
        return False
    host = manager._gateway_message_injector
    interrupt = getattr(host[0], "_interrupt_plugin_session", None) if host is not None else None
    if not callable(interrupt):
        logger.warning("interrupt_session: no live gateway is available")
        return False
    try:
        return bool(interrupt(origin=dict(origin), plugin_home=manager.home_path, plugin_id=ctx.plugin_id))
    except Exception:
        logger.warning("interrupt_session: gateway interrupt failed for plugin %s", ctx.plugin_id, exc_info=True)
        return False
