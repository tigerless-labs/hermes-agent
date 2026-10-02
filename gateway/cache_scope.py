"""Move inbound attachments into their chat's cache directory (``terminal.docker_cache_scope: chat``).

Platform adapters cache attachments before the gateway knows the session, so the files land in the
shared cache roots. With chat scoping on, the gateway moves each one into
``<cache dir>/chats/<platform>-<chat>/`` as soon as the event arrives — the only directory that
chat's sandbox mounts — and rewrites the event's paths.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _platform_name(source: Any) -> str:
    platform = getattr(source, "platform", None)
    return str(getattr(platform, "value", platform) or "")


def scope_inbound_media(event: Any, source: Any) -> None:
    from hermes_constants import chat_cache_scope_enabled, chat_scope_slug, get_hermes_dir
    from tools.credential_files import _CACHE_DIRS

    media = getattr(event, "media_urls", None)
    if not media or source is None or not chat_cache_scope_enabled():
        return
    platform, chat_id = _platform_name(source), str(getattr(source, "chat_id", "") or "")
    if not platform or not chat_id:
        return
    slug = chat_scope_slug(platform, chat_id)
    roots = [get_hermes_dir(sub, old, chat_scoped=False).resolve() for sub, old in _CACHE_DIRS]
    for index, raw in enumerate(media):
        path = Path(raw)
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        root = next((candidate for candidate in roots if resolved.parent == candidate), None)
        if root is None or not resolved.is_file():
            continue
        target = root / "chats" / slug / resolved.name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(resolved, target)
        except OSError:
            logger.warning("Could not move inbound attachment %s into its chat cache", resolved.name)
            continue
        media[index] = str(target)
