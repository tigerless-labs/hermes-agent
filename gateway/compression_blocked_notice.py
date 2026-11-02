"""``compression.blocked_notices``: a deployment keeps the compression-blocked warning out of chat.

The warning asks the reader to run ``/new`` or ``/compress``. On a shared deployment the people in the chat neither
own the session nor can act on it, so with the switch off chat surfaces drop it and the gateway log keeps it.
Unset or true keeps upstream's always-visible failure notice.
"""

from __future__ import annotations

import logging
import re

from agent.conversation_compression import CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE

logger = logging.getLogger(__name__)

_BLOCKED_WARNING = re.compile(
    ".+?".join(re.escape(part) for part in re.split(r"\{[^{}]*\}", CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE)))


def blocked_notices_shown() -> bool:
    from gateway.run import _load_gateway_config

    section = _load_gateway_config().get("compression")
    return not (isinstance(section, dict) and section.get("blocked_notices") is False)


def withheld_from_chat(text: str) -> bool:
    if not _BLOCKED_WARNING.search(text) or blocked_notices_shown():
        return False
    logger.warning("Compression-blocked notice kept out of chat: %s", text)
    return True
