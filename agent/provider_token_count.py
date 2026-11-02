"""``compression.provider_token_count`` (fork, Lara): the pre-request pressure check takes the provider's own count.

Real usage arrives only with a response; what was appended since and a transcript a compaction rewrote are estimated
at about half their size on CJK-with-JSON payloads. With the switch on, when the usage anchor does not cover the
request about to be sent, a provider that can count it does so and the figure stands in for real usage. Which
providers count, and how, lives in each provider's own module; every failure keeps the estimate.
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any, Optional

from agent.usage_anchor import _anchor_matches

logger = logging.getLogger(__name__)

PROVIDER_COUNTERS = {"anthropic_messages": "agent.anthropic_token_count"}


def _counting_timeout() -> Optional[float]:
    try:
        from hermes_cli.config import load_config_readonly
        section = load_config_readonly().get("compression")
    except Exception:
        return None
    if not isinstance(section, dict) or section.get("provider_token_count") is not True:
        return None
    timeout = section.get("provider_token_count_timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        return None
    return float(timeout)


def _anchor_covers(messages: list[dict[str, Any]], anchor: Any) -> bool:
    if not isinstance(anchor, dict) or not _anchor_matches(messages, anchor):
        return False
    since = messages[int(anchor["base_count"]):]
    return not since or (len(since) == 1 and isinstance(since[0], dict) and since[0].get("role") == "assistant")


def _positive_count(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def counted_request_pressure(agent: Any, messages: list[dict[str, Any]], api_messages: list[dict[str, Any]],
                             tools_for_api: Optional[list[dict[str, Any]]]) -> Optional[int]:
    """The provider's count of the request about to be sent, or None to keep the estimate."""
    timeout = _counting_timeout()
    counter = PROVIDER_COUNTERS.get(getattr(agent, "api_mode", ""))
    if timeout is None or counter is None or _anchor_covers(messages, getattr(agent, "_usage_anchor", None)):
        return None
    started = time.monotonic()
    try:
        counted = importlib.import_module(counter).count_request_tokens(agent, api_messages, tools_for_api,
                                                                         timeout=timeout)
    except Exception as exc:
        logger.warning("Provider token count failed (%s); keeping the estimate", type(exc).__name__)
        return None
    counted = _positive_count(counted)
    if counted is not None:
        logger.info("Provider token count: %d tokens in %.0f ms", counted, (time.monotonic() - started) * 1000)
        from agent.compaction_tail import record_provider_count
        record_provider_count(agent, counted, api_messages, tools_for_api)
    return counted
