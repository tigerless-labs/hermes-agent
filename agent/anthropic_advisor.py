"""Anthropic's advisor tool on the main conversation (Lara fork addition).

The executor model consults a stronger advisor model inside the same Messages request. Off unless
``advisor.model`` is configured; only on the native Anthropic API-key route and for Claude executors.
The advisor's result is opaque (encrypted for newer advisors) and replays verbatim in its place,
because the thinking blocks after it are signed against it. A request can end with ``pause_turn``
while the advisor is still working; it is resent with the partial reply until the advisor answers.
The advisor's tokens are not in the request's top-level usage, so they are logged on their own line.
"""

import copy
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

ADVISOR_TOOL_TYPE = "advisor_20260301"
ADVISOR_TOOL_NAME = "advisor"
ADVISOR_BETA = "advisor-tool-2026-03-01"
ADVISOR_BLOCK_TYPES = frozenset({"server_tool_use", "advisor_tool_result"})
ADVISOR_ITERATION = "advisor_message"
PAUSED = "pause_turn"
DEFAULT_RESUMES = 1


def advisor_tool(config: Optional[dict]) -> Optional[dict[str, Any]]:
    """The advisor tool definition from ``config["advisor"]``, or None when no advisor model is set."""
    section = (config or {}).get("advisor") or {}
    model = str(section.get("model") or "").strip()
    if not model:
        return None
    tool: dict[str, Any] = {"type": ADVISOR_TOOL_TYPE, "name": ADVISOR_TOOL_NAME, "model": model}
    for key in ("max_uses", "max_tokens"):
        if section.get(key):
            tool[key] = int(section[key])
    if section.get("caching"):
        tool["caching"] = {"type": "ephemeral", "ttl": str(section["caching"])}
    return tool


def apply_advisor(kwargs: dict, *, model: str, base_url: Optional[str], is_oauth: bool, tool: Optional[dict]) -> None:
    """Add the advisor tool and its beta to built Messages kwargs, in place."""
    from agent.anthropic_adapter import (
        _beta_header, _common_betas_for_base_url, _is_claude_model, _is_third_party_anthropic_endpoint,
    )
    if not tool or is_oauth or "tools" not in kwargs:
        return
    if _is_third_party_anthropic_endpoint(base_url) or not _is_claude_model(model):
        return
    kwargs["tools"] = [*kwargs["tools"], dict(tool)]
    headers = dict(kwargs.get("extra_headers") or {})
    current = [b for b in headers.get("anthropic-beta", "").split(",") if b] or _common_betas_for_base_url(base_url)
    kwargs["extra_headers"] = {**headers, **_beta_header([*current, ADVISOR_BETA])}


def advisor_in(kwargs: dict) -> Optional[dict]:
    return next((t for t in kwargs.get("tools") or [] if isinstance(t, dict) and t.get("type") == ADVISOR_TOOL_TYPE),
                None)


def carries_advice(blocks: list[dict]) -> bool:
    return any(isinstance(b, dict) and b.get("type") in ADVISOR_BLOCK_TYPES for b in blocks)


def replay_server_tool_use(b: dict) -> Optional[dict]:
    if not b.get("id"):
        return None
    return {"type": "server_tool_use", "id": b["id"], "name": b.get("name", ADVISOR_TOOL_NAME), "input": b.get("input") or {}}


def replay_advisor_tool_result(b: dict) -> Optional[dict]:
    if not b.get("tool_use_id") or not isinstance(b.get("content"), dict):
        return None
    return {"type": "advisor_tool_result", "tool_use_id": b["tool_use_id"], "content": copy.deepcopy(b["content"])}


def _wire_blocks(blocks: list[Any]) -> list[dict]:
    from agent.anthropic_message_convert import _sanitize_replay_block, _to_plain_data
    return [clean for b in blocks if (clean := _sanitize_replay_block(_to_plain_data(b))) is not None]


def resume_paused(send: Callable[[dict], Any], kwargs: dict, reply: Any) -> Any:
    """Resend a turn the advisor paused, with the partial reply, until it ends; join the parts."""
    tool = advisor_in(kwargs)
    if tool is None or getattr(reply, "stop_reason", None) != PAUSED:
        return reply
    parts = [reply]
    while getattr(parts[-1], "stop_reason", None) == PAUSED and len(parts) <= tool.get("max_uses", DEFAULT_RESUMES):
        partial = [b for part in parts for b in part.content]
        parts.append(send({**kwargs, "messages": [*kwargs["messages"],
                                                  {"role": "assistant", "content": _wire_blocks(partial)}]}))
    final = parts[-1]
    final.content = [b for part in parts for b in part.content]
    return final


def log_advisor_usage(reply: Any, log_prefix: str = "") -> None:
    usage = getattr(reply, "usage", None)
    for iteration in getattr(usage, "iterations", None) or []:
        data = iteration if isinstance(iteration, dict) else getattr(iteration, "__dict__", {})
        if data.get("type") == ADVISOR_ITERATION:
            logger.info("%sAdvisor call: model=%s in=%d out=%d", log_prefix, data.get("model", "unknown"),
                        int(data.get("input_tokens") or 0), int(data.get("output_tokens") or 0))
