"""What a compaction keeps (fork, Lara): ``compression.tail_budget_real_tokens`` / ``tail_tool_rounds`` /
``tail_demote_to_file``.

Upstream budgets the verbatim tail in chars/4, which on CJK-with-JSON payloads keeps about twice the budget, and
never demotes the newest six tool rounds. A real-token cap is converted through this session's own real-to-estimate
ratio, accumulated between consecutive provider counts of the same wire tool list (no extra calls); a compaction
or a newly loaded tool is not a sample. Older tail tool results move to the chat's spill file and leave a one-line
path, the way Claude Code keeps large files as paths after compacting. Unset keeps upstream.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

MIN_ESTIMATED_GROWTH = 2_000
RATIO_FLOOR, RATIO_CEILING = 1.0, 4.0
SAVED_PATH_PREFIX = "Full output saved to: "


def _section() -> dict:
    try:
        from hermes_cli.config import load_config_readonly
        section = load_config_readonly().get("compression")
    except Exception:
        return {}
    return section if isinstance(section, dict) else {}


def _positive_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def tail_tool_rounds(default: int) -> int:
    return _positive_int(_section().get("tail_tool_rounds")) or default


def _wire_tool_names(tools: Optional[list], api_messages: list) -> tuple:
    from agent.responses_tool_search import plain_tools

    names = []
    for tool in plain_tools(tools, api_messages) or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        names.append(function.get("name") if isinstance(function, dict) else None)
    return tuple(names)


def record_provider_count(agent: Any, counted: Any, api_messages: list, tools: Optional[list]) -> None:
    """Accumulate real over estimated growth between this count and the session's previous one."""
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None or _positive_int(counted) is None:
        return
    from agent.model_metadata import estimate_messages_tokens_rough

    estimated, tools_seen = estimate_messages_tokens_rough(api_messages), _wire_tool_names(tools, api_messages)
    previous = getattr(compressor, "_provider_count_last", None)
    if previous and previous[2] == tools_seen and counted > previous[0] and estimated > previous[1]:
        real_growth, estimated_growth = getattr(compressor, "_provider_count_growth", (0, 0))
        compressor._provider_count_growth = (real_growth + counted - previous[0], estimated_growth + estimated - previous[1])
    compressor._provider_count_last = (counted, estimated, tools_seen)


def _session_ratio(compressor: Any) -> Optional[float]:
    real_growth, estimated_growth = getattr(compressor, "_provider_count_growth", (0, 0))
    if estimated_growth < MIN_ESTIMATED_GROWTH:
        return None
    return min(RATIO_CEILING, max(RATIO_FLOOR, real_growth / estimated_growth))


def calibrated_tail_budget(compressor: Any, upstream_budget: int) -> int:
    cap = _positive_int(_section().get("tail_budget_real_tokens"))
    if cap is None:
        return upstream_budget
    ratio = _session_ratio(compressor)
    return max(1, min(upstream_budget, cap if ratio is None else int(cap / ratio)))


def _inline_tools() -> frozenset:
    from tools.budget_config import _configured_deployment_caps

    return _configured_deployment_caps()[2]


def _saved_path(content: str) -> Optional[str]:
    from tools.tool_result_storage import PERSISTED_OUTPUT_TAG

    if PERSISTED_OUTPUT_TAG not in content:
        return None
    return next((line.removeprefix(SAVED_PATH_PREFIX).strip() for line in content.splitlines()
                 if line.startswith(SAVED_PATH_PREFIX)), None)


def _spill(message: dict, content: str) -> Optional[str]:
    import tools.tool_result_storage as storage

    if storage._outside_any_chat():
        return None
    call_id = message.get("tool_call_id") or "compacted_" + hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:12]
    host_path = storage._write_to_spillover(storage._pageable_text(content), storage._safe_result_filename(call_id))
    return storage._path_for_the_coming_sandbox(host_path) if host_path else None


def demoted_tail_content(message: dict, upstream_stub: str) -> str:
    """The demoted form of an older tail tool result: a path to its full text in the chat's file."""
    content = message.get("content")
    tool = message.get("tool_name") or "tool"
    if _section().get("tail_demote_to_file") is not True or not isinstance(content, str):
        return upstream_stub
    if tool in _inline_tools():
        return content
    try:
        path = _saved_path(content) or _spill(message, content)
    except Exception as exc:
        logger.warning("Compaction could not save a tail tool result (%s); keeping the upstream hint", type(exc).__name__)
        path = None
    if not path:
        return upstream_stub
    return f"[{tool} output moved out of context at compaction — {len(content):,} chars saved to {path}; read the file for details]"


SKILL_TRUNCATED = "\n[skill truncated at compaction — read it again for the rest]"


def demote_before_measuring() -> bool:
    return _section().get("tail_demote_to_file") is True


def _skill_tools() -> frozenset:
    tools = _section().get("tail_skill_tools")
    return frozenset(tools) if isinstance(tools, list) and all(isinstance(name, str) for name in tools) else frozenset()


def _rough_cap(compressor: Any, key: str) -> Optional[int]:
    cap = _positive_int(_section().get(key))
    return None if cap is None else max(1, int(cap / (_session_ratio(compressor) or RATIO_FLOOR)))


def _calls_by_id(messages: list, start: int) -> dict:
    calls = {}
    for message in messages[start:]:
        for call in (message.get("tool_calls") or []) if message.get("role") == "assistant" else []:
            if isinstance(call, dict) and call.get("id"):
                calls[call["id"]] = call.get("function") or {}
    return calls


def _skill_key(tool: str, function: dict, content: str) -> tuple:
    arguments = function.get("arguments")
    try:
        label = json.dumps(json.loads(arguments), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        label = arguments if isinstance(arguments, str) and arguments else hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:12]
    return tool, label[:200]


def _truncated_to(content: str, rough_budget: int) -> str:
    from agent.model_metadata import estimate_tokens_rough

    if estimate_tokens_rough(content) <= rough_budget:
        return content
    low, high = 0, len(content)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens_rough(content[:middle]) <= rough_budget:
            low = middle
        else:
            high = middle - 1
    return content[:low] + SKILL_TRUNCATED


def _compact_skills(compressor: Any, messages: list, start: int, protected_ids: set) -> list:
    from agent.context_compressor import _rewritten
    from agent.model_metadata import estimate_tokens_rough

    skill_tools, per, total = _skill_tools(), _rough_cap(compressor, "tail_skill_tokens"), _rough_cap(compressor, "tail_skills_total_tokens")
    if not skill_tools or per is None or total is None:
        return messages
    calls, seen, used, result = _calls_by_id(messages, start), set(), 0, list(messages)
    for i in range(len(messages) - 1, start - 1, -1):
        message = messages[i]
        tool, content = message.get("tool_name"), message.get("content")
        if message.get("role") != "tool" or tool not in skill_tools or not isinstance(content, str):
            continue
        if message.get("tool_call_id") in protected_ids or content.endswith(SKILL_TRUNCATED):
            continue
        key = _skill_key(tool, calls.get(message.get("tool_call_id"), {}), content)
        if key in seen:
            result[i] = _rewritten(message, f"[{tool} {key[1]} read again later in this conversation; the newer copy is kept]")
            continue
        seen.add(key)
        allowed = min(per, total - used)
        if allowed <= 0:
            result[i] = _rewritten(message, f"[{tool} {key[1]} dropped at compaction — read it again if needed]")
            continue
        kept = _truncated_to(content, allowed)
        used += estimate_tokens_rough(kept)
        if kept != content:
            result[i] = _rewritten(message, kept)
    return result


def _moved_arguments(call_id: str, arguments: str) -> Optional[str]:
    import tools.tool_result_storage as storage

    host_path = storage._write_to_spillover(arguments, storage._safe_result_filename(f"{call_id}.input"))
    if not host_path:
        return None
    return json.dumps({"moved_at_compaction": storage._path_for_the_coming_sandbox(host_path), "chars": len(arguments)},
                      ensure_ascii=False)


def _move_tool_inputs(messages: list, start: int, protected_ids: set, min_chars: int) -> list:
    import tools.tool_result_storage as storage

    if _section().get("tail_demote_tool_inputs") is not True or storage._outside_any_chat():
        return messages
    result = list(messages)
    for i in range(start, len(messages)):
        message = messages[i]
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not calls:
            continue
        moved_calls, changed = [], False
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if (not call.get("id") or call["id"] in protected_ids or not isinstance(arguments, str)
                    or len(arguments) < min_chars):
                moved_calls.append(call)
                continue
            try:
                moved = _moved_arguments(call["id"], arguments)
            except Exception as exc:
                logger.warning("Compaction could not save a tool input (%s); keeping it", type(exc).__name__)
                moved = None
            moved_calls.append({**call, "function": {**function, "arguments": moved}} if moved else call)
            changed = changed or bool(moved)
        if changed:
            result[i] = {**message, "tool_calls": moved_calls}
    return result


def demote_tail_inputs_and_skills(compressor: Any, messages: list, start: int, protected_ids: set, min_chars: int) -> list:
    """Her own long tool inputs move to the chat's file and skill bodies keep their newest copy within the caps;
    the newest kept rounds are untouched."""
    return _move_tool_inputs(_compact_skills(compressor, messages, start, protected_ids), start, protected_ids, min_chars)
