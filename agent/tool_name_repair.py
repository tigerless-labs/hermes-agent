"""``agent.tool_name_fuzzy_repair`` (fork, Lara): whether an unknown tool name may be renamed to the nearest one.

With a large on-demand catalogue whose write tools carry API-style names, the fuzzy step turned guessed read calls
into write calls (``drive_files_list`` ran as ``drive_files_delete``, ``messages_modify`` as ``messages_send``).
Switched off, only spellings of the same name repair — the deterministic steps upstream already takes, plus
``<platform>.<tool>`` for ``<platform>__<tool>`` — and anything else stays unknown. Unset keeps upstream.
"""

from __future__ import annotations

from typing import Iterable

NAMESPACE_DOT, NAMESPACE_SEPARATOR = ".", "__"


def fuzzy_repair_allowed() -> bool:
    try:
        from hermes_cli.config import load_config_readonly
        section = load_config_readonly().get("agent")
    except Exception:
        return True
    return not (isinstance(section, dict) and section.get("tool_name_fuzzy_repair") is False)


def namespaced_spellings(candidates: Iterable[str]) -> set:
    return {c.replace(NAMESPACE_DOT, NAMESPACE_SEPARATOR, 1) for c in candidates if c and NAMESPACE_DOT in c}
