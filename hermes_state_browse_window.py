"""Browsing by start time (fork, ``sessions.browse_by_date``): session_search's browse lists only the most
recent sessions, so asking what happened last week meant inventing a query and paging through matches. Here
the range is filtered and ordered in SQL by when each conversation started, so a page boundary is a start
time and the next page begins exactly where the last one stopped."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from hermes_state_common import _LISTABLE_CHILD_SQL, _shape_preview, _sql_preview_raw, _sql_session_last_active_by_id
from hermes_state_sessions import _delegate_from_json

UPSTREAM_BROWSE_MAX_RESULTS = 10


@dataclass(frozen=True)
class BrowseSettings:
    by_date: bool
    max_results: int


def browse_settings() -> BrowseSettings:
    try:
        from hermes_cli.config import load_config_readonly

        sessions = load_config_readonly().get("sessions")
    except Exception:
        sessions = None
    sessions = sessions if isinstance(sessions, dict) else {}
    max_results = sessions.get("browse_max_results")
    return BrowseSettings(
        by_date=sessions.get("browse_by_date") is True,
        max_results=max_results if isinstance(max_results, int) and not isinstance(max_results, bool)
        and max_results >= 1 else UPSTREAM_BROWSE_MAX_RESULTS)


def sessions_started_between(db: Any, *, after_ts: Optional[float], before_ts: Optional[float], limit: int,
                             exclude_sources: Sequence[str]) -> list[dict[str, Any]]:
    clauses = ["s.archived = 0", "s.hidden = 0", f"{_delegate_from_json('s.model_config')} IS NULL",
               _LISTABLE_CHILD_SQL]
    params: list[Any] = []
    if exclude_sources:
        clauses.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
        params.extend(exclude_sources)
    if after_ts is not None:
        clauses.append("s.started_at >= ?")
        params.append(after_ts)
    if before_ts is not None:
        clauses.append("s.started_at < ?")
        params.append(before_ts)
    query = f"""
        SELECT s.id, s.source, s.title, s.started_at, s.message_count,
               {_sql_session_last_active_by_id('s.id')} AS last_active,
               {_sql_preview_raw('s.id')}
        FROM sessions s
        WHERE {' AND '.join(clauses)}
        ORDER BY s.started_at DESC, s.id DESC
        LIMIT ?"""
    with db._read_ctx() as conn:
        rows = conn.execute(query, [*params, max(1, int(limit))]).fetchall()
    sessions = []
    for row in rows:
        session = dict(row)
        session["preview"] = _shape_preview(session.pop("_preview_raw", ""))
        sessions.append(session)
    return sessions
