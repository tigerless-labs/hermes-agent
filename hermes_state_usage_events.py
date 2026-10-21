"""Time-stamped usage records (fork, ``sessions.usage_events``): ``session_model_usage`` only keeps
running totals, so each delta it absorbs is also appended here with the time it was written. History
from before the switch is carried over once, dated at each row's last activity, so the records of a
session always add up to its totals."""

from __future__ import annotations

from typing import Any, Sequence

CARRIED_OVER_KEY = "usage_events_carried_over"

EVENT_INSERT_SQL = """INSERT INTO session_usage_events (
                   session_id, at, model, billing_provider, billing_base_url, billing_mode, task,
                   api_call_count, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                   reasoning_tokens, estimated_cost_usd, actual_cost_usd, carried_over
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)"""

CARRY_OVER_SQL = """INSERT INTO session_usage_events (
                   session_id, at, model, billing_provider, billing_base_url, billing_mode, task,
                   api_call_count, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                   reasoning_tokens, estimated_cost_usd, actual_cost_usd, carried_over
               ) SELECT session_id, COALESCE(last_seen, first_seen, 0), model, billing_provider, billing_base_url,
                   billing_mode, task, api_call_count, input_tokens, output_tokens, cache_read_tokens,
                   cache_write_tokens, reasoning_tokens, estimated_cost_usd, actual_cost_usd, 1
               FROM session_model_usage"""


def usage_events_enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly

        sessions = load_config_readonly().get("sessions")
        return isinstance(sessions, dict) and sessions.get("usage_events") is True
    except Exception:
        return False


def carry_over_once(conn: Any) -> None:
    if conn.execute("SELECT 1 FROM state_meta WHERE key = ?", (CARRIED_OVER_KEY,)).fetchone() is not None:
        return
    conn.execute(CARRY_OVER_SQL)
    conn.execute("INSERT INTO state_meta (key, value) VALUES (?, '1')", (CARRIED_OVER_KEY,))


def record_event(conn: Any, session_id: str, at: float, route: Sequence[Any], counts: Sequence[Any],
                 estimated_cost_usd: float, actual_cost_usd: float) -> None:
    conn.execute(EVENT_INSERT_SQL, (session_id, at, *route, *counts, estimated_cost_usd, actual_cost_usd))
