"""Time-stamped usage records (fork): with ``sessions.usage_events`` on, every usage delta written to
``session_model_usage`` is also appended to ``session_usage_events`` with the time it was written, so
usage over any period can be summed; history written before the switch is carried over once."""
import pytest

import hermes_state_usage_events
from hermes_state import SessionDB

COUNTERS = ("api_call_count", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens")


def _db(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr(hermes_state_usage_events, "usage_events_enabled", lambda: enabled)
    return SessionDB(tmp_path / "state.db")


def _rows(db, sql, *params):
    db.flush_token_counts()
    with db._lock:
        return [dict(row) for row in db._conn.execute(sql, params).fetchall()]


def _totals(db, table, session_id):
    columns = ", ".join(f"SUM({name}) AS {name}" for name in (*COUNTERS, "estimated_cost_usd"))
    (row,) = _rows(db, f"SELECT {columns} FROM {table} WHERE session_id = ?", session_id)
    return {name: round(value or 0, 9) for name, value in row.items()}


def _events(db, session_id=None):
    where, params = ("WHERE session_id = ?", (session_id,)) if session_id else ("", ())
    return _rows(db, f"SELECT * FROM session_usage_events {where} ORDER BY id", *params)


def _spend(db, session_id, calls=1, model="main-model", **extra):
    for _ in range(calls):
        db.update_token_counts(session_id, input_tokens=100, output_tokens=10, cache_read_tokens=50,
                               estimated_cost_usd=0.01, model=model, billing_provider="meta", api_call_count=1,
                               **extra)


def test_switched_off_the_engine_records_usage_exactly_as_upstream(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch, enabled=False)
    db.create_session("s1", source="slack")
    _spend(db, "s1", calls=2)
    db.record_auxiliary_usage("s1", "advisor", model="advisor-model", input_tokens=5)
    assert _events(db) == []
    assert _totals(db, "session_model_usage", "s1")["api_call_count"] == 3


def test_every_delta_is_recorded_with_its_time_route_and_task(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch, enabled=True)
    db.create_session("s1", source="slack")
    _spend(db, "s1")
    db.record_auxiliary_usage("s1", "advisor", model="advisor-model", billing_provider="anthropic", input_tokens=7)
    main, advisor = _events(db, "s1")
    assert (main["model"], main["task"], advisor["model"], advisor["task"]) == (
        "main-model", "", "advisor-model", "advisor")
    assert 0 < main["at"] <= advisor["at"]
    assert not main["carried_over"] and not advisor["carried_over"]


def test_records_always_add_up_to_the_cumulative_usage_queued_merged_and_auxiliary_alike(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch, enabled=True)
    for session_id in ("s1", "s2"):
        db.create_session(session_id, source="slack")
    for _ in range(5):
        db.queue_token_counts("s1", input_tokens=3, output_tokens=2, model="main-model", billing_provider="meta",
                              api_call_count=1, estimated_cost_usd=0.002)
    db.queue_token_counts("s2", input_tokens=9, model="main-model", api_call_count=1)
    _spend(db, "s1", model="other-model")
    db.record_auxiliary_usage("s1", "compression", model="aux-model", input_tokens=40, api_call_count=2)
    for session_id in ("s1", "s2"):
        assert _totals(db, "session_usage_events", session_id) == _totals(db, "session_model_usage", session_id)


def test_absolute_updates_record_nothing_as_they_add_nothing_per_model(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch, enabled=True)
    db.create_session("s1", source="slack")
    db.update_token_counts("s1", input_tokens=500, model="main-model", api_call_count=4, absolute=True)
    assert _events(db, "s1") == []


def test_history_from_before_the_switch_is_carried_over_once_at_its_last_activity(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch, enabled=False)
    db.create_session("old", source="slack")
    _spend(db, "old", calls=3)
    db.record_auxiliary_usage("old", "title_generation", model="aux-model", input_tokens=11)
    last_seen = {row["task"]: row["last_seen"] for row in _rows(db, "SELECT task, last_seen FROM session_model_usage")}
    db.close()

    db = _db(tmp_path, monkeypatch, enabled=True)
    db.create_session("new", source="slack")
    _spend(db, "new")
    _spend(db, "old")
    _spend(db, "new")
    carried = [row for row in _events(db) if row["carried_over"]]
    assert {row["task"]: row["at"] for row in carried} == last_seen
    assert {row["session_id"] for row in carried} == {"old"}
    for session_id in ("old", "new"):
        assert _totals(db, "session_usage_events", session_id) == _totals(db, "session_model_usage", session_id)


def test_the_carry_over_happens_once_even_across_reopens(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch, enabled=False)
    db.create_session("old", source="slack")
    _spend(db, "old")
    db.close()
    for _ in range(2):
        db = _db(tmp_path, monkeypatch, enabled=True)
        _spend(db, "old")
        db.close()
    db = _db(tmp_path, monkeypatch, enabled=True)
    assert sum(row["carried_over"] for row in _events(db)) == 1
    assert _totals(db, "session_usage_events", "old") == _totals(db, "session_model_usage", "old")


def test_deleting_a_session_deletes_its_records(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch, enabled=True)
    for session_id in ("gone", "kept"):
        db.create_session(session_id, source="slack")
        _spend(db, session_id)
    db.flush_token_counts()
    assert db.delete_session("gone")
    assert {row["session_id"] for row in _events(db)} == {"kept"}


@pytest.mark.parametrize("value, expected", [(True, True), (False, False), ("true", False), (None, False)])
def test_only_a_literal_true_turns_the_records_on(monkeypatch, value, expected):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"sessions": {"usage_events": value}})
    assert hermes_state_usage_events.usage_events_enabled() is expected


def test_an_unreadable_config_keeps_the_records_off(monkeypatch):
    def broken():
        raise OSError("config unreadable")
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", broken)
    assert hermes_state_usage_events.usage_events_enabled() is False
