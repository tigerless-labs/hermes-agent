"""Browsing by start time (fork, ``sessions.browse_by_date``): with no query, ``after``/``before`` bound
when a session started, newest first, a full page names the ``before`` of the next one, and the browse
limit follows ``sessions.browse_max_results``. Every listed session still passes the visibility gate."""
import json
import time
from datetime import UTC, datetime

import pytest

import hermes_state_browse_window
from hermes_state import SessionDB
from hermes_state_browse_window import BrowseSettings
from tests.tools.test_session_visibility import Provider, Rule, _session, visibility
from tools import session_search_tool
from tools.session_search_tool import session_search

DAY = 86400
UPSTREAM = BrowseSettings(by_date=False, max_results=10)


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def browse(monkeypatch):
    def install(by_date=True, max_results=50):
        settings = BrowseSettings(by_date=by_date, max_results=max_results)
        monkeypatch.setattr(hermes_state_browse_window, "browse_settings", lambda: settings)
        return settings
    install(by_date=False, max_results=UPSTREAM.max_results)
    return install


def _iso(ts):
    return datetime.fromtimestamp(ts, UTC).isoformat()


def _listed(payload):
    return [r["session_id"] for r in json.loads(payload)["results"]]


def _week(db, now, count=7):
    for day in range(count):
        _session(db, f"day{day}", started=now - (day + 1) * DAY, lines=[("user", f"task of day {day}")])
    db._conn.commit()


def _pages(db, after, before, limit):
    seen, bound = [], before
    for _ in range(50):
        payload = json.loads(session_search(after=after, before=bound, limit=limit, db=db))
        seen += [r["session_id"] for r in payload["results"]]
        bound = payload.get("next_before")
        if bound is None:
            return seen
    raise AssertionError("paging never ended")


def test_switched_off_browse_ignores_the_range_as_upstream_does(db, browse, visibility):
    visibility()
    now = int(time.time())
    _week(db, now)
    browse(by_date=False)
    assert _listed(session_search(after=_iso(now - 3 * DAY), limit=10, db=db)) == _listed(session_search(limit=10, db=db))


def test_browse_lists_only_sessions_started_in_the_range_newest_first(db, browse, visibility):
    visibility()
    now = int(time.time())
    _week(db, now)
    browse()
    listed = _listed(session_search(after=_iso(now - 5 * DAY - 1), before=_iso(now - 2 * DAY + 1), limit=10, db=db))
    assert listed == ["day1", "day2", "day3", "day4"]


def test_a_relative_bound_works_for_browsing_too(db, browse, visibility):
    visibility()
    now = int(time.time())
    _week(db, now)
    browse()
    listed = set(_listed(session_search(after="3d", limit=10, db=db)))
    assert {"day0", "day1"} <= listed and not listed & {"day3", "day4", "day5", "day6"}


def test_pages_cover_the_range_once_each(db, browse, visibility):
    visibility()
    now = int(time.time())
    _week(db, now)
    browse()
    everything = _listed(session_search(after=_iso(now - 30 * DAY), limit=50, db=db))
    assert _pages(db, _iso(now - 30 * DAY), None, limit=2) == everything
    assert len(everything) == 7


def test_the_last_page_names_no_next_one(db, browse, visibility):
    visibility()
    now = int(time.time())
    _week(db, now, count=3)
    browse()
    assert "next_before" not in json.loads(session_search(after=_iso(now - 30 * DAY), limit=3, db=db))


def test_browse_takes_up_to_the_configured_maximum_and_discovery_keeps_its_own(db, browse, visibility):
    visibility()
    now = int(time.time())
    for n in range(30):
        _session(db, f"s{n}", started=now - n * 60 - 60, lines=[("user", f"weekly report {n}")])
    db._conn.commit()
    browse(max_results=25)
    assert len(_listed(session_search(after=_iso(now - DAY), limit=100, db=db))) == 25
    assert len(_listed(session_search(limit=100, db=db))) == 25
    assert len(_listed(session_search(query="weekly report", limit=100, db=db))) <= UPSTREAM.max_results


def test_redteam_hidden_sources_children_and_archived_sessions_are_not_listed(db, browse, visibility):
    visibility()
    now = int(time.time())
    _session(db, "root", started=now - 3 * DAY, end_reason="compression", lines=[("user", "kickoff")])
    _session(db, "continuation", started=now - 3 * DAY + 60, parent="root", lines=[("user", "more")])
    _session(db, "spawned", started=now - 2 * DAY, parent="root", lines=[("user", "sub task")])
    for source in session_search_tool._HIDDEN_SESSION_SOURCES:
        _session(db, f"from-{source}", started=now - DAY, source=source, lines=[("user", "integration run")])
    _session(db, "archived", started=now - DAY, lines=[("user", "old")])
    db._conn.execute("UPDATE sessions SET archived = 1 WHERE id = 'archived'")
    db._conn.commit()
    browse()
    assert _listed(session_search(after=_iso(now - 30 * DAY), limit=50, db=db)) == ["root"]


def test_redteam_only_sessions_the_gate_lets_through_are_listed_and_paging_passes_the_rest(db, browse, visibility):
    now = int(time.time())
    _week(db, now)
    visible = {"day1", "day4", "day6"}
    rule = visibility(Provider(Rule(visible=visible)))._judgement
    browse()
    assert _pages(db, _iso(now - 30 * DAY), None, limit=1) == ["day1", "day4", "day6"]
    assert {place.id for place in rule.places} >= {"day0", "day2", "day3", "day5"}


def test_redteam_a_judgement_that_fails_hides_the_session(db, browse, visibility):
    now = int(time.time())
    _week(db, now, count=3)
    visibility(Provider(Rule(visible={"day0", "day1", "day2"}, fail_on={"day1"})))
    browse()
    assert _listed(session_search(after=_iso(now - 30 * DAY), limit=10, db=db)) == ["day0", "day2"]


def test_redteam_required_visibility_without_a_provider_lists_nothing(db, browse, visibility):
    now = int(time.time())
    _week(db, now, count=3)
    visibility(require=True)
    browse()
    assert _listed(session_search(after=_iso(now - 30 * DAY), limit=10, db=db)) == []


def test_an_invalid_bound_is_refused(db, browse, visibility):
    visibility()
    browse()
    assert json.loads(session_search(after="last tuesday", db=db))["success"] is False


def test_the_description_says_browsing_takes_the_range_only_when_switched_on(browse):
    browse(by_date=False)
    assert session_search_tool._session_search_schema_overrides() == {}
    settings = browse(max_results=40)
    overrides = session_search_tool._session_search_schema_overrides()
    properties = overrides["parameters"]["properties"]
    assert all("brows" in properties[name]["description"].lower() for name in ("after", "before", "limit"))
    assert str(settings.max_results) in properties["limit"]["description"]
    assert overrides["description"] != session_search_tool.SESSION_SEARCH_SCHEMA["description"]
    assert "after/before" in overrides["description"]
    assert set(properties) == set(session_search_tool.SESSION_SEARCH_SCHEMA["parameters"]["properties"])
