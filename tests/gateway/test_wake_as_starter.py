"""A background wake runs as the person who started the work (``gateway.wake_as_starter``).

Upstream builds a wake's whole source from the session's stored origin (or its last cached source), user
included. In a shared thread every wake then runs as the thread's opener, whoever dispatched the work, and a
gateway that maps the turn's user to credentials lends the opener's to it. With the switch on the wake keeps the
origin's place and takes its person from the work; work that names no starter wakes as no one.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

import gateway.run as gateway_run
import tools.process_registry as process_registry_module
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource

KEY = "agent:main:slack:group:T1:C1:1792778489.064329"
OPENER, OPENER_NAME = "U0OPENER", "opener"
STARTER, STARTER_NAME = "U0STARTER", "starter"


class _Admitting(AsyncMock):
    async def _execute_mock_call(self, event, *args, **kwargs):
        result = await super()._execute_mock_call(event, *args, **kwargs)
        event._gateway_accepted = True
        return result


def _runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: object) -> GatewayRunner:
    (tmp_path / "config.yaml").write_text(config if isinstance(config, str) else yaml.safe_dump(config),
                                          encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[Platform.SLACK] = SimpleNamespace(send=AsyncMock(), handle_message=_Admitting())
    return runner


@pytest.fixture
def switched_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GatewayRunner:
    return _runner(tmp_path, monkeypatch, {"gateway": {"wake_as_starter": True}})


@pytest.fixture
def switched_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GatewayRunner:
    return _runner(tmp_path, monkeypatch, {})


def _opener_source() -> SessionSource:
    return SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group", thread_id="1792778489.064329",
                         user_id=OPENER, user_name=OPENER_NAME, user_id_alt="alt-opener", scope_id="T1")


def _stored_origin(runner: GatewayRunner) -> SessionSource:
    origin = _opener_source()
    runner.session_store._entries[KEY] = SimpleNamespace(origin=origin)
    return origin


def _delegation(**starter: str) -> dict:
    return {"type": "async_delegation", "delegation_id": "deleg_1", "session_key": KEY, **starter}


def _process_completion(session_id: str = "proc_1") -> dict:
    return {"type": "completion", "session_id": session_id, "session_key": KEY, "command": "sleep 240; date"}


def _registry_knows(monkeypatch: pytest.MonkeyPatch, sessions: dict[str, SimpleNamespace]) -> None:
    monkeypatch.setattr(process_registry_module.process_registry, "get", lambda session_id: sessions.get(session_id))


def test_the_switch_is_off_by_default_and_known_to_the_config() -> None:
    from hermes_cli.config import _validate_config_key
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["gateway"]["wake_as_starter"] is False
    assert _validate_config_key("gateway.wake_as_starter")[0]


def test_switched_off_a_wake_keeps_upstreams_stored_origin_whoever_started_it(switched_off) -> None:
    _stored_origin(switched_off)
    source = switched_off._build_process_event_source(_delegation(user_id=STARTER, user_name=STARTER_NAME))
    assert (source.user_id, source.user_name) == (OPENER, OPENER_NAME)


def test_a_delegation_wake_in_someone_elses_thread_runs_as_its_dispatcher_in_that_thread(switched_on) -> None:
    origin = _stored_origin(switched_on)
    source = switched_on._build_process_event_source(_delegation(user_id=STARTER, user_name=STARTER_NAME))
    assert (source.user_id, source.user_name, source.user_id_alt) == (STARTER, STARTER_NAME, None)
    assert (source.platform, source.chat_id, source.chat_type, source.thread_id, source.scope_id) == (
        origin.platform, origin.chat_id, origin.chat_type, origin.thread_id, origin.scope_id)


def test_redteam_the_stored_origin_is_never_rewritten_by_a_wake(switched_on) -> None:
    origin = _stored_origin(switched_on)
    switched_on._build_process_event_source(_delegation(user_id=STARTER, user_name=STARTER_NAME))
    assert switched_on.session_store._entries[KEY].origin is origin
    assert (origin.user_id, origin.user_name, origin.user_id_alt) == (OPENER, OPENER_NAME, "alt-opener")


def test_redteam_the_last_turns_cached_source_does_not_lend_its_person_either(switched_on) -> None:
    switched_on._session_sources = OrderedDict({KEY: _opener_source()})
    source = switched_on._build_process_event_source(_delegation(user_id=STARTER, user_name=STARTER_NAME))
    assert source.user_id == STARTER and source.chat_id == "C1"
    assert switched_on._session_sources[KEY].user_id == OPENER


def test_a_background_command_wakes_as_the_person_whose_turn_started_it(switched_on, monkeypatch) -> None:
    _stored_origin(switched_on)
    _registry_knows(monkeypatch, {"proc_1": SimpleNamespace(watcher_user_id=STARTER, watcher_user_name=STARTER_NAME)})
    source = switched_on._build_process_event_source(_process_completion())
    assert (source.user_id, source.user_name) == (STARTER, STARTER_NAME)


@pytest.mark.parametrize("event", [
    _delegation(),
    _delegation(user_id="  ", user_name=STARTER_NAME),
    {"session_key": KEY},
], ids=["delegation-without-a-person", "blank-user-id", "bare-event"])
def test_redteam_work_that_names_no_starter_wakes_as_no_one_not_the_opener(switched_on, monkeypatch, event) -> None:
    _stored_origin(switched_on)
    _registry_knows(monkeypatch, {})
    source = switched_on._build_process_event_source(event)
    assert (source.user_id, source.user_name, source.user_id_alt) == (None, None, None)
    assert source.chat_id == "C1" and source.thread_id == "1792778489.064329"


@pytest.mark.parametrize("known", [
    {},
    {"proc_1": SimpleNamespace(watcher_user_id="", watcher_user_name=OPENER_NAME)},
], ids=["process-gone", "process-without-a-person"])
def test_redteam_a_background_command_with_no_recorded_starter_wakes_as_no_one(switched_on, monkeypatch, known) -> None:
    _stored_origin(switched_on)
    _registry_knows(monkeypatch, known)
    source = switched_on._build_process_event_source(_process_completion())
    assert (source.user_id, source.user_name) == (None, None)


def test_redteam_a_failing_registry_lookup_wakes_as_no_one(switched_on, monkeypatch) -> None:
    _stored_origin(switched_on)

    def broken(session_id):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(process_registry_module.process_registry, "get", broken)
    assert switched_on._build_process_event_source(_process_completion()).user_id is None


def test_redteam_a_delegation_never_borrows_a_process_that_shares_its_id(switched_on, monkeypatch) -> None:
    _stored_origin(switched_on)
    _registry_knows(monkeypatch, {"deleg_1": SimpleNamespace(watcher_user_id=OPENER, watcher_user_name=OPENER_NAME)})
    event = {**_delegation(), "session_id": "deleg_1"}
    assert switched_on._build_process_event_source(event).user_id is None


def test_with_no_stored_or_cached_source_the_event_still_names_the_person(switched_on) -> None:
    event = {"session_key": "", "platform": "slack", "chat_type": "group", "chat_id": "C1",
             "user_id": STARTER, "user_name": STARTER_NAME}
    source = switched_on._build_process_event_source(event)
    assert (source.chat_id, source.user_id) == ("C1", STARTER)


def test_redteam_an_unreadable_config_keeps_upstreams_behaviour(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch, "gateway: [wake_as_starter\n")
    _stored_origin(runner)
    assert runner._build_process_event_source(_delegation(user_id=STARTER)).user_id == OPENER


def test_a_switch_that_is_not_a_mapping_keeps_upstreams_behaviour(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch, {"gateway": "wake_as_starter"})
    _stored_origin(runner)
    assert runner._build_process_event_source(_delegation(user_id=STARTER)).user_id == OPENER


@pytest.mark.asyncio
async def test_the_injected_wake_reaches_the_adapter_as_the_starter_in_the_openers_thread(switched_on) -> None:
    _stored_origin(switched_on)
    await switched_on._inject_watch_notification("[done]", _delegation(user_id=STARTER, user_name=STARTER_NAME))
    adapter = switched_on.adapters[Platform.SLACK]
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert (event.source.user_id, event.source.chat_id, event.source.thread_id) == (
        STARTER, "C1", "1792778489.064329")
