"""The gateway side of ``PluginContext.steer_session``: a plugin's words reach a running turn only when
that turn acts for the origin's own sender (same platform, scope and user id), whatever the session's
busy settings; otherwise nothing changes and the plugin may start a turn instead."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource

SESSION = "agent:main:slack:dm:T1:D1:background-1"
OWNER = SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm", user_id="U_ALICE",
                      thread_id="background-1", scope_id="T1")
WORDS = "<@U_ALICE> added: only the PDFs"


class _Agent:
    def __init__(self, takes=True):
        self.payloads: list[str] = []
        self.takes = takes
        self._active_children = []
        self._active_children_lock = threading.Lock()

    def steer(self, text):
        self.payloads.append(text)
        return self.takes


def _source(**changes) -> SessionSource:
    return SessionSource(**{**OWNER.__dict__, **changes})


def _runner(tmp_path, monkeypatch, *, agent=None, turn_source=OWNER, holder=None):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner = GatewayRunner(config=GatewayConfig())
    agent = agent if agent is not None else _Agent()
    turn = runner._session_state(SESSION).turn
    turn.agent = agent
    turn.ctx = SimpleNamespace(source=turn_source, agent_holder=[agent] if holder is None else holder)
    return runner, agent


async def _steer(runner, source=OWNER, content=WORDS) -> bool:
    return await runner._dispatch_plugin_session_steer(session_key=SESSION, source=source, content=content,
                                                       plugin_id="lara")


@pytest.mark.asyncio
async def test_the_turn_s_own_sender_steers_it_with_the_plugins_words(tmp_path, monkeypatch):
    runner, agent = _runner(tmp_path, monkeypatch)
    assert await _steer(runner) is True
    assert agent.payloads == [WORDS]


STRANGERS = {
    "another person": _source(user_id="U_BOB"),
    "the owner's id in another workspace": _source(scope_id="T2"),
    "a bot": _source(is_bot=True),
    "no user id": _source(user_id=None),
    "another platform": _source(platform=Platform.DISCORD),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("stranger", STRANGERS.values(), ids=STRANGERS.keys())
async def test_redteam_a_turn_acting_for_someone_else_never_takes_the_words(tmp_path, monkeypatch, stranger):
    runner, agent = _runner(tmp_path, monkeypatch)
    assert await _steer(runner, source=stranger) is False
    assert agent.payloads == []


@pytest.mark.asyncio
async def test_redteam_a_turn_with_no_recorded_sender_takes_nothing(tmp_path, monkeypatch):
    runner, agent = _runner(tmp_path, monkeypatch, turn_source=None)
    assert await _steer(runner) is False and agent.payloads == []


@pytest.mark.asyncio
async def test_redteam_a_turn_held_by_another_agent_takes_nothing(tmp_path, monkeypatch):
    runner, agent = _runner(tmp_path, monkeypatch, holder=[_Agent()])
    assert await _steer(runner) is False and agent.payloads == []


@pytest.mark.asyncio
async def test_nothing_running_is_false(tmp_path, monkeypatch):
    runner, _ = _runner(tmp_path, monkeypatch)
    runner._session_state(SESSION).turn.agent = None
    assert await _steer(runner) is False


@pytest.mark.asyncio
async def test_a_turn_still_starting_is_false(tmp_path, monkeypatch):
    from gateway.run import _AGENT_PENDING_SENTINEL
    runner, _ = _runner(tmp_path, monkeypatch)
    runner._session_state(SESSION).turn.agent = _AGENT_PENDING_SENTINEL
    assert await _steer(runner) is False


@pytest.mark.asyncio
async def test_an_agent_that_does_not_take_it_is_false(tmp_path, monkeypatch):
    runner, agent = _runner(tmp_path, monkeypatch, agent=_Agent(takes=False))
    assert await _steer(runner) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   "])
async def test_empty_words_reach_nothing(tmp_path, monkeypatch, content):
    runner, agent = _runner(tmp_path, monkeypatch)
    assert await _steer(runner, content=content) is False and agent.payloads == []


def test_from_another_thread_it_waits_for_the_answer_on_the_live_loop(tmp_path, monkeypatch):
    runner, agent = _runner(tmp_path, monkeypatch)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        runner._gateway_loop, runner._running = loop, True
        monkeypatch.setattr(runner, "_plugin_origin_source", lambda origin, **_: OWNER)
        monkeypatch.setattr(runner, "_session_key_for_source", lambda source: SESSION)
        assert runner._steer_plugin_session(origin={}, content=WORDS, plugin_home=None, plugin_id="lara") is True
        assert agent.payloads == [WORDS]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def test_a_gateway_not_running_or_an_origin_it_refuses_is_false(tmp_path, monkeypatch):
    runner, agent = _runner(tmp_path, monkeypatch)
    assert runner._steer_plugin_session(origin={}, content=WORDS, plugin_home=None, plugin_id="lara") is False
    loop = asyncio.new_event_loop()
    try:
        runner._gateway_loop, runner._running = loop, True
        monkeypatch.setattr(runner, "_plugin_origin_source", lambda origin, **_: None)
        assert runner._steer_plugin_session(origin={}, content=WORDS, plugin_home=None, plugin_id="lara") is False
    finally:
        loop.close()
    assert agent.payloads == []


@pytest.mark.asyncio
async def test_called_on_the_gateway_loop_itself_it_declines_rather_than_block(tmp_path, monkeypatch):
    runner, agent = _runner(tmp_path, monkeypatch)
    runner._gateway_loop, runner._running = asyncio.get_running_loop(), True
    monkeypatch.setattr(runner, "_plugin_origin_source", lambda origin, **_: OWNER)
    monkeypatch.setattr(runner, "_session_key_for_source", lambda source: SESSION)
    assert runner._steer_plugin_session(origin={}, content=WORDS, plugin_home=None, plugin_id="lara") is False
    assert agent.payloads == []
