"""``platforms.slack.extra.expected_reply_cues``: the bot's in-progress cues follow ``reply_expected``.

With the switch on, the :eyes: reaction and the thread's loading status are shown for a turn that is
not allowed to stay unanswered (``reply_expected`` is not False) and for nothing else, so a message
the bot may ignore in a shared thread leaves no trace in Slack. On the Agent Sessions API Slack takes
only lifecycle values: ``processing`` when the thread starts working and ``active`` when it stops,
each sent once per change (a typing refresh sends nothing), and a thread that never started is not
touched when the turn ends. A pause (approval, clarify, final delivery) returns the thread to
``active`` until the refresh resumes. Off, upstream behaviour holds. Driven through the real
``_handle_slack_message`` and the base adapter's processing loop.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

import plugins.platforms.slack.adapter as _slack_mod
from gateway.session import build_session_key
from tests.gateway.test_slack_ignore_other_user_mentions import (  # noqa: F401 - fixtures
    BOT_USER_ID, CHANNEL_ID, _redirect_cache, adapter,
)
from tests.gateway.test_slack_thread_reply_judgement import (
    PEER_REPLY, THREAD, _admit, _arm, _history, _message,
)

SLOW_TURN_SECONDS = 0.05
LOOP_TURNS = 10


@pytest.fixture
def sessions_api(monkeypatch, adapter):
    monkeypatch.setattr(_slack_mod, "_AGENT_SESSIONS_SUPPORTED", True)
    adapter._app.client.agents_sessions_setStatus = AsyncMock()
    adapter._app.client.assistant_threads_setStatus = AsyncMock()
    return adapter._app.client.agents_sessions_setStatus


def _cues_on(adapter):
    adapter.config.extra["expected_reply_cues"] = True


def _statuses(set_status):
    return [call.kwargs["status"] for call in set_status.await_args_list]


async def _settle():
    for _ in range(LOOP_TURNS):
        await asyncio.sleep(0)


def _reactions(adapter):
    return [call.kwargs["name"] for call in adapter._app.client.reactions_add.await_args_list]


async def _process(adapter, event):
    async def _slow_silent_turn(_event):
        await asyncio.sleep(SLOW_TURN_SECONDS)
        return None

    adapter._message_handler = _slow_silent_turn
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(event, session_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("history, text, cued", [
    (_history(), "and the week before?", True),
    (_history(PEER_REPLY), "ok send it over", False),
    (_history(PEER_REPLY), f"<@{BOT_USER_ID}> summarize", True),
], ids=["only-sender", "peer-spoke", "bot-mentioned"])
async def test_a_turn_shows_eyes_and_loading_exactly_when_it_must_be_answered(
        adapter, sessions_api, history, text, cued):
    _arm(adapter, history=history)
    _cues_on(adapter)
    admitted = await _admit(adapter, _message(text, "1.1"))
    await _process(adapter, admitted)
    assert ("eyes" in _reactions(adapter)) is cued
    assert _statuses(sessions_api) == (["processing", "active"] if cued else [])


@pytest.mark.asyncio
async def test_a_one_to_one_dm_shows_both_cues(adapter, sessions_api):
    _arm(adapter)
    _cues_on(adapter)
    admitted = await _admit(adapter, _message("hi", "1.2", channel="D0001", channel_type="im", thread_ts=None))
    await _process(adapter, admitted)
    assert "eyes" in _reactions(adapter)
    assert _statuses(sessions_api) == ["processing", "active"]


@pytest.mark.asyncio
async def test_redteam_a_turn_that_may_stay_unanswered_makes_no_cue_call_to_slack(adapter, sessions_api):
    _arm(adapter, history=_history(PEER_REPLY))
    _cues_on(adapter)
    admitted = await _admit(adapter, _message("ok send it over", "2.1"))
    await _process(adapter, admitted)
    client = adapter._app.client
    assert client.reactions_add.await_count == 0
    assert sessions_api.await_count == 0
    assert client.assistant_threads_setStatus.await_count == 0


@pytest.mark.asyncio
async def test_off_keeps_upstream_reactions_and_status_text(adapter, sessions_api):
    _arm(adapter)
    follow_up = await _admit(adapter, _message("and the week before?", "3.1"))
    await _process(adapter, follow_up)
    assert "eyes" not in _reactions(adapter)
    assert "processing" not in _statuses(sessions_api)
    assert _statuses(sessions_api)[0] == "is thinking..."


@pytest.mark.asyncio
async def test_a_refresh_sends_nothing_and_a_stop_returns_the_thread_to_active_once(adapter, sessions_api):
    _cues_on(adapter)
    metadata = {"thread_id": THREAD}
    for _ in range(3):
        await adapter.send_typing(CHANNEL_ID, metadata=metadata)
    await adapter.stop_typing(CHANNEL_ID, metadata=metadata)
    await adapter.stop_typing(CHANNEL_ID, metadata=metadata)
    assert _statuses(sessions_api) == ["processing", "active"]
    assert sessions_api.await_args.kwargs["thread_ts"] == THREAD


@pytest.mark.asyncio
async def test_redteam_ending_a_turn_never_touches_a_thread_that_did_not_start(adapter, sessions_api):
    _cues_on(adapter)
    await adapter.stop_typing(CHANNEL_ID, metadata={"thread_id": THREAD})
    assert sessions_api.await_count == 0


@pytest.mark.asyncio
async def test_a_failed_status_is_sent_again_on_the_next_refresh(adapter, sessions_api):
    _cues_on(adapter)
    sessions_api.side_effect = [RuntimeError("ratelimited"), None]
    metadata = {"thread_id": THREAD}
    for _ in range(3):
        await adapter.send_typing(CHANNEL_ID, metadata=metadata)
    assert _statuses(sessions_api) == ["processing", "processing"]


@pytest.mark.asyncio
async def test_a_pause_returns_the_thread_to_active_until_the_refresh_resumes(adapter, sessions_api):
    _cues_on(adapter)
    metadata = {"thread_id": THREAD}
    await adapter.send_typing(CHANNEL_ID, metadata=metadata)
    adapter.pause_typing_for_chat(CHANNEL_ID)
    await _settle()
    adapter.resume_typing_for_chat(CHANNEL_ID)
    await adapter.send_typing(CHANNEL_ID, metadata=metadata)
    assert _statuses(sessions_api) == ["processing", "active", "processing"]


@pytest.mark.asyncio
async def test_a_pause_from_the_agent_thread_reaches_slack(adapter, sessions_api):
    _cues_on(adapter)
    await adapter.send_typing(CHANNEL_ID, metadata={"thread_id": THREAD})
    await asyncio.to_thread(adapter.pause_typing_for_chat, CHANNEL_ID)
    await _settle()
    assert _statuses(sessions_api) == ["processing", "active"]


@pytest.mark.asyncio
async def test_the_legacy_status_api_keeps_its_text(adapter, monkeypatch):
    monkeypatch.setattr(_slack_mod, "_AGENT_SESSIONS_SUPPORTED", False)
    adapter._app.client.assistant_threads_setStatus = AsyncMock()
    _cues_on(adapter)
    await adapter.send_typing(CHANNEL_ID, metadata={"thread_id": THREAD})
    assert adapter._app.client.assistant_threads_setStatus.await_args.kwargs["status"] == "is thinking..."
