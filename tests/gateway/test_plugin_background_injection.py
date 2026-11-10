"""``ctx.inject_message(origin=..., background=True)``: the turn runs and is recorded like any
other, and nothing of it reaches the platform (no reply, no typing, no error notice).

Same harness as the origin-injection tests: real ``GatewayRunner`` and ``BasePlatformAdapter``
ingress with a recording transport; only the model call is replaced.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

from gateway.background_turn import BACKGROUND_TURN, background_event, background_metadata, background_turn
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.warning_notifications import diagnostic_wake_muted
from tests.gateway.test_plugin_origin_injection import (  # noqa: F401  (gateway is a fixture)
    PLUGIN, _context, _origin, _publish, _settle, gateway,
)


def _grant_background(home) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"entries": {PLUGIN: {"allow_gateway_injection": True, "allow_background_injection": True}}},
    }), encoding="utf-8")


def _record_typing(adapter) -> list:
    typed = []

    async def _typing(chat_id, metadata=None):
        typed.append(chat_id)

    adapter.send_typing = _typing
    return typed


def _record_turns(runner, *, fail: bool = False, hold: asyncio.Event | None = None) -> list:
    turns = []

    async def _turn(message, context_prompt, history, source, session_id, **kwargs):
        turns.append(SimpleNamespace(message=message, session_id=session_id,
                                     display_metadata=kwargs.get("persist_user_display_metadata") or {}))
        if hold is not None:
            await hold.wait()
        if fail:
            raise RuntimeError("provider detail")
        return {"final_response": f"reply-{len(turns)}", "messages": [], "tools": [],
                "history_offset": len(history), "last_prompt_tokens": 0}

    runner._run_agent_inner = _turn
    return turns


@pytest.mark.asyncio
async def test_background_turn_runs_and_sends_nothing(gateway):
    runner, alpha = gateway.runner, gateway.adapters["alpha"]
    _grant_background(gateway.homes["alpha"])
    typed, turns = _record_typing(alpha), _record_turns(runner)
    ctx, manager = _context(gateway.homes["alpha"])
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[manager]):
        await _publish(runner)
        assert ctx.inject_message("Check the inbox", origin=_origin("100"), background=True) is True
        await _settle(runner, alpha)
        assert ctx.inject_message("Say hello", origin=_origin("101")) is True
        await _settle(runner, alpha)

    background, plain = turns
    assert background.message.endswith("Check the inbox")
    assert background.display_metadata.get(BACKGROUND_TURN) is True
    assert BACKGROUND_TURN not in plain.display_metadata
    assert alpha.sent == [("101", "reply-2")]
    assert "100" not in typed
    assert "agent:alpha:telegram:group:100:7" in runner.session_store._entries


@pytest.mark.asyncio
async def test_background_needs_its_own_grant(gateway):
    """The injection grant alone does not allow a silent turn: refused synchronously, nothing runs."""
    runner, alpha = gateway.runner, gateway.adapters["alpha"]
    turns = _record_turns(runner)
    ctx, manager = _context(gateway.homes["alpha"])
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[manager]):
        await _publish(runner)
        assert ctx.inject_message("x", origin=_origin("100"), background=True) is False
        await _settle(runner, alpha)
    assert turns == [] and alpha.sent == [] and runner.session_store._entries == {}


@pytest.mark.asyncio
async def test_background_only_applies_to_an_origin(gateway):
    """A session-key injection cannot be marked background: the flag belongs to a place the plugin names."""
    _grant_background(gateway.homes["alpha"])
    ctx, manager = _context(gateway.homes["alpha"])
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[manager]):
        await _publish(gateway.runner)
        assert ctx.inject_message("x", session_key="agent:alpha:telegram:group:100:7", background=True) is False


@pytest.mark.asyncio
async def test_failed_background_turn_sends_nothing(gateway):
    runner, alpha = gateway.runner, gateway.adapters["alpha"]
    _grant_background(gateway.homes["alpha"])
    turns = _record_turns(runner, fail=True)
    ctx, manager = _context(gateway.homes["alpha"])
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[manager]):
        await _publish(runner)
        assert ctx.inject_message("x", origin=_origin("100"), background=True) is True
        await _settle(runner, alpha)
    assert len(turns) == 1 and alpha.sent == []


@pytest.mark.asyncio
async def test_background_injection_queued_behind_a_running_one_stays_silent(gateway):
    runner, alpha = gateway.runner, gateway.adapters["alpha"]
    _grant_background(gateway.homes["alpha"])
    hold = asyncio.Event()
    turns = _record_turns(runner, hold=hold)
    ctx, manager = _context(gateway.homes["alpha"])
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[manager]):
        await _publish(runner)
        assert ctx.inject_message("first", origin=_origin("100"), background=True) is True
        for _ in range(200):
            await asyncio.sleep(0.01)
            if turns:
                break
        assert ctx.inject_message("second", origin=_origin("100"), background=True) is True
        await asyncio.sleep(0.05)
        hold.set()
        await _settle(runner, alpha)
    assert [turn.message.split()[-1] for turn in turns] == ["first", "second"]
    assert turns[0].session_id == turns[1].session_id
    assert alpha.sent == []


@pytest.mark.asyncio
async def test_a_person_in_a_background_session_is_answered(gateway):
    """Red team: only a trusted injected event is background; a human message carrying the same
    metadata, in the very session a background turn made, gets its reply."""
    runner, alpha = gateway.runner, gateway.adapters["alpha"]
    _grant_background(gateway.homes["alpha"])
    turns = _record_turns(runner)
    ctx, manager = _context(gateway.homes["alpha"])
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[manager]):
        await _publish(runner)
        assert ctx.inject_message("x", origin=_origin("100"), background=True) is True
        await _settle(runner, alpha)
        import gateway.run as gateway_run
        source = alpha.build_source(chat_id="100", chat_type="group", thread_id="7", user_id="u1")
        async with gateway_run._async_profile_runtime_scope(gateway.homes["alpha"]):
            await alpha.handle_message(MessageEvent(text="what happened?", message_type=MessageType.TEXT,
                                                    source=source, metadata={BACKGROUND_TURN: True}))
        await _settle(runner, alpha)
    assert turns[0].session_id == turns[1].session_id
    assert BACKGROUND_TURN not in turns[1].display_metadata
    assert alpha.sent == [("100", "reply-2")]


def _event(*, internal: bool, metadata: dict) -> MessageEvent:
    return MessageEvent(text="x", message_type=MessageType.TEXT, internal=internal, metadata=metadata,
                        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat"))


def test_only_a_trusted_injected_event_is_background():
    marked = _event(internal=True, metadata={BACKGROUND_TURN: True})
    assert background_event(marked) and background_metadata(marked) == {BACKGROUND_TURN: True}
    assert diagnostic_wake_muted(marked) is True
    for event in (_event(internal=False, metadata={BACKGROUND_TURN: True}),
                  _event(internal=True, metadata={BACKGROUND_TURN: "yes"}),
                  _event(internal=True, metadata={})):
        assert not background_event(event) and background_metadata(event) == {}
        assert diagnostic_wake_muted(event) is False
    assert background_turn({BACKGROUND_TURN: True}) is True
    assert not any(background_turn(value) for value in (None, {}, {BACKGROUND_TURN: 1}, {"other": True}))


def test_background_turn_rows_stay_ordinary():
    """A diagnostic wake hides its rows from transcript readers; a background turn is real work the
    person may look up later, so its rows are recorded like any conversation's."""
    from unittest.mock import patch as _patch

    from hermes_state import SessionDB
    with _patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                            quiet_mode=True, session_db=db, session_id="sess-bg", skip_context_files=True,
                            skip_memory=True)
            agent._mute_notification_reply = True
            agent._background_turn = True
            tail = [{"role": "user", "content": "run"}, {"role": "assistant", "content": "done"}]
            assert agent._flush_messages_to_session_db(tail, []) is True
            assert [(row["content"], row.get("display_kind")) for row in db.get_messages("sess-bg")] == [
                ("run", None), ("done", None)]
            db.close()


def test_background_turn_puts_no_compaction_on_the_task_card():
    from tests.gateway.test_slack_task_card_compaction import ONE_COMPACTION, _adapter, _queued, _report, _turn

    adapter, _client = _adapter()
    shown, shown_events = _turn(adapter)
    _report(shown, ONE_COMPACTION)
    silent, silent_events = _turn(adapter)
    silent._ctx.persist_user_display_metadata = {BACKGROUND_TURN: True}
    silent._ctx.mute_notification_reply = True
    _report(silent, ONE_COMPACTION)
    assert _queued(shown_events) and _queued(silent_events) == []


@pytest.mark.asyncio
async def test_notices_before_the_agent_runs_stay_unsent_in_a_background_turn():
    """Session-reset and compression notices are sent with no event in reach: they read the mark the
    turn's own task carries, and a later ordinary turn in the same task clears it."""
    from gateway.background_turn import enter_turn, handling_background_turn
    from gateway.run import GatewayRunner

    warned = []

    class _Adapter:
        async def emit_warning(self, chat_id, message, **_kwargs):
            warned.append(message)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: _Adapter()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat")

    enter_turn(_event(internal=True, metadata={BACKGROUND_TURN: True}))
    assert handling_background_turn() is True
    await runner._hmwa_hygiene_notify(source, {}, "compressing", "notice")
    assert warned == []
    enter_turn(_event(internal=False, metadata={BACKGROUND_TURN: True}))
    assert handling_background_turn() is False
    await runner._hmwa_hygiene_notify(source, {}, "compressing", "notice")
    assert warned == ["compressing"]
