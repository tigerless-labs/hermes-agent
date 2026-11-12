"""A held turn (``PluginContext.hold_turn``) ends without words even when it must be answered.

The agent result of a held turn carries the hold; with it, the silence marker of a turn addressed
to the bot is not replaced by the unexpected-silence notice. A marker the model wrote itself on
such a turn still gets the notice.
"""
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from agent import turn_hold
from gateway.config import GatewayConfig
from gateway.response_filters import is_intentional_silence_response
from tests.gateway.test_busy_steer_reply_expected import SYSTEM_KEY, SYSTEM_SOURCE, _system_runner
from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source

SESSION_KEY = "agent:main:telegram:group:-1001:12345"


def _result(**extra):
    return {"final_response": turn_hold.HELD_REPLY,
            "messages": [{"role": "user", "content": "@bot send the report"},
                         {"role": "assistant", "content": turn_hold.HELD_REPLY}],
            "tools": [], "history_offset": 0, "last_prompt_tokens": 0, "api_calls": 1, "failed": False, **extra}


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_expected", [None, True], ids=["adapter-unknown", "addressed"])
async def test_a_held_turn_that_must_be_answered_sends_nothing(monkeypatch, tmp_path, reply_expected):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_result(**{turn_hold.RESULT_KEY: True}))
    response = await runner._handle_message_with_agent(
        _event(reply_expected=reply_expected), _source(), SESSION_KEY, 1)
    assert response == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [None, False, "true", 1], ids=["absent", "false", "a-string", "a-number"])
async def test_redteam_a_marker_on_a_turn_that_is_not_held_still_gets_the_notice(monkeypatch, tmp_path, held):
    runner = _runner(monkeypatch, tmp_path)
    extra = {} if held is None else {turn_hold.RESULT_KEY: held}
    runner._run_agent = AsyncMock(return_value=_result(**extra))
    response = await runner._handle_message_with_agent(_event(reply_expected=True), _source(), SESSION_KEY, 1)
    assert response and not is_intentional_silence_response(response)


@pytest.mark.asyncio
async def test_redteam_a_failed_turn_is_never_a_silent_held_one(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_result(failed=True, **{turn_hold.RESULT_KEY: True}))
    response = await runner._handle_message_with_agent(_event(reply_expected=True), _source(), SESSION_KEY, 1)
    assert response


def _queued_turn():
    return SimpleNamespace(
        session_key=SESSION_KEY, stream_consumer_holder=[None], mute_notification_reply=False,
        persist_user_display_kind=None, reply_expected=True, source=_source(), _status_thread_metadata=None,
        event_message_id=None, inbound_message_id="msg-42", run_generation=1)


@pytest.mark.asyncio
async def test_a_held_turn_with_a_message_queued_behind_it_sends_nothing_first():
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner._deliver_queued_first_response = AsyncMock()
    result = {"final_response": turn_hold.HELD_REPLY, "failed": False, turn_hold.RESULT_KEY: True}
    await runner._run_agent_deliver_first_response(_queued_turn(), None, result, result, None)
    sent = runner._deliver_queued_first_response.await_args
    assert sent is None or not sent.args[0]


@pytest.mark.asyncio
async def test_redteam_a_queued_marker_that_is_not_held_still_gets_the_notice():
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner._deliver_queued_first_response = AsyncMock()
    result = {"final_response": turn_hold.HELD_REPLY, "failed": False}
    await runner._run_agent_deliver_first_response(_queued_turn(), None, result, result, None)
    delivered = runner._deliver_queued_first_response.await_args.args[0]
    assert delivered and not is_intentional_silence_response(delivered)


class _HeldAgent:
    held = True

    def __init__(self, **kwargs):
        self.tools = []
        self._pending_steer = None
        self._active_children = []
        self._active_children_lock = threading.Lock()

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        hold = {turn_hold.RESULT_KEY: True} if type(self).held else {}
        return {"final_response": turn_hold.HELD_REPLY, "messages": [], "api_calls": 1, "completed": True, **hold}


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [True, False], ids=["held", "the-model-s-own-marker"])
async def test_the_hold_on_the_agents_result_reaches_the_gateways_own_result(monkeypatch, tmp_path, held):
    runner, adapter = _system_runner(monkeypatch, tmp_path, _HeldAgent)
    _HeldAgent.held = held
    result = await runner._run_agent(
        message="@bot send the report", context_prompt="", history=[], source=SYSTEM_SOURCE,
        session_id="sess-held", session_key=SYSTEM_KEY, reply_expected=True)
    assert turn_hold.held(result) is held
    assert not any(content and not is_intentional_silence_response(content) for content in adapter.sent)
