"""A steered message keeps its own reply verdict (``display.busy_steer_reply_expected``).

A steer hands the running agent text only. Upstream therefore loses whether that message needed
an answer: a steer the model reads leaves the turn judged by its opener alone, and a steer that
lands after the turn's last model call runs as the next turn with no verdict, which counts as
addressed. In a shared thread an unaddressed follow-up then may not stay silent, and the gateway
posts its fallback. With the switch on, every accepted steer is recorded with its message on the
turn it entered; when that turn ends, the steers the model read fold into its verdict and the
leftover ones give the next turn theirs, combined as one turn answering several messages is.
"""

from __future__ import annotations

import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.response_filters import is_intentional_silence_response
from gateway.session import SessionEntry, SessionSource
from gateway.turn_context import TurnContext

SESSION = "agent:main:slack:group:T1:C1:100.1"
OPENER = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group", user_id="U_ALICE",
                       user_name="Alice", thread_id="100.1", scope_id="T1")
BOB = SessionSource(**{**OPENER.__dict__, "user_id": "U_BOB", "user_name": "Bob"})


class _Agent:
    def __init__(self, accepts=True):
        self.accepts = accepts
        self.payloads: list[str] = []
        self._active_children = []
        self._active_children_lock = threading.Lock()

    def steer(self, text):
        if self.accepts:
            self.payloads.append(text)
        return self.accepts


class _Adapter:
    def __init__(self):
        self._pending_messages = {}

    async def send(self, *args, **kwargs):
        return SimpleNamespace(success=True, message_id="ack")


def _event(text="and the second one", source=OPENER, *, reply_expected=False, message_id="200.2") -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id=message_id,
                        reply_expected=reply_expected)


def _runner(tmp_path, monkeypatch, *, agent=None, **display):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"display": {"busy_input_mode": "steer", **display}}), encoding="utf-8")
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = gateway_run.GatewayRunner(config=GatewayConfig())
    agent, adapter = agent or _Agent(), _Adapter()
    turn = runner._session_state(SESSION).turn
    turn.agent = agent
    turn.event = _event("opening message", reply_expected=True, message_id="100.9")
    turn.ctx = TurnContext(source=OPENER, session_key=SESSION, reply_expected=True)
    turn.ctx.agent_holder[0] = agent
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: adapter)
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    monkeypatch.setattr(runner, "_admit_bot_message_for_source", lambda source: True)
    return runner, agent, adapter


async def _steer(runner, agent, event, route="busy"):
    if route == "busy":
        assert await runner._handle_active_session_busy_message(event, SESSION) is True
    elif route == "priority":
        runner._hm_busy_steer(event, agent, SESSION)
    else:
        command = MessageEvent(text=f"/steer {event.text}", message_type=MessageType.TEXT, source=event.source,
                               message_id=event.message_id, reply_expected=event.reply_expected)
        await runner._busy_steer_command(command, SESSION, event.source)


def _turn(runner):
    return runner._session_state(SESSION).turn


def _settle(runner, *, leftover: str | None, response=None):
    turn_ctx = _turn(runner).ctx
    result = {"final_response": "NO_REPLY", "messages": []}
    if leftover is not None:
        result["pending_steer"] = leftover
    response = {"final_response": "NO_REPLY"} if response is None else response
    runner._settle_turn_steers(turn_ctx, result, response)
    return turn_ctx, result, response


def _wire_followup(runner, monkeypatch, followup_result=None):
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value=followup_result or {"final_response": "NO_REPLY", "messages": []})
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._deliver_queued_first_response = AsyncMock()
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: None)


async def _followup(runner, turn_ctx, result, *, pending, pending_event=None):
    turn_ctx.session_id, turn_ctx.history, turn_ctx.run_generation = "sid", [], 1
    return await runner._run_agent_queued_followup(
        turn_ctx, None, pending, pending_event, {"final_response": "NO_REPLY"}, result, None)


# --- recording --------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["busy", "priority", "explicit_command"])
async def test_switch_off_records_nothing_and_a_leftover_steer_stays_unknown(tmp_path, monkeypatch, route):
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True)
    await _steer(runner, agent, _event(), route)
    assert agent.payloads and _turn(runner).steers == []
    turn_ctx, result, _ = _settle(runner, leftover=agent.payloads[0])
    _wire_followup(runner, monkeypatch)
    await _followup(runner, turn_ctx, result, pending=agent.payloads[0])
    assert runner._run_agent.await_args.kwargs["reply_expected"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["busy", "priority", "explicit_command"])
async def test_every_steering_route_records_the_steer_with_its_message(tmp_path, monkeypatch, route):
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True, busy_steer_reply_expected=True)
    event = _event()
    await _steer(runner, agent, event, route)
    [(text, recorded)] = _turn(runner).steers
    assert text == agent.payloads[0].strip()
    assert recorded.source.user_id == event.source.user_id and recorded.reply_expected is False


@pytest.mark.asyncio
async def test_redteam_a_steer_the_agent_refused_is_not_recorded(tmp_path, monkeypatch):
    runner, agent, adapter = _runner(tmp_path, monkeypatch, agent=_Agent(accepts=False),
                                     busy_steer_same_sender=True, busy_steer_reply_expected=True)
    await _steer(runner, agent, _event(reply_expected=True))
    assert _turn(runner).steers == []
    assert adapter._pending_messages.get(SESSION) is not None


@pytest.mark.asyncio
async def test_redteam_someone_elses_message_queues_and_never_joins_the_turns_verdict(tmp_path, monkeypatch):
    runner, agent, adapter = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True,
                                     busy_steer_reply_expected=True)
    await _steer(runner, agent, _event("bob asks Lara", BOB, reply_expected=True))
    assert agent.payloads == [] and _turn(runner).steers == []
    turn_ctx, _, _ = _settle(runner, leftover=None)
    assert turn_ctx.reply_expected is True and adapter._pending_messages[SESSION].source == BOB


@pytest.mark.asyncio
async def test_redteam_a_turn_slot_held_by_another_agent_records_nothing(tmp_path, monkeypatch):
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_reply_expected=True)
    runner._hm_busy_steer(_event(), _Agent(), SESSION)
    assert _turn(runner).steers == []


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_expected, armed", [(True, True), (False, False), (None, False)])
async def test_an_addressed_steer_arms_the_turns_reply_reminder(tmp_path, monkeypatch, reply_expected, armed):
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True, busy_steer_reply_expected=True)
    agent._reply_required = False
    await _steer(runner, agent, _event(reply_expected=reply_expected))
    assert agent._reply_required is armed


# --- the turn's end ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", [False, None, True])
async def test_a_leftover_steer_gives_the_next_turn_its_own_verdict(tmp_path, monkeypatch, verdict):
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True, busy_steer_reply_expected=True)
    await _steer(runner, agent, _event(reply_expected=verdict))
    turn_ctx, result, _ = _settle(runner, leftover=agent.payloads[0].strip())
    assert turn_ctx.reply_expected is True
    _wire_followup(runner, monkeypatch)
    merged = await _followup(runner, turn_ctx, result, pending=result["pending_steer"])
    followup = runner._run_agent.await_args.kwargs
    assert followup["reply_expected"] is verdict
    assert (followup["persist_user_display_metadata"] or {}).get("reply_expected") is verdict
    assert merged["queued_terminal_reply_expected"] is verdict
    assert _turn(runner).steers == []


@pytest.mark.asyncio
@pytest.mark.parametrize("verdicts, combined", [
    ((False, False), False), ((False, None), None), ((None, False), None), ((False, True), True),
    ((True, False), True), ((None, True), True),
])
async def test_several_leftover_steers_combine_as_one_turn_answering_them_all(tmp_path, monkeypatch, verdicts,
                                                                              combined):
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True, busy_steer_reply_expected=True)
    for index, verdict in enumerate(verdicts):
        await _steer(runner, agent, _event(f"follow-up {index}", reply_expected=verdict, message_id=f"20{index}.1"))
    leftover = "\n".join(payload.strip() for payload in agent.payloads)
    turn_ctx, result, _ = _settle(runner, leftover=leftover)
    _wire_followup(runner, monkeypatch)
    await _followup(runner, turn_ctx, result, pending=leftover)
    assert runner._run_agent.await_args.kwargs["reply_expected"] is combined


@pytest.mark.asyncio
@pytest.mark.parametrize("opener, steered, judged", [
    (False, True, True), (False, None, None), (True, False, True), (False, False, False), (None, False, None),
])
async def test_a_steer_the_model_read_folds_into_the_turns_own_verdict(tmp_path, monkeypatch, opener, steered,
                                                                       judged):
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True, busy_steer_reply_expected=True)
    _turn(runner).event.reply_expected = _turn(runner).ctx.reply_expected = opener
    await _steer(runner, agent, _event(reply_expected=steered))
    turn_ctx, _, response = _settle(runner, leftover=None)
    assert turn_ctx.reply_expected is judged
    assert response["queued_terminal_reply_expected"] is judged


@pytest.mark.asyncio
async def test_a_read_and_a_leftover_steer_each_judge_their_own_turn(tmp_path, monkeypatch):
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True, busy_steer_reply_expected=True)
    _turn(runner).event.reply_expected = _turn(runner).ctx.reply_expected = False
    await _steer(runner, agent, _event("read in this turn", reply_expected=True, message_id="201.1"))
    await _steer(runner, agent, _event("arrived too late", reply_expected=False, message_id="202.1"))
    turn_ctx, result, _ = _settle(runner, leftover=agent.payloads[1].strip())
    assert turn_ctx.reply_expected is True
    _wire_followup(runner, monkeypatch)
    await _followup(runner, turn_ctx, result, pending=result["pending_steer"])
    assert runner._run_agent.await_args.kwargs["reply_expected"] is False


@pytest.mark.asyncio
async def test_redteam_an_interrupt_message_never_borrows_a_steers_verdict(tmp_path, monkeypatch):
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True, busy_steer_reply_expected=True)
    await _steer(runner, agent, _event(reply_expected=False))
    turn_ctx, result, _ = _settle(runner, leftover=agent.payloads[0].strip())
    _wire_followup(runner, monkeypatch)
    await _followup(runner, turn_ctx, {**result, "interrupted": True}, pending="stop and do this instead")
    assert runner._run_agent.await_args.kwargs["reply_expected"] is None


@pytest.mark.asyncio
async def test_the_chains_last_turn_keeps_the_verdict_its_own_steers_settled(tmp_path, monkeypatch):
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True, busy_steer_reply_expected=True)
    await _steer(runner, agent, _event(reply_expected=False))
    turn_ctx, result, _ = _settle(runner, leftover=agent.payloads[0].strip())
    _wire_followup(runner, monkeypatch, followup_result={
        "final_response": "NO_REPLY", "messages": [], "queued_terminal_reply_expected": True})
    merged = await _followup(runner, turn_ctx, result, pending=result["pending_steer"])
    assert merged["queued_terminal_reply_expected"] is True


# --- what the thread sees ---------------------------------------------------------------------


def _handler_runner(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters, runner._pending_messages = {}, {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=SESSION, session_id="sess", created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.SLACK, chat_type="group")
    runner.session_store.load_transcript.return_value = []
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_kw: 100_000)
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("steered, silent", [(False, True), (True, False)], ids=["unaddressed", "addressed"])
async def test_an_unaddressed_follow_up_that_became_its_own_turn_may_stay_silent(tmp_path, monkeypatch, steered,
                                                                                silent):
    """The production case: two unaddressed messages in a shared thread from one person; the second
    steered into the first's single model call and ran as the next turn."""
    runner, agent, _ = _runner(tmp_path, monkeypatch, busy_steer_same_sender=True, busy_steer_reply_expected=True)
    _turn(runner).event.reply_expected = _turn(runner).ctx.reply_expected = False
    await _steer(runner, agent, _event("就是应该有多少个", reply_expected=steered))
    turn_ctx, result, _ = _settle(runner, leftover=agent.payloads[0].strip())
    _wire_followup(runner, monkeypatch, followup_result={
        "final_response": "[SILENT]", "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
        "api_calls": 1, "failed": False,
        "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "[SILENT]"}]})
    merged = await _followup(runner, turn_ctx, result, pending=result["pending_steer"])

    handler = _handler_runner(monkeypatch, tmp_path)
    handler._run_agent = AsyncMock(return_value=merged)
    response = await handler._handle_message_with_agent(
        _event("总结一下", reply_expected=False), OPENER, SESSION, 1)
    if silent:
        assert response == ""
    else:
        assert response and not is_intentional_silence_response(response)


# --- through the real turn runner -------------------------------------------------------------


def _system_runner(monkeypatch, tmp_path, agent_class, **display):
    import importlib
    import sys
    import types

    from gateway.config import PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    class _CaptureAdapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
            self.sent = []

        async def connect(self):
            return True

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent.append(content)
            return SendResult(success=True, message_id="sent-1")

        async def send_typing(self, chat_id, metadata=None):
            return None

        async def stop_typing(self, chat_id):
            return None

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_class
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    module = importlib.import_module("gateway.run")
    monkeypatch.setattr(module, "_hermes_home", tmp_path)
    monkeypatch.setattr(module, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"display": {"busy_input_mode": "steer", **display}}), encoding="utf-8")

    adapter = _CaptureAdapter()
    runner = object.__new__(module.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode, runner._prefill_messages, runner._ephemeral_system_prompt = {}, [], ""
    runner._reasoning_config, runner._provider_routing, runner._fallback_model = None, {}, None
    runner._session_db, runner._running_agents, runner._session_run_generation = None, {}, {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(thread_sessions_per_user=False, group_sessions_per_user=False,
                                    stt_enabled=False)
    runner._model, runner._base_url = "openai/gpt-4.1-mini", None
    return runner, adapter


SYSTEM_KEY = "agent:main:telegram:group:-1001"
SYSTEM_SOURCE = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", user_id="U_ALICE")


class _LeftoverSteerAgent:
    """The same sender's message reaches the gateway during this turn's only model call; it is
    steered and, with no later call to deliver it, comes back as the leftover steer text."""

    runner = None
    follow_up_verdict = False
    runs: list = []

    def __init__(self, **kwargs):
        self.tools = []
        self._pending_steer = None
        self._active_children = []
        self._active_children_lock = threading.Lock()

    def steer(self, text):
        self._pending_steer = (self._pending_steer + "\n" + text) if self._pending_steer else text.strip()
        return True

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).runs.append({"message": message, "reply_required": getattr(self, "_reply_required", None)})
        if len(type(self).runs) > 1:
            return {"final_response": "NO_REPLY", "messages": [], "api_calls": 1}
        import time

        turn = type(self).runner._session_state(SYSTEM_KEY).turn
        deadline = time.monotonic() + 5
        while turn.agent is not self and time.monotonic() < deadline:
            time.sleep(0.02)
        follow_up = MessageEvent(text="就是应该有多少个", message_type=MessageType.TEXT, source=SYSTEM_SOURCE,
                                 message_id="m-2", reply_expected=type(self).follow_up_verdict)
        type(self).runner._hm_busy_steer(follow_up, self, SYSTEM_KEY)
        leftover, self._pending_steer = self._pending_steer, None
        return {"final_response": "NO_REPLY", "messages": [], "api_calls": 1, "pending_steer": leftover}


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", [True, False], ids=["switch-on", "switch-off"])
async def test_a_turn_runs_its_leftover_steer_with_the_steered_messages_verdict(monkeypatch, tmp_path, switch):
    runner, adapter = _system_runner(monkeypatch, tmp_path, _LeftoverSteerAgent, busy_steer_same_sender=True,
                                     busy_steer_reply_expected=switch)
    _LeftoverSteerAgent.runner, _LeftoverSteerAgent.runs, _LeftoverSteerAgent.follow_up_verdict = runner, [], False

    result = await runner._run_agent(
        message="总结一下", context_prompt="", history=[], source=SYSTEM_SOURCE,
        session_id="sess-steer-verdict", session_key=SYSTEM_KEY, reply_expected=False)

    assert len(_LeftoverSteerAgent.runs) == 2
    assert "就是应该有多少个" in str(_LeftoverSteerAgent.runs[1]["message"])
    if switch:
        assert result["queued_terminal_reply_expected"] is False
        assert not any(content and not is_intentional_silence_response(content) for content in adapter.sent)
    else:
        assert result["queued_terminal_reply_expected"] is None
