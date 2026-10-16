"""A message that arrives after the agent's turn ended queues as its own turn
(``agent.steer_closes_at_turn_end``).

Between the agent's last read of its steer buffer and the gateway releasing the turn slot, the slot
still names the agent, so a follow-up from the turn's own sender is steered into an agent that will
not read it again. With the switch on the agent refuses it, and every steering route queues the
message with its own sender, verdict and files; the adapter runs a late queued message as the next
turn. Switch off keeps upstream: the text is left in the finished agent's buffer.
"""

from __future__ import annotations

import importlib
import threading
import time
from types import SimpleNamespace

import pytest
import yaml

import gateway.run  # noqa: F401
from agent.interrupt_control import InterruptControlMixin
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

KEY = "agent:main:telegram:group:-1001"
SOURCE = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", user_id="U_ALICE")


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def stop_typing(self, chat_id):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _Agent(InterruptControlMixin):
    """The real steer buffer; its turn ends with the last read the finalizer makes."""

    runner = None
    closes = True
    runs: list = []

    def __init__(self, **kwargs):
        self.tools = []
        self._pending_steer, self._pending_steer_lock = None, threading.Lock()
        self._active_children, self._active_children_lock = [], threading.Lock()

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        turn = type(self).runner._session_state(KEY).turn
        deadline = time.monotonic() + 5
        while turn.agent is not self and time.monotonic() < deadline:
            time.sleep(0.02)
        type(self).runs.append(message)
        leftover = self._last_steer_read(close=type(self).closes)
        result = {"final_response": f"answer {len(type(self).runs)}", "messages": [], "api_calls": 1}
        if leftover:
            result["pending_steer"] = leftover
        return result


def _runner(monkeypatch, tmp_path):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=_Agent))
    run = importlib.import_module("gateway.run")
    monkeypatch.setattr(run, "_hermes_home", tmp_path)
    monkeypatch.setattr(run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"display": {
        "busy_input_mode": "steer", "busy_steer_same_sender": True, "busy_steer_reply_expected": True}}))
    adapter = _Adapter()
    runner = object.__new__(run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode, runner._prefill_messages, runner._ephemeral_system_prompt = {}, [], ""
    runner._reasoning_config, runner._provider_routing, runner._fallback_model = None, {}, None
    runner._session_db, runner._running_agents, runner._session_run_generation = None, {}, {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(thread_sessions_per_user=False, group_sessions_per_user=False, stt_enabled=False)
    runner._model, runner._base_url = "openai/gpt-4.1-mini", None
    runner._busy_input_mode, runner._busy_text_mode = "steer", "steer"
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    monkeypatch.setattr(runner, "_admit_bot_message_for_source", lambda source: True)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    return runner, adapter


BEFORE_THE_QUEUE_CHECK, AFTER_THE_QUEUE_CHECK = "_run_agent_finalize_streaming_tts", "_run_agent_cleanup_turn_tasks"


async def _turn_with_a_late_message(monkeypatch, tmp_path, *, closes: bool, verdict=False,
                                    where=BEFORE_THE_QUEUE_CHECK):
    """The late message arrives after the agent's last read, at *where*: before the turn checks its
    queue (it then runs in the same chain) or after it (left for the adapter's late-arrival drain);
    both before the gateway releases the slot."""
    runner, adapter = _runner(monkeypatch, tmp_path)
    _Agent.runner, _Agent.closes, _Agent.runs = runner, closes, []
    late = MessageEvent(text="one more thing", message_type=MessageType.TEXT, source=SOURCE, message_id="m-2",
                        reply_expected=verdict)
    seen = {}
    step = getattr(runner, where)

    async def arrive_after_the_agent_finished(turn_ctx, *args, **kwargs):
        if not seen:
            seen["agent"] = runner._session_state(KEY).turn.agent
            assert await runner._handle_active_session_busy_message(late, KEY) is True
            seen["recorded"] = list(runner._session_state(KEY).turn.steers)
            seen["left_in_agent"] = seen["agent"]._pending_steer
        return await step(turn_ctx, *args, **kwargs)

    monkeypatch.setattr(runner, where, arrive_after_the_agent_finished)
    result = await runner._run_agent(message="first", context_prompt="", history=[], source=SOURCE,
                                     session_id="s", session_key=KEY, reply_expected=True)
    return runner, adapter, seen, result


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", [False, True, None])
async def test_a_message_after_the_agents_last_read_runs_as_its_own_next_turn(monkeypatch, tmp_path, verdict):
    runner, adapter, seen, result = await _turn_with_a_late_message(monkeypatch, tmp_path, closes=True,
                                                                    verdict=verdict)
    assert seen["left_in_agent"] is None and seen["recorded"] == []
    assert len(_Agent.runs) == 2 and "one more thing" in str(_Agent.runs[1])
    assert result["queued_terminal_reply_expected"] is verdict
    assert adapter._pending_messages.get(KEY) is None
    assert runner._session_state(KEY).turn.agent is None


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", [False, True, None])
async def test_a_message_after_the_turns_queue_check_is_left_queued_for_the_adapter(monkeypatch, tmp_path,
                                                                                    verdict):
    runner, adapter, seen, _ = await _turn_with_a_late_message(monkeypatch, tmp_path, closes=True, verdict=verdict,
                                                               where=AFTER_THE_QUEUE_CHECK)
    queued = adapter._pending_messages.get(KEY)
    assert queued is not None and queued.text == "one more thing"
    assert queued.source.user_id == SOURCE.user_id and queued.reply_expected is verdict
    assert seen["left_in_agent"] is None and seen["recorded"] == []
    assert _Agent.runs == ["first"]
    assert runner._session_state(KEY).turn.agent is None


@pytest.mark.asyncio
@pytest.mark.parametrize("where", [BEFORE_THE_QUEUE_CHECK, AFTER_THE_QUEUE_CHECK])
async def test_switch_off_leaves_the_message_in_the_finished_agent_as_upstream(monkeypatch, tmp_path, where):
    runner, adapter, seen, _ = await _turn_with_a_late_message(monkeypatch, tmp_path, closes=False, where=where)
    assert adapter._pending_messages.get(KEY) is None
    assert _Agent.runs == ["first"]
    assert "one more thing" in (seen["agent"]._pending_steer or "")
    assert runner._session_state(KEY).turn.agent is None


@pytest.mark.asyncio
async def test_a_steer_command_after_the_turn_ended_queues_as_its_own_turn(monkeypatch, tmp_path):
    runner, adapter = _runner(monkeypatch, tmp_path)
    agent = _Agent()
    agent._last_steer_read(close=True)
    turn = runner._session_state(KEY).turn
    turn.agent = agent
    turn.ctx = SimpleNamespace(source=SOURCE, agent_holder=[agent])
    command = MessageEvent(text="/steer one more thing", message_type=MessageType.TEXT, source=SOURCE,
                           message_id="m-3")
    reply = await runner._busy_steer_command(command, KEY, SOURCE)
    queued = adapter._pending_messages.get(KEY)
    assert queued is not None and queued.text == "one more thing" and queued.source.user_id == SOURCE.user_id
    assert "queued" in reply.lower() and "rejected" not in reply.lower()
    assert agent._pending_steer is None
