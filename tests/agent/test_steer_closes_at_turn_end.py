"""A finished turn takes no more steers (``agent.steer_closes_at_turn_end``).

The turn's last read of the steer buffer happens in the finalizer. Upstream the agent keeps taking
steers after it: the text sits in the buffer of an agent whose turn is over, and is read only when
the session's next turn starts (possibly someone else's) or is lost with the agent. With the switch
on, the last read also closes the buffer under the same lock, so a steer is either in that read or
refused (the gateway then queues it as its own turn); the next turn opens the buffer again.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_steer import _bare_agent
from tests.agent.test_run_agent import _mock_response


def _switch(monkeypatch, on: bool) -> None:
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda *a, **k: {"agent": {"steer_closes_at_turn_end": on}})


def _agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._disable_streaming = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run_turn(agent, message: str, during_call=None) -> dict:
    payloads = []

    def model_call(api_kwargs):
        payloads.append([dict(m) for m in api_kwargs["messages"]])
        if during_call:
            during_call()
        return _mock_response(content=f"reply to {message}", finish_reason="stop")

    agent._interruptible_api_call = model_call
    with (
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(message)
    result["_payloads"] = payloads
    return result


def test_switch_off_a_finished_agent_still_takes_a_steer_as_upstream(monkeypatch):
    _switch(monkeypatch, False)
    agent = _agent()
    _run_turn(agent, "first")
    assert agent.steer("late") is True
    assert agent._pending_steer == "late"


def test_a_finished_turn_refuses_a_steer_and_keeps_nothing(monkeypatch):
    _switch(monkeypatch, True)
    agent = _agent()
    _run_turn(agent, "first")
    assert agent.steer_intake_closed() is True
    assert agent.steer("late") is False
    assert agent._pending_steer is None


def test_a_steer_during_the_last_call_still_comes_back_as_the_leftover(monkeypatch):
    _switch(monkeypatch, True)
    agent = _agent()
    result = _run_turn(agent, "first", during_call=lambda: agent.steer("during the call"))
    assert result.get("pending_steer") == "during the call"
    assert agent._pending_steer is None


def test_the_next_turn_takes_steers_again(monkeypatch):
    _switch(monkeypatch, True)
    agent = _agent()
    _run_turn(agent, "first")
    assert agent.steer("late") is False
    accepted = []
    result = _run_turn(agent, "second", during_call=lambda: accepted.append(agent.steer("mid-turn")))
    assert accepted == [True]
    assert result.get("pending_steer") == "mid-turn"
    assert agent.steer_intake_closed() is True


@pytest.mark.parametrize("close", [True, False], ids=["closing read", "plain read"])
def test_a_steer_racing_the_last_read_is_read_or_refused_never_left_behind(close):
    agent = _bare_agent()
    texts = [f"steer-{index}" for index in range(400)]
    outcomes: dict[str, bool] = {}
    start = threading.Barrier(3)

    def send(batch):
        start.wait()
        for text in batch:
            outcomes[text] = agent.steer(text)

    senders = [threading.Thread(target=send, args=(texts[half::2],)) for half in (0, 1)]
    for sender in senders:
        sender.start()
    start.wait()
    read = agent._last_steer_read(close=close) or ""
    for sender in senders:
        sender.join()

    left_behind = agent._pending_steer
    read_lines = set(read.splitlines())
    accepted = {text for text, ok in outcomes.items() if ok}
    if close:
        assert left_behind is None
        assert accepted == read_lines
    else:
        assert accepted == read_lines | set((left_behind or "").splitlines())


def test_redteam_a_parent_closing_does_not_close_its_subagent():
    parent, child = _bare_agent(), _bare_agent()
    parent._last_steer_read(close=True)
    assert parent.steer("to parent") is False
    assert child.steer("to child") is True
