"""The group-chat verdict on a message's source, interim messages only in one-to-one conversations
(``display.interim_assistant_messages_one_to_one_only``), and the verdict as a per-turn session value.

Interim messages go out while the model has not yet decided whether to answer. In a shared
conversation where the turn may stay silent, that posts a line and then nothing. The adapter's
verdict rides on the message's source (``SessionSource.group_chat``). With the switch on, a turn
sends interim messages only when that verdict is a plain ``False``. The same verdict is bound per
turn as ``HERMES_SESSION_GROUP_CHAT`` for plugins.
"""

from __future__ import annotations

import dataclasses

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.interim_scope import group_chat_value, interim_allowed
from gateway.session import SessionContext, SessionSource
from gateway.session_context import get_session_env

SLACK = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group", user_id="U1", thread_id="100.1",
                      scope_id="T1")
VERDICTS = {"one-to-one": False, "group chat": True}
UNCLEAR = {"no verdict": None, "verdict as text": "false", "verdict as number": 0}


def _source(group_chat) -> SessionSource:
    return dataclasses.replace(SLACK, group_chat=group_chat)


def _runner(tmp_path, monkeypatch, **display):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"display": {"platforms": {"slack": {"interim_assistant_messages": True}}, **display}}),
        encoding="utf-8")
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    return gateway_run.GatewayRunner(config=GatewayConfig())


def test_only_a_plain_false_counts_as_one_to_one() -> None:
    assert interim_allowed(_source(False), True)
    for group_chat in [True, *UNCLEAR.values()]:
        assert not interim_allowed(_source(group_chat), True)
    assert all(interim_allowed(_source(group_chat), False) for group_chat in [True, False, *UNCLEAR.values()])


@pytest.mark.parametrize("group_chat", [*VERDICTS.values(), *UNCLEAR.values()], ids=[*VERDICTS, *UNCLEAR])
def test_switch_off_keeps_upstream_interim_messages(tmp_path, monkeypatch, group_chat) -> None:
    runner = _runner(tmp_path, monkeypatch)
    assert runner._run_agent_display_settings(_source(group_chat)).interim_assistant_messages_enabled


def test_a_one_to_one_turn_sends_interim_messages(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch, interim_assistant_messages_one_to_one_only=True)
    assert runner._run_agent_display_settings(_source(False)).interim_assistant_messages_enabled


@pytest.mark.parametrize("group_chat", [True, *UNCLEAR.values()], ids=["group chat", *UNCLEAR])
def test_redteam_a_group_chat_or_an_unclear_verdict_sends_no_interim_message(tmp_path, monkeypatch,
                                                                             group_chat) -> None:
    runner = _runner(tmp_path, monkeypatch, interim_assistant_messages_one_to_one_only=True)
    one_to_one = runner._run_agent_display_settings(_source(False))
    disp = runner._run_agent_display_settings(_source(group_chat))
    assert not disp.interim_assistant_messages_enabled
    assert disp.tool_progress_enabled == one_to_one.tool_progress_enabled


@pytest.mark.parametrize("group_chat, bound", [(True, "1"), (False, "0"), (None, ""), ("true", "")],
                         ids=["group chat", "one-to-one", "no verdict", "verdict as text"])
def test_each_turn_binds_its_sources_verdict(tmp_path, monkeypatch, group_chat, bound) -> None:
    runner = _runner(tmp_path, monkeypatch)
    tokens = runner._set_session_env(SessionContext(source=_source(group_chat), connected_platforms=[],
                                                    home_channels={}))
    try:
        assert get_session_env("HERMES_SESSION_GROUP_CHAT") == bound == group_chat_value(_source(group_chat))
    finally:
        runner._clear_session_env(tokens)
    assert get_session_env("HERMES_SESSION_GROUP_CHAT") == ""


def test_redteam_the_verdict_never_travels_in_a_serialized_source() -> None:
    marked = _source(False)
    assert "group_chat" not in marked.to_dict()
    assert SessionSource.from_dict({**SLACK.to_dict(), "group_chat": False}).group_chat is None
    assert marked == _source(True) == SLACK


@pytest.mark.asyncio
@pytest.mark.parametrize("group_chat", VERDICTS.values(), ids=VERDICTS.keys())
async def test_redteam_a_turn_taken_from_the_queue_runs_with_its_own_messages_verdict(tmp_path, monkeypatch,
                                                                                     group_chat) -> None:
    from tests.gateway import test_busy_steer_reply_expected as steer_harness

    runner, _, _ = steer_harness._runner(tmp_path, monkeypatch)
    turn_ctx, result, _ = steer_harness._settle(runner, leftover=None)
    steer_harness._wire_followup(runner, monkeypatch)
    queued = steer_harness._event("queued follow-up", source=dataclasses.replace(steer_harness.OPENER,
                                                                                 group_chat=group_chat),
                                  message_id="300.3")
    await steer_harness._followup(runner, turn_ctx, result, pending=queued.text, pending_event=queued)
    assert runner._run_agent.await_args.kwargs["source"].group_chat is group_chat
