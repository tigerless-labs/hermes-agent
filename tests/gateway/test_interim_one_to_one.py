"""Interim assistant messages only in one-to-one conversations
(``display.interim_assistant_messages_one_to_one_only``).

Interim messages go out while the model has not yet decided whether to answer. In a shared
conversation where the turn may stay silent, that posts a line and then nothing. With this switch
a turn sends interim messages only when the message that opened it carries the adapter's verdict
that the conversation is not a group chat; a group chat, or no verdict at all, sends none.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.interim_scope import group_chat_metadata
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway import test_busy_steer_reply_expected as steer_harness
from tests.gateway import test_gateway_silence_tokens as silence_harness

SLACK = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group", user_id="U1", thread_id="100.1",
                      scope_id="T1")
VERDICTS = {"one-to-one": False, "group chat": True}
UNCLEAR = {"no verdict": None, "verdict as text": "false", "verdict as number": 0}


def _event(group_chat=None, *, metadata_missing=False) -> MessageEvent:
    metadata = {} if metadata_missing else {"slack_group_chat": group_chat}
    return MessageEvent(text="look this up", message_type=MessageType.TEXT, source=SLACK, message_id="200.2",
                        metadata=metadata)


def _display(tmp_path, monkeypatch, **display):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"display": {"platforms": {"slack": {"interim_assistant_messages": True}}, **display}}),
        encoding="utf-8")
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner = gateway_run.GatewayRunner(config=GatewayConfig())
    return runner, runner._run_agent_display_settings(SLACK)


def test_the_metadata_carries_only_a_plain_verdict() -> None:
    assert group_chat_metadata(_event(True)) == {"group_chat": True}
    assert group_chat_metadata(_event(False)) == {"group_chat": False}
    for unclear in UNCLEAR.values():
        assert group_chat_metadata(_event(unclear)) == {}
    assert group_chat_metadata(_event(metadata_missing=True)) == {}


@pytest.mark.parametrize("group_chat", [*VERDICTS.values(), *UNCLEAR.values()],
                         ids=[*VERDICTS, *UNCLEAR])
def test_switch_off_keeps_upstream_interim_messages(tmp_path, monkeypatch, group_chat) -> None:
    runner, disp = _display(tmp_path, monkeypatch)
    assert disp.interim_assistant_messages_enabled
    scoped = runner._run_agent_interim_scope(disp, group_chat_metadata(_event(group_chat)))
    assert scoped.interim_assistant_messages_enabled


def test_a_one_to_one_turn_sends_interim_messages(tmp_path, monkeypatch) -> None:
    runner, disp = _display(tmp_path, monkeypatch, interim_assistant_messages_one_to_one_only=True)
    scoped = runner._run_agent_interim_scope(disp, group_chat_metadata(_event(False)))
    assert scoped.interim_assistant_messages_enabled


@pytest.mark.parametrize("metadata", [group_chat_metadata(_event(True)), {}, None, {"group_chat": "false"}],
                         ids=["group chat", "no verdict", "no metadata", "verdict as text"])
def test_redteam_a_group_chat_or_an_unclear_verdict_sends_no_interim_message(tmp_path, monkeypatch, metadata) -> None:
    runner, disp = _display(tmp_path, monkeypatch, interim_assistant_messages_one_to_one_only=True)
    scoped = runner._run_agent_interim_scope(disp, metadata)
    assert not scoped.interim_assistant_messages_enabled
    assert scoped.tool_progress_enabled == disp.tool_progress_enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("group_chat", VERDICTS.values(), ids=VERDICTS.keys())
async def test_a_first_turn_carries_its_messages_verdict(monkeypatch, tmp_path, group_chat) -> None:
    runner = silence_harness._runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": [], "api_calls": 1,
                                                "failed": False})
    event = MessageEvent(text="look this up", source=silence_harness._source(), message_id="msg-42",
                         metadata={"slack_group_chat": group_chat})
    await runner._handle_message_with_agent(event, silence_harness._source(), "agent:main:telegram:group:-1001:12345", 1)
    assert runner._run_agent.await_args.kwargs["persist_user_display_metadata"]["group_chat"] is group_chat


@pytest.mark.asyncio
@pytest.mark.parametrize("group_chat", VERDICTS.values(), ids=VERDICTS.keys())
async def test_redteam_a_turn_taken_from_the_queue_carries_its_own_messages_verdict(tmp_path, monkeypatch,
                                                                                    group_chat) -> None:
    runner, _, _ = steer_harness._runner(tmp_path, monkeypatch)
    turn_ctx, result, _ = steer_harness._settle(runner, leftover=None)
    steer_harness._wire_followup(runner, monkeypatch)
    queued = steer_harness._event("queued follow-up", message_id="300.3")
    queued.metadata["slack_group_chat"] = group_chat
    await steer_harness._followup(runner, turn_ctx, result, pending=queued.text, pending_event=queued)
    assert runner._run_agent.await_args.kwargs["persist_user_display_metadata"]["group_chat"] is group_chat
