"""A context compaction shows on the Slack-native task card like a tool (``extra.task_card_compaction_title``)."""

import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent.conversation_compression import (
    COMPACTION_DONE_STATUS, COMPACTION_HEARTBEAT_STATUS, COMPACTION_STATUS, PREFLIGHT_COMPRESSION_STATUS_TEMPLATE,
)
from gateway.config import PlatformConfig
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.slack.adapter import SlackAdapter

TITLE = "Compacting context"
PREFLIGHT = PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(tokens=535355, threshold=500000)
ONE_COMPACTION = [("lifecycle", PREFLIGHT), ("lifecycle", COMPACTION_STATUS),
                  ("lifecycle", COMPACTION_HEARTBEAT_STATUS), ("compacted", COMPACTION_DONE_STATUS)]
REFUSED = ("warn", "⚠️ Compression refused: the generated summary would have GROWN the conversation instead of "
                   "shrinking it. No messages were dropped — conversation continues unchanged.")


def _adapter(title=TITLE):
    extra = {"native_task_cards": True, "reply_in_thread": True}
    if title is not None:
        extra["task_card_compaction_title"] = title
    adapter = SlackAdapter(PlatformConfig(extra=extra))
    client = SimpleNamespace(api_call=AsyncMock(side_effect=lambda *a, **kw: {"ts": "900.1"}))
    adapter._get_client = Mock(return_value=client)
    adapter._app = object()
    return adapter, client


def _turn(adapter, *, cards=True, interrupted=False):
    events = queue.Queue()
    ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="C1"), progress_queue=events, _native_slack_task_cards=cards,
        _run_still_current=lambda: True, agent_holder=[SimpleNamespace(is_interrupted=interrupted)],
        mute_notification_reply=False, _status_adapter=None,
        _progress_reply_to="100.2", _progress_metadata={"thread_id": "100.1"}, _cleanup_progress=False,
        tool_progress_enabled=False,
    )
    return TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda source: adapter), ctx), events


def _queued(events: queue.Queue) -> list[dict]:
    drained = []
    while not events.empty():
        drained.append(events.get_nowait())
    return drained


def _report(turn, statuses):
    for event_type, message in statuses:
        turn._status_callback_sync(event_type, message)


def test_unset_title_keeps_upstream_compaction_off_the_card():
    adapter, _client = _adapter(title=None)
    assert adapter.native_task_card_compaction_title() == ""
    turn, events = _turn(adapter)
    _report(turn, ONE_COMPACTION)
    assert _queued(events) == []


def test_a_compaction_is_one_task_from_its_first_line_to_done():
    turn, events = _turn(_adapter()[0])
    _report(turn, ONE_COMPACTION)
    started, completed = _queued(events)
    assert started["type"] == "tool.started" and completed["type"] == "tool.completed"
    assert started["tool_call_id"] == completed["tool_call_id"]
    assert started["tool_name"] == TITLE
    assert not completed["is_error"]


def test_each_compaction_of_a_turn_is_its_own_task():
    turn, events = _turn(_adapter()[0])
    _report(turn, ONE_COMPACTION + ONE_COMPACTION)
    ids = [event["tool_call_id"] for event in _queued(events)]
    assert len(ids) == 4 and ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]


def test_a_compaction_that_ends_in_a_compression_warning_is_a_failed_task():
    turn, events = _turn(_adapter()[0])
    _report(turn, [("lifecycle", COMPACTION_STATUS), ("warn", "⚠ Slack rate limited, retrying"), REFUSED])
    started, failed = _queued(events)
    assert failed["type"] == "tool.completed" and failed["tool_call_id"] == started["tool_call_id"]
    assert failed["is_error"]
    _report(turn, ONE_COMPACTION)
    assert _queued(events)[0]["tool_call_id"] != started["tool_call_id"]


@pytest.mark.parametrize("event_type,message", [
    ("lifecycle", "⚠️  Thinking block signature invalid, suppressed rejected Anthropic replay blocks and retrying..."),
    ("lifecycle", "Session is free; loading the latest transcript..."),
    ("warn", "⚠ Context is over the compression threshold (~1 tokens >= 1) but compression is currently blocked (x)."),
    ("compacted", COMPACTION_DONE_STATUS),
    REFUSED,
])
def test_thinking_and_other_status_lines_add_no_task(event_type, message):
    turn, events = _turn(_adapter()[0])
    turn._status_callback_sync(event_type, message)
    assert _queued(events) == []


@pytest.mark.parametrize("cards,interrupted", [(False, False), (True, True)])
def test_no_task_without_a_card_lane_or_after_stop(cards, interrupted):
    turn, events = _turn(_adapter()[0], cards=cards, interrupted=interrupted)
    _report(turn, ONE_COMPACTION)
    assert _queued(events) == []


@pytest.mark.asyncio
async def test_a_turn_that_only_compacts_shows_the_card_while_it_runs():
    adapter, client = _adapter()
    turn, events = _turn(adapter)
    _report(turn, ONE_COMPACTION)
    turn._ctx._run_still_current = lambda: not events.empty()
    await turn._send_native_task_card_progress(adapter)

    methods = [call.args[0] for call in client.api_call.await_args_list]
    assert methods == ["chat.startStream", "chat.appendStream", "chat.appendStream", "chat.stopStream"]
    frames = [[chunk for chunk in call.kwargs["json"]["chunks"] if chunk["type"] == "task_update"]
              for call in client.api_call.await_args_list if call.args[0] == "chat.appendStream"]
    assert [[(task["title"], task["status"]) for task in frame] for frame in frames] == [
        [(TITLE, "in_progress")], [(TITLE, "complete")]]
