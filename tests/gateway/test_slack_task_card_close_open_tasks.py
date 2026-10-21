"""A task still running when its card ends is closed on the card (``extra.task_card_close_open_tasks``).

``/stop`` ends the turn while a tool runs: the run stops being current before the tool's completion
arrives, so upstream finalizes the card with that task in progress and Slack spins it for good.
"""

import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.slack.adapter import SlackAdapter


def _adapter(close_open_tasks):
    extra = {"native_task_cards": True}
    if close_open_tasks is not None:
        extra["task_card_close_open_tasks"] = close_open_tasks
    adapter = SlackAdapter(PlatformConfig(extra=extra))
    client = SimpleNamespace(api_call=AsyncMock(side_effect=lambda *a, **kw: {"ts": "900.1"}))
    adapter._get_client = Mock(return_value=client)
    adapter._app = object()
    return adapter, client


class Events(queue.Queue):
    """The run stays current until ``current_for`` events have been drained."""

    def __init__(self, events, current_for, late=()):
        super().__init__()
        self.drained = 0
        self.current_for = current_for
        self.late = list(late)
        for event in events:
            self.put(event)

    def get_nowait(self):
        event = super().get_nowait()
        self.drained += 1
        if self.drained == self.current_for:
            for late in self.late:
                self.put(late)
        return event


def _started(call_id, tool="terminal"):
    return {"type": "tool.started", "tool_call_id": call_id, "tool_name": tool, "preview": "sleep 60"}


def _completed(call_id, tool="terminal"):
    return {"type": "tool.completed", "tool_call_id": call_id, "tool_name": tool, "is_error": False}


async def _card(close_open_tasks, events, *, interrupted):
    adapter, client = _adapter(close_open_tasks)
    agent = SimpleNamespace(is_interrupted=False)
    ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="C1"), _progress_reply_to="100.2", _progress_metadata={"thread_id": "100.1"},
        _cleanup_progress=False, progress_queue=events, tool_progress_enabled=False, agent_holder=[agent],
    )

    def still_current():
        current = events.drained < events.current_for
        agent.is_interrupted = interrupted and not current
        return current

    ctx._run_still_current = still_current
    await TurnRunner(None, ctx)._send_native_task_card_progress(adapter)
    calls = client.api_call.await_args_list
    frames = [{chunk["id"]: chunk for chunk in call.kwargs["json"]["chunks"] if chunk["type"] == "task_update"}
              for call in calls if call.args[0] == "chat.appendStream"]
    return [call.args[0] for call in calls], frames


@pytest.mark.asyncio
async def test_a_stopped_turns_running_task_is_closed_before_the_card_ends():
    events = Events([_started("read", "read_file"), _completed("read", "read_file"),
                     _started("sleep")], current_for=3)
    methods, frames = await _card(True, events, interrupted=True)
    assert methods[-1] == "chat.stopStream"
    assert frames[-2]["sleep"]["status"] == "in_progress"
    last = frames[-1]
    assert all(task["status"] != "in_progress" for task in last.values())
    assert last["sleep"]["title"] == frames[-2]["sleep"]["title"]
    assert last["sleep"]["status"] == "error"
    assert last["read"] == frames[-2]["read"]


@pytest.mark.asyncio
async def test_a_task_left_open_by_a_finished_turn_closes_as_complete():
    methods, frames = await _card(True, Events([_started("sleep")], current_for=1), interrupted=False)
    assert methods[-1] == "chat.stopStream"
    assert frames[-1]["sleep"]["status"] == "complete"


@pytest.mark.asyncio
async def test_closing_adds_no_task_that_arrived_after_the_stop():
    events = Events([_started("sleep")], current_for=1, late=[_started("late")])
    _methods, frames = await _card(True, events, interrupted=True)
    assert all("late" not in frame for frame in frames)


@pytest.mark.asyncio
async def test_a_card_with_nothing_open_ends_as_upstreams_does():
    plain = await _card(None, Events([_started("a"), _completed("a")], current_for=2), interrupted=False)
    switched = await _card(True, Events([_started("a"), _completed("a")], current_for=2), interrupted=False)
    assert switched == plain


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, False])
async def test_switched_off_upstream_ends_the_card_with_the_task_running(value):
    methods, frames = await _card(value, Events([_started("sleep")], current_for=1), interrupted=True)
    assert methods == ["chat.startStream", "chat.appendStream", "chat.stopStream"]
    assert frames[-1]["sleep"]["status"] == "in_progress"
