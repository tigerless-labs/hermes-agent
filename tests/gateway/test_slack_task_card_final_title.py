"""The card ends titled with how the turn ended and how long it took (``extra.task_card_done_title`` /
``extra.task_card_stopped_title``).

Upstream seals the card with ``chat.stopStream`` alone, so a finished turn's card keeps its working title.
"""

import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gateway.run_turn_runner as run_turn_runner
from gateway.config import PlatformConfig
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.slack.adapter import SlackAdapter

WORKING, DONE, STOPPED = "Lara is working", "Lara worked for {duration}", "Lara stopped after {duration}"


def _adapter(**titles):
    extra = {"native_task_cards": True, "task_card_title": WORKING, **titles}
    adapter = SlackAdapter(PlatformConfig(extra=extra))
    client = SimpleNamespace(api_call=AsyncMock(side_effect=lambda *a, **kw: {"ts": "900.1"}))
    adapter._get_client = Mock(return_value=client)
    adapter._app = object()
    return adapter, client


class Clock:
    def __init__(self, *readings):
        self.readings = list(readings)

    def __call__(self):
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]


async def _end_card(adapter, *, interrupted=False, seconds=75.0, monkeypatch):
    monkeypatch.setattr(run_turn_runner, "time", SimpleNamespace(monotonic=Clock(1000.0, 1000.0 + seconds)))
    events = queue.Queue()
    events.put({"type": "tool.started", "tool_call_id": "call-1", "tool_name": "terminal", "preview": "ls"})
    agent = SimpleNamespace(is_interrupted=False)
    ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="C1"), _progress_reply_to="100.2", _progress_metadata={"thread_id": "100.1"},
        _cleanup_progress=False, progress_queue=events, tool_progress_enabled=False, agent_holder=[agent],
    )

    def still_current():
        current = not events.empty()
        agent.is_interrupted = interrupted and not current
        return current

    ctx._run_still_current = still_current
    await TurnRunner(None, ctx)._send_native_task_card_progress(adapter)


def _stop_payload(client):
    stop = client.api_call.await_args_list[-1]
    assert stop.args[0] == "chat.stopStream"
    return stop.kwargs["json"]


def _plan_titles(payload):
    return [chunk["title"] for chunk in payload.get("chunks", []) if chunk["type"] == "plan_update"]


@pytest.mark.asyncio
async def test_a_finished_turns_card_ends_with_the_done_title_and_its_length(monkeypatch):
    adapter, client = _adapter(task_card_done_title=DONE, task_card_stopped_title=STOPPED)
    await _end_card(adapter, seconds=75.0, monkeypatch=monkeypatch)
    [title] = _plan_titles(_stop_payload(client))
    assert title == adapter.native_task_card_final_title(75.0, stopped=False)
    assert title != WORKING and title.startswith(DONE.split("{")[0])


@pytest.mark.asyncio
async def test_a_stopped_turns_card_ends_with_the_stopped_title(monkeypatch):
    adapter, client = _adapter(task_card_done_title=DONE, task_card_stopped_title=STOPPED)
    await _end_card(adapter, interrupted=True, seconds=40.0, monkeypatch=monkeypatch)
    [title] = _plan_titles(_stop_payload(client))
    assert title == adapter.native_task_card_final_title(40.0, stopped=True)
    assert title.startswith(STOPPED.split("{")[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("titles", [{}, {"task_card_done_title": ""}, {"task_card_stopped_title": STOPPED}])
async def test_without_a_done_title_a_finished_card_ends_as_upstreams_does(titles, monkeypatch):
    adapter, client = _adapter(**titles)
    await _end_card(adapter, monkeypatch=monkeypatch)
    assert _stop_payload(client) == {"channel": "C1", "ts": "900.1"}


@pytest.mark.asyncio
async def test_without_a_stopped_title_a_stopped_card_ends_as_upstreams_does(monkeypatch):
    adapter, client = _adapter(task_card_done_title=DONE)
    await _end_card(adapter, interrupted=True, monkeypatch=monkeypatch)
    assert "chunks" not in _stop_payload(client)


@pytest.mark.asyncio
async def test_an_adapter_without_final_titles_is_stopped_as_before(monkeypatch):
    stops = []

    class PlainAdapter:
        async def send_native_task_card_progress(self, *args, **kwargs):
            return SimpleNamespace(success=True, message_id="900.1")

        async def stop_native_task_card_progress(self, chat_id, *, reply_to=None, metadata=None):
            stops.append(chat_id)

    await _end_card(PlainAdapter(), monkeypatch=monkeypatch)
    assert stops == ["C1"]


def test_the_duration_reads_in_the_largest_units_that_fit():
    adapter, _client = _adapter(task_card_done_title="{duration}")
    seconds, minutes, hours = (adapter.native_task_card_final_title(s, stopped=False) for s in (59.9, 75.0, 3725.0))
    assert seconds.endswith("s") and "m" not in seconds
    assert minutes.startswith("1m ") and minutes.endswith("s")
    assert hours.startswith("1h ") and hours.endswith("m")
    assert adapter.native_task_card_final_title(-3.0, stopped=False) == adapter.native_task_card_final_title(
        0.0, stopped=False)


def test_a_title_without_a_duration_or_with_other_braces_is_kept_as_written():
    adapter, _client = _adapter(task_card_done_title="Done {ok}", task_card_stopped_title="Stopped")
    assert adapter.native_task_card_final_title(5.0, stopped=False) == "Done {ok}"
    assert adapter.native_task_card_final_title(5.0, stopped=True) == "Stopped"


def test_a_long_title_is_cut_to_what_slack_takes():
    adapter, _client = _adapter(task_card_done_title="x" * 400 + "{duration}")
    assert len(adapter.native_task_card_final_title(5.0, stopped=False)) == 256
