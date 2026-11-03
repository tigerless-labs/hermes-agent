"""A card Slack sealed during a long turn does not stay behind with its working title
(``extra.task_card_replace_sealed``).

Upstream reopens a fresh card when an update hits a sealed stream and leaves the sealed one in the
thread, still titled as working; a seal that lands after the last update leaves the only card that way.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter

METADATA = {"thread_id": "100.1"}
RUNNING = [{"id": "call-1", "title": "terminal", "status": "in_progress"}]
DONE = [{"id": "call-1", "title": "terminal", "status": "complete"},
        {"id": "call-2", "title": "web_search", "status": "complete"}]


class Sealed(Exception):
    def __init__(self):
        super().__init__("sealed")
        self.response = {"ok": False, "error": "message_not_in_streaming_state"}


class Slack:
    """Fake Web API: cards get ``card-N``; a sealed card refuses appends and stops."""

    def __init__(self, *, failing=()):
        self.cards, self.sealed, self.failing, self.calls = 0, set(), set(failing), []

    async def __call__(self, method, *, json):
        self.calls.append((method, dict(json)))
        if method in {"chat.appendStream", "chat.stopStream"} and json["ts"] in self.sealed:
            raise Sealed()
        if method in self.failing:
            raise RuntimeError(f"{method} failed")
        if method == "chat.startStream":
            self.cards += 1
            return {"ok": True, "ts": f"card-{self.cards}"}
        return {"ok": True}

    def methods(self):
        return [(method, payload.get("ts")) for method, payload in self.calls]

    def deleted(self):
        return [payload["ts"] for method, payload in self.calls if method == "chat.delete"]


def _adapter(slack, replace_sealed):
    extra = {"native_task_cards": True}
    if replace_sealed is not None:
        extra["task_card_replace_sealed"] = replace_sealed
    adapter = SlackAdapter(PlatformConfig(extra=extra))
    adapter._get_client = Mock(return_value=SimpleNamespace(api_call=AsyncMock(side_effect=slack.__call__)))
    adapter._app = object()
    return adapter


async def _sealed_mid_turn(slack, replace_sealed):
    adapter = _adapter(slack, replace_sealed)
    await adapter.send_native_task_card_progress("C1", RUNNING, metadata=METADATA)
    slack.sealed.add("card-1")
    result = await adapter.send_native_task_card_progress("C1", DONE, metadata=METADATA)
    return adapter, result


async def _sealed_before_the_end(slack, replace_sealed, title="Worked for 12m 3s"):
    adapter = _adapter(slack, replace_sealed)
    await adapter.send_native_task_card_progress("C1", DONE, metadata=METADATA)
    slack.sealed.add("card-1")
    await adapter.stop_native_task_card_progress("C1", metadata=METADATA, title=title)
    return adapter


def _tasks(payload):
    return [(chunk["id"], chunk["status"]) for chunk in payload["chunks"] if chunk["type"] == "task_update"]


@pytest.mark.asyncio
async def test_a_card_sealed_mid_turn_is_deleted_once_its_reopened_card_is_up():
    slack = Slack()
    _adapter_, result = await _sealed_mid_turn(slack, True)
    assert result.success and result.message_id == "card-2"
    assert slack.deleted() == ["card-1"]
    methods = slack.methods()
    assert methods.index(("chat.appendStream", "card-2")) < methods.index(("chat.delete", "card-1"))


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_sealed", [None, False])
async def test_switched_off_the_sealed_card_stays_as_upstream_leaves_it(replace_sealed):
    slack = Slack()
    _adapter_, result = await _sealed_mid_turn(slack, replace_sealed)
    assert result.success and result.message_id == "card-2"
    assert slack.deleted() == []


@pytest.mark.asyncio
async def test_a_card_sealed_before_the_end_is_reopened_with_the_end_title_and_its_tasks_then_deleted():
    slack = Slack()
    adapter = await _sealed_before_the_end(slack, True)
    stop = next(payload for method, payload in slack.calls if method == "chat.stopStream" and payload["ts"] == "card-2")
    assert [chunk["title"] for chunk in stop["chunks"] if chunk["type"] == "plan_update"] == ["Worked for 12m 3s"]
    assert _tasks(stop) == [("call-1", "complete"), ("call-2", "complete")]
    methods = slack.methods()
    assert methods.index(("chat.stopStream", "card-2")) < methods.index(("chat.delete", "card-1"))
    assert slack.deleted() == ["card-1"] and adapter._native_task_card_streams == {}


@pytest.mark.asyncio
async def test_the_reopened_end_card_goes_to_the_same_thread_and_person():
    slack = Slack()
    await _sealed_before_the_end(slack, True)
    first, reopened = [payload for method, payload in slack.calls if method == "chat.startStream"]
    assert reopened == first


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_sealed, title", [(True, ""), (None, "Worked for 12m 3s"), (False, "Worked for 12m 3s")])
async def test_without_an_end_title_or_switched_off_a_card_sealed_before_the_end_is_left_alone(replace_sealed, title):
    slack = Slack()
    adapter = await _sealed_before_the_end(slack, replace_sealed, title)
    assert slack.cards == 1 and slack.deleted() == []
    assert adapter._native_task_card_streams == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["chat.startStream", "chat.stopStream"])
async def test_the_sealed_card_is_kept_when_its_replacement_does_not_land(failing):
    slack = Slack()
    adapter = _adapter(slack, True)
    await adapter.send_native_task_card_progress("C1", DONE, metadata=METADATA)
    slack.sealed.add("card-1")
    slack.failing.add(failing)
    await adapter.stop_native_task_card_progress("C1", metadata=METADATA, title="Worked for 12m 3s")
    assert slack.deleted() == [] and adapter._native_task_card_streams == {}


@pytest.mark.asyncio
async def test_a_failed_delete_leaves_the_turn_going():
    slack = Slack(failing={"chat.delete"})
    adapter, result = await _sealed_mid_turn(slack, True)
    assert result.success and result.message_id == "card-2"
    await adapter.stop_native_task_card_progress("C1", metadata=METADATA, title="Worked for 1s")
    assert ("chat.stopStream", "card-2") in slack.methods()


@pytest.mark.asyncio
async def test_only_this_streams_own_sealed_card_is_ever_deleted():
    slack = Slack()
    adapter = _adapter(slack, True)
    other = {"thread_id": "200.1"}
    await adapter.send_native_task_card_progress("C1", RUNNING, metadata=METADATA)
    await adapter.send_native_task_card_progress("C1", RUNNING, metadata=other)
    slack.sealed.add("card-1")
    await adapter.send_native_task_card_progress("C1", DONE, metadata=METADATA)
    await adapter.stop_native_task_card_progress("C1", metadata=other, title="Worked for 1s")
    await adapter.stop_native_task_card_progress("C1", metadata=METADATA, title="Worked for 1s")
    assert slack.deleted() == ["card-1"]
