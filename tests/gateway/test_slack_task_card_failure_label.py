"""A failed task on the Slack-native card follows ``extra.task_card_failure_label``."""

import pytest

from tests.gateway.test_slack_approval_buttons import _make_adapter

METADATA = {"thread_id": "thread-1", "slack_team_id": "T1"}
TASKS = [
    {"id": "call-1", "title": "terminal - ls", "status": "error"},
    {"id": "call-2", "title": "read_file - a.txt", "status": "complete"},
    {"id": "call-3", "title": "web_search - news", "status": "in_progress"},
]


async def _answer(method, *, json):
    return {"ts": "stream-1"} if method == "chat.startStream" else {"ok": True}


async def _sent_frames(adapter, tasks=TASKS, frames=1):
    client = adapter._team_clients["T1"]
    client.api_call.side_effect = _answer
    for _ in range(frames):
        result = await adapter.send_native_task_card_progress("C1", tasks, metadata=METADATA)
        assert result.success
    appends = [call.kwargs["json"] for call in client.api_call.await_args_list if call.args[0] == "chat.appendStream"]
    return [{chunk["id"]: chunk for chunk in append["chunks"] if chunk["type"] == "task_update"} for append in appends]


def _labelled(label):
    adapter = _make_adapter()
    adapter.config.extra["task_card_failure_label"] = label
    return adapter


@pytest.mark.asyncio
async def test_failed_task_names_the_label_in_its_title_and_carries_no_error_status():
    adapter = _labelled("fail")
    [chunks] = await _sent_frames(adapter)
    assert all(chunk["status"] != "error" for chunk in chunks.values())
    failed = chunks["call-1"]
    assert failed["status"] == "complete"
    assert failed["title"].startswith(TASKS[0]["title"])
    assert failed["title"].endswith(adapter.config.extra["task_card_failure_label"])
    assert not {"output", "details"} & set(failed)


@pytest.mark.asyncio
async def test_every_frame_sends_the_same_failed_task_so_nothing_accumulates_on_the_card():
    frames = await _sent_frames(_labelled("fail"), frames=3)
    assert len(frames) == 3
    assert all(frame["call-1"] == frames[0]["call-1"] for frame in frames)


@pytest.mark.asyncio
async def test_tasks_that_did_not_fail_are_unchanged_by_the_switch():
    [plain] = await _sent_frames(_make_adapter())
    [switched] = await _sent_frames(_labelled("fail"))
    for task_id in ("call-2", "call-3"):
        assert switched[task_id] == plain[task_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, ""])
async def test_unset_switch_keeps_upstream_error_status(value):
    adapter = _make_adapter()
    if value is not None:
        adapter.config.extra["task_card_failure_label"] = value
    [chunks] = await _sent_frames(adapter)
    assert chunks["call-1"] == {"type": "task_update", **TASKS[0]}


@pytest.mark.asyncio
async def test_label_survives_a_title_at_slacks_chunk_limit():
    plain_limit = len((await _sent_frames(_make_adapter(), [{"id": "x", "title": "t" * 1000, "status": "error"}]))[0]["x"]["title"])
    adapter = _labelled("fail")
    [chunks] = await _sent_frames(adapter, [{"id": "x", "title": "t" * 1000, "status": "error"}])
    assert len(chunks["x"]["title"]) <= plain_limit
    assert chunks["x"]["title"].endswith(adapter.config.extra["task_card_failure_label"])


@pytest.mark.asyncio
async def test_an_overlong_label_still_fits_slacks_chunk_limit():
    plain_limit = len((await _sent_frames(_make_adapter(), [{"id": "x", "title": "t" * 1000, "status": "error"}]))[0]["x"]["title"])
    [chunks] = await _sent_frames(_labelled("x" * 1000))
    assert 0 < len(chunks["call-1"]["title"]) <= plain_limit
