"""A failed task on the Slack-native card follows ``extra.task_card_failure_output``."""

import pytest

from tests.gateway.test_slack_approval_buttons import _make_adapter

METADATA = {"thread_id": "thread-1", "slack_team_id": "T1"}
TASKS = [
    {"id": "call-1", "title": "terminal - ls", "status": "error"},
    {"id": "call-2", "title": "read_file - a.txt", "status": "complete"},
    {"id": "call-3", "title": "web_search - news", "status": "in_progress"},
]


async def _sent_task_chunks(adapter):
    client = adapter._team_clients["T1"]
    client.api_call.side_effect = _answer
    result = await adapter.send_native_task_card_progress("C1", TASKS, metadata=METADATA)
    assert result.success
    appends = [call.kwargs["json"] for call in client.api_call.await_args_list if call.args[0] == "chat.appendStream"]
    return {chunk["id"]: chunk for chunk in appends[-1]["chunks"] if chunk["type"] == "task_update"}


async def _answer(method, *, json):
    return {"ts": "stream-1"} if method == "chat.startStream" else {"ok": True}


@pytest.mark.asyncio
async def test_failed_task_carries_the_configured_output_and_no_error_status():
    adapter = _make_adapter()
    adapter.config.extra["task_card_failure_output"] = "fail"
    chunks = await _sent_task_chunks(adapter)
    assert all(chunk["status"] != "error" for chunk in chunks.values())
    assert chunks["call-1"]["status"] == "complete"
    assert chunks["call-1"]["output"] == adapter.config.extra["task_card_failure_output"]
    assert chunks["call-1"]["title"] == TASKS[0]["title"]


@pytest.mark.asyncio
async def test_tasks_that_did_not_fail_are_unchanged_by_the_switch():
    plain, switched = _make_adapter(), _make_adapter()
    switched.config.extra["task_card_failure_output"] = "fail"
    plain_chunks, switched_chunks = await _sent_task_chunks(plain), await _sent_task_chunks(switched)
    for task_id in ("call-2", "call-3"):
        assert switched_chunks[task_id] == plain_chunks[task_id]
        assert "output" not in switched_chunks[task_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, ""])
async def test_unset_switch_keeps_upstream_error_status(value):
    adapter = _make_adapter()
    if value is not None:
        adapter.config.extra["task_card_failure_output"] = value
    chunks = await _sent_task_chunks(adapter)
    assert chunks["call-1"]["status"] == "error"
    assert "output" not in chunks["call-1"]


@pytest.mark.asyncio
async def test_failure_output_respects_slacks_chunk_limit():
    adapter = _make_adapter()
    adapter.config.extra["task_card_failure_output"] = "x" * 1000
    chunks = await _sent_task_chunks(adapter)
    assert 0 < len(chunks["call-1"]["output"]) <= 256
