"""Slack under ``extra.media_in_one_message``: a reply's rich text and its files go out as one message;
``extra.report_dropped_media`` names in the chat the files a reply asked for but could not attach."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import _ExtractedResponse
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.slack.adapter import FINAL_TEXT_FILES, SlackAdapter

CHAT, TEXT_TS, FILES_TS = "D0DM", "111.222", "555.666"
REPLY = "**Done**\n\n| Item | Owner |\n|---|---|\n| Site | Ann |"


def _make(extra=None, *, failing_uploads=0):
    config = PlatformConfig(enabled=True, token="xoxb-fake", extra={
        "rich_blocks": True, "media_in_one_message": True, **(extra or {})})
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": TEXT_TS})
    uploads = []

    async def files_upload_v2(**kwargs):
        uploads.append(kwargs)
        if len(uploads) <= failing_uploads:
            return {"ok": False, "error": "upload_failed"}
        return {"ok": True, "files": [{"id": f"F{len(uploads)}"}]}

    client.files_upload_v2 = AsyncMock(side_effect=files_upload_v2)
    client.files_info = AsyncMock(return_value={"ok": True, "file": {"shares": {"private": {CHAT: [{"ts": FILES_TS}]}}}})
    adapter._get_client = MagicMock(return_value=client)
    adapter.stop_typing = AsyncMock()
    adapter.emit_media_warning = AsyncMock()
    adapter._running = True
    return adapter, client, uploads


@pytest.fixture
def paths(tmp_path):
    made = []
    for name in ("table.xlsx", "notes.csv"):
        path = tmp_path / name
        path.write_bytes(b"data")
        made.append(str(path))
    return made


def _compact(blocks):
    return json.dumps(blocks, ensure_ascii=False, separators=(",", ":"))


@pytest.mark.asyncio
async def test_a_reply_with_files_is_one_message_with_its_rich_text_and_every_file(paths):
    adapter, client, uploads = _make()
    result = await adapter.send(CHAT, REPLY, metadata={FINAL_TEXT_FILES: paths})
    [upload] = uploads
    assert upload["blocks"] == _compact(adapter._maybe_blocks(REPLY)) and "initial_comment" not in upload
    assert [entry["file"] for entry in upload["file_uploads"]] == paths
    client.chat_postMessage.assert_not_awaited()
    assert result.success and result.message_id == FILES_TS


@pytest.mark.asyncio
async def test_rich_text_past_the_cap_goes_on_top_and_its_files_go_into_its_thread(paths):
    adapter, client, uploads = _make({"one_message_max_bytes": 10})
    result = await adapter.send(CHAT, REPLY, metadata={FINAL_TEXT_FILES: paths})
    client.chat_postMessage.assert_awaited_once()
    assert client.chat_postMessage.await_args.kwargs.get("blocks")
    [upload] = uploads
    assert "blocks" not in upload and not upload.get("initial_comment") and upload["thread_ts"] == TEXT_TS
    assert result.success and result.message_id == TEXT_TS


@pytest.mark.asyncio
async def test_inside_a_thread_the_text_and_its_files_stay_in_that_thread(paths):
    adapter, client, uploads = _make({"one_message_max_bytes": 10})
    await adapter.send(CHAT, REPLY, metadata={FINAL_TEXT_FILES: paths, "thread_id": "100.000"})
    assert client.chat_postMessage.await_args.kwargs["thread_ts"] == "100.000"
    assert [upload["thread_ts"] for upload in uploads] == ["100.000"]


@pytest.mark.asyncio
async def test_the_cap_counts_bytes_so_chinese_text_is_measured_as_slack_may_count_it(paths):
    text = "报告" * 20
    adapter, client, uploads = _make({"rich_blocks": False, "one_message_max_bytes": len(text) * 2})
    await adapter.send(CHAT, text, metadata={FINAL_TEXT_FILES: paths})
    client.chat_postMessage.assert_awaited_once()
    assert not uploads[0].get("initial_comment")


@pytest.mark.asyncio
async def test_without_rich_text_the_reply_rides_as_the_upload_comment(paths):
    adapter, client, uploads = _make({"rich_blocks": False})
    await adapter.send(CHAT, REPLY, metadata={FINAL_TEXT_FILES: paths})
    [upload] = uploads
    assert upload["initial_comment"] == adapter.format_message(REPLY) and "blocks" not in upload
    client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_when_the_one_message_upload_fails_the_text_and_the_files_still_go(paths):
    adapter, client, uploads = _make(failing_uploads=1)
    result = await adapter.send(CHAT, REPLY, metadata={FINAL_TEXT_FILES: paths})
    client.chat_postMessage.assert_awaited_once()
    assert len(uploads) == 2 and "blocks" not in uploads[1]
    assert result.success and result.message_id == TEXT_TS
    adapter.emit_media_warning.assert_not_awaited()


@pytest.mark.asyncio
async def test_files_that_never_go_up_are_named_in_the_chat(paths):
    adapter, client, uploads = _make(failing_uploads=2)
    result = await adapter.send(CHAT, REPLY, metadata={FINAL_TEXT_FILES: paths})
    assert result.success and result.message_id == TEXT_TS
    notices = " ".join(call.args[1] for call in adapter.emit_media_warning.await_args_list)
    assert all(name in notices for name in ("table.xlsx", "notes.csv"))


def _extracted(media_files):
    return _ExtractedResponse(text_content="done", images=[], media_files=media_files, local_files=[],
                              force_document_attachments=False, pre_extract="done")


def test_the_reply_s_files_travel_with_its_text_and_voice_notes_stay_apart(paths):
    adapter, _, _ = _make()
    voice = ("/hermes/cache/audio/note.ogg", True)
    metadata, extracted = adapter._attach_files_to_final_text(
        _extracted([(paths[0], False), voice, (paths[1], False)]), {"thread_id": "1.0"}, False)
    assert metadata == {"thread_id": "1.0", FINAL_TEXT_FILES: paths}
    assert extracted.media_files == [voice]


@pytest.mark.parametrize(("extra", "ephemeral"), [({}, True), ({"media_in_one_message": False}, False)])
def test_an_ephemeral_reply_or_the_switch_off_keeps_files_apart(paths, extra, ephemeral):
    adapter, _, _ = _make(extra)
    original = _extracted([(paths[0], False)])
    metadata, extracted = adapter._attach_files_to_final_text(original, {"thread_id": "1.0"}, ephemeral)
    assert metadata == {"thread_id": "1.0"} and extracted is original


def _event():
    source = SessionSource(platform=Platform.SLACK, chat_id=CHAT, chat_type="dm", user_id="U0ANN")
    return MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1.0")


@pytest.mark.asyncio
async def test_a_files_only_reply_is_one_message_too(paths):
    adapter, client, uploads = _make()
    record = MagicMock()
    await adapter._deliver_media_attachments(
        _event(), [(path, False) for path in paths], [], force_document_attachments=False, human_delay=0,
        metadata={}, record_delivery=record)
    [upload] = uploads
    assert [entry["file"] for entry in upload["file_uploads"]] == paths and not upload.get("initial_comment")
    assert record.call_args.args[0].success


@pytest.mark.parametrize("report", [True, False])
@pytest.mark.asyncio
async def test_files_a_reply_asked_for_but_could_not_attach_are_named_only_under_the_switch(report):
    adapter, _, _ = _make({"report_dropped_media": report})
    extracted = _extracted([])
    extracted.dropped_media.append({"path": "/root/out/gone.xlsx", "reason": "not found on this host"})
    await adapter._deliver_attachments(_event(), extracted, {}, anything_sent=True, record_delivery=MagicMock())
    notices = [call.args[1] for call in adapter.emit_media_warning.await_args_list]
    assert (["gone.xlsx" in notice for notice in notices] == [True]) is report
