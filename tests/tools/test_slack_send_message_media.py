"""Slack media delivery for send_message.

Covers ``plugins/platforms/slack/adapter.py::_standalone_send`` media path:
text+file, media-only, caption-on-upload, missing-file warnings.

``slack_sdk`` is optional in CI, so tests inject a fake module into
``sys.modules`` (same pattern as ``tests/gateway/test_slack.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.slack.adapter import _standalone_send


def _pconfig(token: str = "xoxb-test"):
    return SimpleNamespace(token=token, extra={})


def _tmpfile(suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"%PDF-1.4 test")
    f.close()
    return f.name


def _mock_client(*, post_ok=True, upload_ok=True):
    client = MagicMock()
    client.chat_postMessage = AsyncMock(
        return_value={
            "ok": post_ok,
            "ts": "111.222",
            "error": None if post_ok else "channel_not_found",
        }
    )
    if upload_ok:
        client.files_upload_v2 = AsyncMock(
            return_value={
                "ok": True,
                "file": {
                    "id": "F123",
                    "timestamp": 1234567890,
                    "shares": {"public": {"C012AB3CD": [{"ts": "333.444"}]}},
                },
            }
        )
    else:
        client.files_upload_v2 = AsyncMock(
            return_value={"ok": False, "error": "not_in_channel"}
        )
    return client


@contextlib.contextmanager
def _fake_slack_sdk(client):
    """Make ``from slack_sdk.web.async_client import AsyncWebClient`` resolve to a factory."""
    sdk = ModuleType("slack_sdk")
    web = ModuleType("slack_sdk.web")
    async_client = ModuleType("slack_sdk.web.async_client")
    async_client.AsyncWebClient = MagicMock(return_value=client)
    sdk.web = web
    web.async_client = async_client

    modules = {
        "slack_sdk": sdk,
        "slack_sdk.web": web,
        "slack_sdk.web.async_client": async_client,
    }
    old = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, prev in old.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def test_text_plus_pdf_uploads_via_files_upload_v2():
    pdf = _tmpfile(".pdf")
    client = _mock_client()
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "C012AB3CD",
                    "Here is the report",
                    media_files=[(pdf, False)],
                )
            )
        assert result["success"] is True
        assert result["platform"] == "slack"
        client.chat_postMessage.assert_awaited_once()
        client.files_upload_v2.assert_awaited_once()
        upload_kwargs = client.files_upload_v2.await_args.kwargs
        assert upload_kwargs["channel"] == "C012AB3CD"
        assert upload_kwargs["file"] == pdf
        assert upload_kwargs["filename"] == os.path.basename(pdf)
        assert upload_kwargs["initial_comment"] == ""
    finally:
        os.unlink(pdf)


def test_media_only_skips_text_post():
    pdf = _tmpfile(".pdf")
    client = _mock_client()
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(
                _standalone_send(
                    _pconfig(),
                    "C012AB3CD",
                    "",
                    media_files=[(pdf, False)],
                )
            )
        assert result["success"] is True
        client.chat_postMessage.assert_not_awaited()
        client.files_upload_v2.assert_awaited_once()
    finally:
        os.unlink(pdf)


def test_send_to_platform_routes_slack_media():
    """_send_to_platform must call Slack standalone_sender with media_files."""
    import httpx

    if not hasattr(httpx, "Proxy") or not hasattr(httpx, "URL"):
        pytest.skip("httpx type annotations incompatible with telegram library")

    from gateway.config import Platform
    from hermes_cli.plugins import discover_plugins
    from gateway.platform_registry import platform_registry
    from tools.send_message_tool import _send_to_platform

    pdf = _tmpfile(".pdf")
    discover_plugins()
    entry = platform_registry.get("slack")
    assert entry is not None and entry.standalone_sender_fn is not None
    original = entry.standalone_sender_fn
    mock_sender = AsyncMock(
        return_value={"success": True, "platform": "slack", "message_id": "1.2"}
    )
    entry.standalone_sender_fn = mock_sender
    try:
        result = asyncio.run(
            _send_to_platform(
                Platform.SLACK,
                _pconfig(),
                "C012AB3CD",
                "Here is the report",
                media_files=[(pdf, False)],
            )
        )
        assert result["success"] is True
        mock_sender.assert_awaited()
        call_kwargs = mock_sender.await_args.kwargs
        assert call_kwargs.get("media_files") == [(pdf, False)]
        # Single captionable file + short text → caption rides the upload.
        assert call_kwargs.get("caption") == "Here is the report"
        assert not result.get("warnings")
    finally:
        entry.standalone_sender_fn = original
        os.unlink(pdf)


ONE_MESSAGE_TS = "555.666"


def _one_message_pconfig():
    return SimpleNamespace(token="xoxb-test", extra={"media_in_one_message": True})


def _one_message_client(chat_id: str, *, upload_ok: bool = True):
    client = _mock_client()
    uploaded = []

    async def files_upload_v2(**kwargs):
        uploaded.append(kwargs)
        if not upload_ok:
            return {"ok": False, "error": "invalid_arguments"}
        return {"ok": True, "files": [{"id": f"F{len(uploaded)}{index}"} for index, _ in
                                      enumerate(kwargs.get("file_uploads") or [kwargs])]}

    async def files_info(file):
        ts = ONE_MESSAGE_TS if file.startswith("F1") else "later.batch"
        return {"ok": True, "file": {"id": file, "shares": {"private": {chat_id: [{"ts": ts}]}}}}

    client.files_upload_v2 = AsyncMock(side_effect=files_upload_v2)
    client.files_info = AsyncMock(side_effect=files_info)
    return client, uploaded


def _send_one_message(client, chat_id, text, paths, **kwargs):
    with _fake_slack_sdk(client):
        return asyncio.run(_standalone_send(_one_message_pconfig(), chat_id, text,
                                            media_files=[(path, False) for path in paths], **kwargs))


def test_one_message_carries_the_text_and_every_file_and_answers_with_its_ts():
    paths = [_tmpfile(suffix) for suffix in (".xlsx", ".csv", ".pdf")]
    client, uploaded = _one_message_client("D0DM")
    try:
        result = _send_one_message(client, "D0DM", "Here are the files", paths)
        assert result["success"] is True and result["message_id"] == ONE_MESSAGE_TS
        client.chat_postMessage.assert_not_awaited()
        [upload] = uploaded
        assert upload["channel"] == "D0DM" and upload["initial_comment"] == "Here are the files"
        assert [entry["file"] for entry in upload["file_uploads"]] == paths
        assert [entry["filename"] for entry in upload["file_uploads"]] == [os.path.basename(p) for p in paths]
    finally:
        for path in paths:
            os.unlink(path)


def test_one_message_keeps_the_thread_it_is_sent_into():
    path = _tmpfile(".pdf")
    client, uploaded = _one_message_client("C0TEAM")
    try:
        _send_one_message(client, "C0TEAM", "report", [path], thread_id="111.000")
        assert uploaded[0]["thread_ts"] == "111.000"
    finally:
        os.unlink(path)


def test_more_files_than_one_upload_takes_continue_in_further_messages_without_the_text():
    from plugins.platforms.slack.adapter import _FILES_PER_UPLOAD

    paths = [_tmpfile(".csv") for _ in range(_FILES_PER_UPLOAD + 2)]
    client, uploaded = _one_message_client("D0DM")
    try:
        result = _send_one_message(client, "D0DM", "many", paths)
        assert [len(upload["file_uploads"]) for upload in uploaded] == [_FILES_PER_UPLOAD, 2]
        assert [upload["initial_comment"] for upload in uploaded] == ["many", ""]
        assert result["message_id"] == ONE_MESSAGE_TS
        assert "thread_ts" not in uploaded[0] and uploaded[1]["thread_ts"] == ONE_MESSAGE_TS
    finally:
        for path in paths:
            os.unlink(path)


def test_a_missing_file_is_skipped_with_a_warning_and_the_rest_still_go_as_one_message():
    path = _tmpfile(".pdf")
    missing = path + ".gone"
    client, uploaded = _one_message_client("D0DM")
    try:
        result = _send_one_message(client, "D0DM", "report", [path, missing])
        assert [entry["file"] for entry in uploaded[0]["file_uploads"]] == [path]
        assert any(missing in warning for warning in result["warnings"])
    finally:
        os.unlink(path)


def test_when_the_one_message_upload_fails_the_text_posts_on_top_and_its_files_try_its_thread():
    path = _tmpfile(".pdf")
    client, uploaded = _one_message_client("D0DM", upload_ok=False)
    try:
        result = _send_one_message(client, "D0DM", "report", [path])
        client.chat_postMessage.assert_awaited_once()
        assert len(uploaded) == 2 and uploaded[1]["thread_ts"] == "111.222"
        assert result["success"] is True and result["message_id"] == "111.222" and result["warnings"]
    finally:
        os.unlink(path)


def test_without_the_switch_text_and_files_stay_separate_messages():
    paths = [_tmpfile(".pdf"), _tmpfile(".csv")]
    client, uploaded = _one_message_client("D0DM")
    try:
        with _fake_slack_sdk(client):
            asyncio.run(_standalone_send(_pconfig(), "D0DM", "report", media_files=[(p, False) for p in paths]))
        client.chat_postMessage.assert_awaited_once()
        assert len(uploaded) == len(paths) and all("file_uploads" not in upload for upload in uploaded)
    finally:
        for path in paths:
            os.unlink(path)


def test_a_share_slack_makes_a_moment_after_the_upload_is_waited_for(monkeypatch):
    from plugins.platforms.slack import adapter

    monkeypatch.setattr(adapter, "_SHARE_POLL_SECONDS", 0)
    path = _tmpfile(".pdf")
    client, _ = _one_message_client("D0DM")
    answers = [{"ok": True, "file": {"id": "F10"}}, {"ok": True, "file": {"id": "F10", "shares": {}}},
               {"ok": True, "file": {"id": "F10", "shares": {"private": {"D0DM": [{"ts": ONE_MESSAGE_TS}]}}}}]
    client.files_info = AsyncMock(side_effect=answers)
    try:
        assert _send_one_message(client, "D0DM", "report", [path])["message_id"] == ONE_MESSAGE_TS
        assert client.files_info.await_count == len(answers)
    finally:
        os.unlink(path)


def test_a_share_that_never_appears_leaves_the_send_standing_without_a_ts(monkeypatch):
    from plugins.platforms.slack import adapter

    monkeypatch.setattr(adapter, "_SHARE_POLL_SECONDS", 0)
    path = _tmpfile(".pdf")
    client, _ = _one_message_client("D0DM")
    client.files_info = AsyncMock(return_value={"ok": True, "file": {"id": "F10"}})
    try:
        result = _send_one_message(client, "D0DM", "report", [path])
        assert result["success"] is True and result["message_id"] is None
        assert client.files_info.await_count == adapter._SHARE_POLL_ATTEMPTS
    finally:
        os.unlink(path)


def test_a_send_from_outside_the_gateway_carries_its_rich_text_with_the_files():
    import json

    paths = [_tmpfile(".xlsx"), _tmpfile(".csv")]
    client, uploaded = _one_message_client("D0DM")
    pconfig = SimpleNamespace(token="xoxb-test", extra={"media_in_one_message": True, "rich_blocks": True})
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(_standalone_send(pconfig, "D0DM", "**Done**\n\n- table\n- notes",
                                                  media_files=[(path, False) for path in paths]))
        [upload] = uploaded
        assert json.loads(upload["blocks"]) and "initial_comment" not in upload
        assert [entry["file"] for entry in upload["file_uploads"]] == paths
        assert result["message_id"] == ONE_MESSAGE_TS
    finally:
        for path in paths:
            os.unlink(path)


def test_files_past_one_message_stay_in_the_thread_the_send_was_made_in():
    from plugins.platforms.slack.adapter import _FILES_PER_UPLOAD

    paths = [_tmpfile(".csv") for _ in range(_FILES_PER_UPLOAD + 1)]
    client, uploaded = _one_message_client("C0TEAM")
    try:
        _send_one_message(client, "C0TEAM", "many", paths, thread_id="111.000")
        assert [upload["thread_ts"] for upload in uploaded] == ["111.000", "111.000"]
    finally:
        for path in paths:
            os.unlink(path)


def test_a_text_too_long_to_ride_along_posts_on_top_and_its_files_go_into_its_thread():
    paths = [_tmpfile(".csv")]
    client, uploaded = _one_message_client("C0TEAM")
    pconfig = SimpleNamespace(token="xoxb-test", extra={"media_in_one_message": True, "rich_blocks": True,
                                                        "one_message_max_bytes": 10})
    try:
        with _fake_slack_sdk(client):
            result = asyncio.run(_standalone_send(pconfig, "C0TEAM", "**Done**\n\n- a long report",
                                                  media_files=[(path, False) for path in paths]))
        client.chat_postMessage.assert_awaited_once()
        assert "thread_ts" not in client.chat_postMessage.await_args.kwargs
        [upload] = uploaded
        assert upload["thread_ts"] == "111.222" and result["message_id"] == "111.222"
    finally:
        for path in paths:
            os.unlink(path)
