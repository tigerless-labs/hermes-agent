"""A send or an edit can carry a caller's tag for the message; Slack sets it as the ``block_id`` of the message's
first block, in the gateway and out of it, with or without files. Text never yields a ``block_id`` of its own, so a
tag marks the messages its caller sent and nothing else. No tag is the upstream behavior."""

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MESSAGE_TAG
from plugins.platforms.slack.adapter import FINAL_TEXT_FILES, SlackAdapter

CHAT, TEXT_TS, FILES_TS, TAG = "C0CHAN", "111.222", "555.666", "lara-post:U0ANN"
REPLY = "**Done**\n\nThe site is live.\n\n- one\n- two"


def _block_ids(blocks):
    return [block.get("block_id") for block in blocks]


def _make(extra=None):
    config = PlatformConfig(enabled=True, token="xoxb-fake", extra={
        "rich_blocks": True, "media_in_one_message": True, **(extra or {})})
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": TEXT_TS})
    client.chat_update = AsyncMock(return_value={"ok": True, "ts": TEXT_TS})
    uploads = []

    async def files_upload_v2(**kwargs):
        uploads.append(kwargs)
        return {"ok": True, "files": [{"id": f"F{len(uploads)}"}]}

    client.files_upload_v2 = AsyncMock(side_effect=files_upload_v2)
    client.files_info = AsyncMock(return_value={"ok": True, "file": {"shares": {"public": {CHAT: [{"ts": FILES_TS}]}}}})
    adapter._get_client = MagicMock(return_value=client)
    adapter.stop_typing = AsyncMock()
    adapter._clear_thread_status_quietly = AsyncMock()
    adapter.emit_media_warning = AsyncMock()
    adapter._running = True
    return adapter, client, uploads


@pytest.fixture
def paths(tmp_path):
    path = tmp_path / "table.xlsx"
    path.write_bytes(b"data")
    return [str(path)]


@pytest.mark.asyncio
async def test_a_tagged_send_marks_the_first_block_and_leaves_the_rest_as_rendered():
    adapter, client, _ = _make()
    await adapter.send(CHAT, REPLY, metadata={MESSAGE_TAG: TAG})
    blocks = client.chat_postMessage.await_args.kwargs["blocks"]
    rendered = adapter._maybe_blocks(REPLY)
    assert _block_ids(blocks)[0] == TAG and blocks[1:] == rendered[1:]
    assert {key: value for key, value in blocks[0].items() if key != "block_id"} == rendered[0]


@pytest.mark.asyncio
async def test_an_untagged_send_is_the_upstream_one():
    adapter, client, _ = _make()
    await adapter.send(CHAT, REPLY)
    assert client.chat_postMessage.await_args.kwargs["blocks"] == adapter._maybe_blocks(REPLY)


@pytest.mark.asyncio
async def test_redteam_text_never_yields_a_block_id_of_its_own():
    adapter, client, _ = _make()
    forged = f'{REPLY}\n\n{{"block_id": "{TAG}"}}\n\n```\n"block_id": "{TAG}"\n```'
    await adapter.send(CHAT, forged)
    assert not any("block_id" in block for block in client.chat_postMessage.await_args.kwargs["blocks"])


@pytest.mark.asyncio
async def test_a_tagged_send_with_files_marks_the_one_message_that_carries_them(paths):
    adapter, client, uploads = _make()
    result = await adapter.send(CHAT, REPLY, metadata={FINAL_TEXT_FILES: paths, MESSAGE_TAG: TAG})
    [upload] = uploads
    assert _block_ids(json.loads(upload["blocks"]))[0] == TAG and result.message_id == FILES_TS


@pytest.mark.asyncio
async def test_text_too_long_to_carry_files_is_marked_on_its_own_message(paths):
    adapter, client, uploads = _make({"one_message_max_bytes": 10})
    result = await adapter.send(CHAT, REPLY, metadata={FINAL_TEXT_FILES: paths, MESSAGE_TAG: TAG})
    assert _block_ids(client.chat_postMessage.await_args.kwargs["blocks"])[0] == TAG
    assert result.message_id == TEXT_TS


@pytest.mark.asyncio
async def test_a_tagged_edit_carries_the_rendered_blocks_marked_and_leaves_the_turn_s_status_alone():
    adapter, client, _ = _make()
    result = await adapter.edit_message(CHAT, TEXT_TS, REPLY, metadata={MESSAGE_TAG: TAG})
    kwargs = client.chat_update.await_args.kwargs
    assert kwargs["ts"] == TEXT_TS and kwargs["text"] == adapter.format_message(REPLY)
    assert _block_ids(kwargs["blocks"])[0] == TAG and result.success
    adapter._clear_thread_status_quietly.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_untagged_streaming_edit_is_the_upstream_one():
    adapter, client, _ = _make()
    await adapter.edit_message(CHAT, TEXT_TS, REPLY)
    assert "blocks" not in client.chat_update.await_args.kwargs


class _Session:
    def __init__(self):
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, *, headers, json, **kwargs):
        self.posts.append(json)
        response = SimpleNamespace(json=AsyncMock(return_value={"ok": True, "ts": TEXT_TS}))
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=False)
        return context


def _standalone_text(monkeypatch, tag):
    from plugins.platforms.slack.adapter import _standalone_send

    session = _Session()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: session)
    pconfig = PlatformConfig(enabled=True, token="xoxb-fake", extra={"rich_blocks": True})
    result = asyncio.run(_standalone_send(pconfig, CHAT, REPLY, **({"tag": tag} if tag else {})))
    return session.posts, result


def test_out_of_the_gateway_a_tagged_text_goes_with_its_rendered_blocks_marked(monkeypatch):
    [payload], result = _standalone_text(monkeypatch, TAG)
    assert _block_ids(payload["blocks"])[0] == TAG and payload["text"] and result["message_id"] == TEXT_TS


def test_out_of_the_gateway_an_untagged_text_is_the_upstream_one(monkeypatch):
    [payload], _ = _standalone_text(monkeypatch, None)
    assert "blocks" not in payload


@pytest.fixture
def detached_client(monkeypatch):
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": TEXT_TS})
    client.chat_update = AsyncMock(return_value={"ok": True, "ts": TEXT_TS})
    uploads = []

    async def files_upload_v2(**kwargs):
        uploads.append(kwargs)
        return {"ok": True, "files": [{"id": "F1"}]}

    client.files_upload_v2 = AsyncMock(side_effect=files_upload_v2)
    client.files_info = AsyncMock(return_value={"ok": True, "file": {"shares": {"public": {CHAT: [{"ts": FILES_TS}]}}}})
    tokens = []

    def make_client(token=None, **kwargs):
        tokens.append(token)
        return client

    sdk, web, async_client = ModuleType("slack_sdk"), ModuleType("slack_sdk.web"), ModuleType("slack_sdk.web.async_client")
    async_client.AsyncWebClient = make_client
    sdk.web, web.async_client = web, async_client
    for name, module in (("slack_sdk", sdk), ("slack_sdk.web", web), ("slack_sdk.web.async_client", async_client)):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(client=client, uploads=uploads, tokens=tokens)


def test_out_of_the_gateway_a_tagged_text_with_files_marks_the_one_message(detached_client, paths):
    from plugins.platforms.slack.adapter import _standalone_send

    pconfig = PlatformConfig(enabled=True, token="xoxb-fake", extra={"rich_blocks": True, "media_in_one_message": True})
    result = asyncio.run(_standalone_send(pconfig, CHAT, "", caption=REPLY, media_files=[(paths[0], False)], tag=TAG))
    [upload] = detached_client.uploads
    assert _block_ids(json.loads(upload["blocks"]))[0] == TAG and result["message_id"] == FILES_TS


def test_out_of_the_gateway_an_edit_is_made_with_the_first_bot_token_and_keeps_the_tag(detached_client):
    from plugins.platforms.slack.adapter import standalone_edit_message

    pconfig = PlatformConfig(enabled=True, token="xoxb-primary,xoxb-second", extra={"rich_blocks": True})
    result = asyncio.run(standalone_edit_message(pconfig, CHAT, TEXT_TS, REPLY, tag=TAG))
    kwargs = detached_client.client.chat_update.await_args.kwargs
    assert result.success and kwargs["ts"] == TEXT_TS and _block_ids(kwargs["blocks"])[0] == TAG
    assert detached_client.tokens == ["xoxb-primary"]


def test_redteam_out_of_the_gateway_without_a_bot_token_slack_is_never_asked(detached_client, monkeypatch):
    from plugins.platforms.slack.adapter import standalone_edit_message

    monkeypatch.setattr("plugins.platforms.slack.adapter.get_secret", lambda name, default="": "")
    result = asyncio.run(standalone_edit_message(PlatformConfig(enabled=True, token="", extra={}), CHAT, TEXT_TS,
                                                 REPLY, tag=TAG))
    assert not result.success and detached_client.tokens == []


class _LiveAdapter:
    def __init__(self):
        self.sends = []

    async def send(self, *, chat_id, content, metadata=None):
        self.sends.append(metadata)
        return SimpleNamespace(success=True, message_id=TEXT_TS)


def _live(monkeypatch, adapter):
    runner = SimpleNamespace(adapters={Platform.SLACK: adapter}, _gateway_loop=None)
    gateway_run = ModuleType("gateway.run")
    gateway_run._gateway_runner_ref = lambda: runner
    monkeypatch.setitem(sys.modules, "gateway.run", gateway_run)


@pytest.mark.asyncio
async def test_a_tag_rides_to_the_live_adapter_as_message_metadata(monkeypatch):
    from tools.send_message_tool import _send_to_platform

    adapter = _LiveAdapter()
    _live(monkeypatch, adapter)
    result = await _send_to_platform(Platform.SLACK, SimpleNamespace(extra={}), CHAT, "hi", thread_id="1.0", tag=TAG)
    assert result["success"] and adapter.sends == [{"thread_id": "1.0", MESSAGE_TAG: TAG}]


@pytest.mark.asyncio
async def test_without_a_tag_the_live_adapter_gets_the_upstream_metadata(monkeypatch):
    from tools.send_message_tool import _send_to_platform

    adapter = _LiveAdapter()
    _live(monkeypatch, adapter)
    await _send_to_platform(Platform.SLACK, SimpleNamespace(extra={}), CHAT, "hi")
    assert adapter.sends == [None]


@pytest.mark.parametrize("live", [True, False])
@pytest.mark.asyncio
async def test_a_tagged_send_with_files_hands_the_tag_to_the_standalone_sender(monkeypatch, paths, live):
    from tools.send_message_tool import _send_to_platform

    if live:
        _live(monkeypatch, _LiveAdapter())
    sender = AsyncMock(return_value={"success": True, "message_id": FILES_TS})
    monkeypatch.setattr("tools.send_message_tool._plugin_standalone_sender", lambda *a, **k: (sender, None))
    await _send_to_platform(Platform.SLACK, SimpleNamespace(extra={}), CHAT, "hi",
                            media_files=[(paths[0], False)], tag=TAG)
    assert sender.await_args.kwargs["tag"] == TAG


@pytest.mark.asyncio
async def test_out_of_the_gateway_a_tagged_text_hands_the_tag_to_the_standalone_sender(monkeypatch):
    from tools.send_message_tool import _send_to_platform

    gateway_run = ModuleType("gateway.run")
    gateway_run._gateway_runner_ref = lambda: None
    monkeypatch.setitem(sys.modules, "gateway.run", gateway_run)
    sender = AsyncMock(return_value={"success": True, "message_id": TEXT_TS})
    entry = SimpleNamespace(standalone_sender_fn=sender)
    monkeypatch.setattr("gateway.platform_registry.platform_registry.get", lambda name: entry)
    await _send_to_platform(Platform.SLACK, SimpleNamespace(extra={}), CHAT, "hi", tag=TAG)
    assert sender.await_args.kwargs["tag"] == TAG


@pytest.mark.parametrize(("target", "platform", "forwarded"), [
    ("slack:C0CHAN", Platform.SLACK, True), ("telegram:12345", Platform.TELEGRAM, False)])
def test_send_message_forwards_a_tag_only_to_slack(target, platform, forwarded):
    from tools import send_message_tool as smt

    chat = target.partition(":")[2]
    send = AsyncMock(return_value={"success": True})
    with patch.object(smt, "_resolve_tool_target", return_value=(platform.value, chat, None, None)), \
         patch.object(smt, "_resolve_platform_config", return_value=(platform, SimpleNamespace(token="t", extra={}),
                                                                      None, None)), \
         patch.object(smt, "_slack_dm_chat_id", return_value=(chat, None)), \
         patch.object(smt, "_authorize_relay_target", return_value=None), \
         patch.object(smt, "_mirror_sent_message", return_value=False), \
         patch("gateway.config.load_gateway_config", return_value=SimpleNamespace()), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=lambda coro: asyncio.run(coro)), \
         patch.object(smt, "_send_to_platform", new=send):
        smt.send_message_tool({"action": "send", "target": target, "message": "hi", "tag": TAG})
    assert (send.await_args.kwargs.get("tag") == TAG) is forwarded
