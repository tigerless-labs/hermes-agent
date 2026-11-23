"""A background wake names the message that started its work, as a per-turn session value.

The wake's own ``message_id`` stays cleared (no stale reply anchor, #52694). The message that
started the work rides on the wake's source (``SessionSource.origin_message_id``, wire-invisible)
and every turn binds it as ``HERMES_SESSION_ORIGIN_MESSAGE_ID`` for plugins: a plugin that must
prove to an outside service whose work this wake carries on can name that message.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionContext, SessionSource
from gateway.session_context import get_session_env
from tests.gateway.test_background_process_notifications import _build_runner

KEY = "agent:main:telegram:dm:123:24296"
ORIGIN = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", thread_id="24296", user_id="1",
                       message_id="777")


async def _wake(monkeypatch, tmp_path, evt: dict):
    runner = _build_runner(monkeypatch, tmp_path, "all")
    runner.session_store._entries[KEY] = SimpleNamespace(origin=ORIGIN)
    await runner._inject_watch_notification("[SYSTEM: Background process matched]", {"session_key": KEY, **evt})
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.handle_message.assert_awaited_once()
    return runner, adapter.handle_message.await_args.args[0]


@pytest.mark.asyncio
async def test_a_wake_carries_the_message_that_started_its_work(monkeypatch, tmp_path) -> None:
    runner, wake = await _wake(monkeypatch, tmp_path, {"session_id": "proc_watch", "message_id": "777"})
    assert wake.source.message_id is None and wake.message_id is None
    assert wake.source.origin_message_id == "777"
    assert "origin_message_id" not in wake.source.to_dict()
    tokens = runner._set_session_env(SessionContext(source=wake.source, connected_platforms=[], home_channels={}))
    try:
        assert get_session_env("HERMES_SESSION_ORIGIN_MESSAGE_ID") == "777"
        assert get_session_env("HERMES_SESSION_MESSAGE_ID") == ""
    finally:
        runner._clear_session_env(tokens)
    assert get_session_env("HERMES_SESSION_ORIGIN_MESSAGE_ID") == ""


@pytest.mark.asyncio
async def test_a_wake_whose_work_named_no_message_names_none(monkeypatch, tmp_path) -> None:
    runner, wake = await _wake(monkeypatch, tmp_path, {"session_id": "proc_watch"})
    assert wake.source.origin_message_id is None
    tokens = runner._set_session_env(SessionContext(source=wake.source, connected_platforms=[], home_channels={}))
    try:
        assert get_session_env("HERMES_SESSION_ORIGIN_MESSAGE_ID") == ""
    finally:
        runner._clear_session_env(tokens)


def test_redteam_a_peer_cannot_set_the_origin_over_the_wire(tmp_path, monkeypatch) -> None:
    carried = dataclasses.replace(ORIGIN, origin_message_id="777")
    sent = {**carried.to_dict(), "origin_message_id": "999"}
    assert SessionSource.from_dict(sent).origin_message_id is None


def test_an_ordinary_turn_binds_no_origin(tmp_path, monkeypatch) -> None:
    runner = _build_runner(monkeypatch, tmp_path, "all")
    tokens = runner._set_session_env(SessionContext(source=ORIGIN, connected_platforms=[], home_channels={}))
    try:
        assert get_session_env("HERMES_SESSION_MESSAGE_ID") == "777"
        assert get_session_env("HERMES_SESSION_ORIGIN_MESSAGE_ID") == ""
    finally:
        runner._clear_session_env(tokens)
