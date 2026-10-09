"""A thread's root message that never addressed the bot starts no turn when it arrives again after the
bot was @-mentioned in that thread (Slack sends ``message_changed`` for the root when a reply raises its
reply count); replies in that thread still wake the bot, and so does an edit that adds a mention."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter

CHANNEL, ROOT_TS, USER, BOT = "C0TEAM00001", "1791498504.084569", "U0RYAN", "U_BOT"


@pytest.fixture
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(return_value={
        "user": {"is_bot": False, "profile": {"display_name": "Ryan"}, "real_name": "Ryan"}})
    a._app.client.conversations_replies = AsyncMock(return_value={"ok": True, "messages": []})
    a._bot_user_id = BOT
    a._running = True
    a.handle_message = AsyncMock()
    a._register_mentioned_thread(ROOT_TS)
    return a


def root(**fields) -> dict:
    return {"user": USER, "channel": CHANNEL, "channel_type": "channel", "ts": ROOT_TS, "thread_ts": ROOT_TS,
            "text": "这是本周数据", "reply_count": 1, **fields}


def changed(message: dict, event_ts: str = "1791498525.000100") -> dict:
    return {"type": "message", "subtype": "message_changed", "channel": CHANNEL, "channel_type": "channel",
            "ts": event_ts, "event_ts": event_ts, "message": message}


def arrive(adapter, event: dict) -> None:
    asyncio.run(adapter._handle_slack_message(event))


def test_the_root_arriving_again_after_a_mention_in_its_thread_starts_no_turn(adapter):
    arrive(adapter, changed(root()))
    adapter.handle_message.assert_not_called()


def test_the_root_edited_without_a_mention_starts_no_turn(adapter):
    arrive(adapter, changed(root(text="这是本周数据(更正)", edited={"user": USER, "ts": "1791498600.000000"})))
    adapter.handle_message.assert_not_called()


def test_an_unmentioned_reply_in_that_thread_still_wakes_the_bot(adapter):
    arrive(adapter, {"user": USER, "channel": CHANNEL, "channel_type": "channel", "ts": "1791498530.000200",
                     "thread_ts": ROOT_TS, "text": "再看一下第二行"})
    adapter.handle_message.assert_called_once()


def test_an_edit_that_adds_a_mention_to_the_root_still_wakes_the_bot(adapter):
    arrive(adapter, changed(root(text=f"<@{BOT}> 这是本周数据", edited={"user": USER, "ts": "1791498600.000000"})))
    adapter.handle_message.assert_called_once()


def test_a_new_top_level_mention_still_wakes_the_bot(adapter):
    arrive(adapter, {"user": USER, "channel": CHANNEL, "channel_type": "channel", "ts": "1791498700.000300",
                     "text": f"<@{BOT}> 帮我看看"})
    adapter.handle_message.assert_called_once()
