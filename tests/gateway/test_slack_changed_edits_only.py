"""``platforms.slack.extra.changed_messages_edits_only``: a ``message_changed`` event is handled only when it
carries a new edit — an ``edited`` stamp whose ts differs from the previous version's. Slack also sends it
when a reply raises a thread root's reply count or a link unfurls; with the setting those changes never start
a turn, also in a process that never saw the original (after a restart) and in a DM, where no channel gate
runs. Unset keeps upstream's behaviour."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter

DM, CHANNEL, ROOT_TS, USER, BOT = "D0RYANDM001", "C0TEAM00001", "1792773795.857519", "U0RYAN", "U_BOT"
CHANGE_TS, EARLIER_EDIT_TS, NEW_EDIT_TS = "1792776443.000100", "1792774000.000000", "1792776500.000000"


def make_adapter(extra: dict) -> SlackAdapter:
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test", extra=extra))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(return_value={
        "user": {"is_bot": False, "profile": {"display_name": "Ryan"}, "real_name": "Ryan"}})
    a._app.client.conversations_replies = AsyncMock(return_value={"ok": True, "messages": []})
    a._bot_user_id = BOT
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture
def adapter() -> SlackAdapter:
    return make_adapter({"changed_messages_edits_only": True})


def message(channel: str = DM, channel_type: str = "im", **fields) -> dict:
    return {"user": USER, "channel": channel, "channel_type": channel_type, "ts": ROOT_TS, "thread_ts": ROOT_TS,
            "text": "小红书链接打通 帮我加到 todo", **fields}


def changed(current: dict, previous: dict | None = None) -> dict:
    event = {"type": "message", "subtype": "message_changed", "channel": current["channel"],
             "channel_type": current["channel_type"], "ts": CHANGE_TS, "event_ts": CHANGE_TS, "message": current}
    if previous is not None:
        event["previous_message"] = previous
    return event


def reply_count_change(**fields) -> dict:
    return changed(message(reply_count=2, latest_reply=CHANGE_TS, **fields),
                   message(reply_count=1, latest_reply=EARLIER_EDIT_TS, **fields))


def arrive(adapter: SlackAdapter, event: dict) -> None:
    asyncio.run(adapter._handle_slack_message(event))


def test_a_dm_thread_root_arriving_again_for_its_reply_count_starts_no_turn(adapter):
    arrive(adapter, reply_count_change())
    adapter.handle_message.assert_not_called()


def test_a_mentioned_channel_thread_root_arriving_again_starts_no_turn(adapter):
    adapter._register_mentioned_thread(ROOT_TS)
    arrive(adapter, reply_count_change(channel=CHANNEL, channel_type="channel"))
    adapter.handle_message.assert_not_called()


def test_a_link_unfurl_starts_no_turn(adapter):
    unfurl = {"from_url": "https://example.com/a", "title": "A", "text": "preview"}
    arrive(adapter, changed(message(attachments=[unfurl]), message()))
    adapter.handle_message.assert_not_called()


def test_an_earlier_edit_carried_by_a_later_change_starts_no_turn(adapter):
    earlier_edit = {"user": USER, "ts": EARLIER_EDIT_TS}
    arrive(adapter, reply_count_change(edited=earlier_edit))
    adapter.handle_message.assert_not_called()


@pytest.mark.parametrize("edited", ["yes", {"user": USER}, {"user": USER, "ts": ""}, None])
def test_a_malformed_edit_stamp_is_no_edit(adapter, edited):
    arrive(adapter, changed(message(edited=edited), message()))
    adapter.handle_message.assert_not_called()


def test_a_new_edit_of_a_message_this_process_never_routed_starts_a_turn(adapter):
    arrive(adapter, changed(message(text="小红书链接打通(更正)", edited={"user": USER, "ts": NEW_EDIT_TS}),
                            message(edited={"user": USER, "ts": EARLIER_EDIT_TS})))
    adapter.handle_message.assert_called_once()


def test_an_edit_that_adds_a_mention_wakes_the_bot_once(adapter):
    mention = message(channel=CHANNEL, channel_type="channel", ts="1792776400.000300", thread_ts=None, text="帮我看看")
    arrive(adapter, mention)
    adapter.handle_message.assert_not_called()
    edited = {**mention, "text": f"<@{BOT}> 帮我看看", "edited": {"user": USER, "ts": NEW_EDIT_TS}}
    arrive(adapter, changed(edited))
    adapter.handle_message.assert_called_once()


def test_without_the_setting_a_reply_count_change_starts_a_turn_as_upstream_does():
    upstream = make_adapter({})
    arrive(upstream, reply_count_change())
    upstream.handle_message.assert_called_once()
