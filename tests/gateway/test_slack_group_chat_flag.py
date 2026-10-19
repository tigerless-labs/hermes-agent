"""``platforms.slack.extra.group_chat_flag``: each admitted message carries the adapter's verdict on
whether its conversation is a group chat, as ``source.group_chat``, for plugins to tell the
model. A 1:1 DM never is; a group DM always is. In a channel thread it is when someone besides the
sender and the bot has spoken in or been @mentioned in the thread (other bots do not count), the same
people ``thread_reply_judgement`` counts; a top-level message counts only whom it @mentions; a thread
that cannot be read counts as a group chat. Off, no verdict. Driven through the real
``_handle_slack_message``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway.test_slack_ignore_other_user_mentions import (  # noqa: F401 - fixtures
    BOT_USER_ID, CHANNEL_ID, OTHER_USER_ID, _redirect_cache, adapter,
)

THREAD = "1700000000.000010"
SENDER = "U_HUMAN"
OTHER_BOT = "U_BOT_OTHER"
ROOT = {"ts": THREAD, "user": SENDER, "text": f"<@{BOT_USER_ID}> pull last week's numbers"}
BOT_REPLY = {"ts": "1700000000.000020", "user": BOT_USER_ID, "bot_id": "B_SELF", "text": "Done."}


def _said(ts, text, user=SENDER, **extra):
    return {"ts": ts, "user": user, "text": text, **extra}


def _message(text, ts, *, user=SENDER, channel=CHANNEL_ID, channel_type="channel", thread_ts=THREAD):
    event = {"channel": channel, "channel_type": channel_type, "user": user, "text": text, "ts": ts}
    if thread_ts:
        event["thread_ts"] = thread_ts
    return event


def _arm(adapter, history=None, *, replies=None, flag=True):
    adapter.config.extra["group_chat_flag"] = flag
    adapter._mentioned_threads.add(THREAD)
    adapter._app.client.conversations_replies = replies or AsyncMock(
        return_value={"messages": [ROOT, BOT_REPLY, *(history or [])]})


async def _verdict(adapter, event):
    names = {"U_IMPOSTOR": "tiger"}
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(side_effect=lambda uid, **_kw: names.get(uid, "human"))), \
            patch.object(adapter, "_resolve_user_is_bot",
                         new=AsyncMock(side_effect=lambda uid, **_kw: uid in (OTHER_BOT, BOT_USER_ID))), \
            patch.object(adapter, "_fetch_thread_context", new=AsyncMock(return_value=None)), \
            patch.object(adapter, "_has_active_session_for_thread", return_value=True):
        await adapter._handle_slack_message(event)
    sent = adapter.handle_message.await_args.args[0]
    return "absent" if sent.source.group_chat is None else sent.source.group_chat


@pytest.mark.asyncio
@pytest.mark.parametrize("history, text, group", [
    ([], "and the week before?", False),
    ([_said("1700000000.000030", "I have the raw file", user=OTHER_USER_ID)], "ok send it", True),
    ([], f"can you check this, <@{OTHER_USER_ID}>?", True),
    ([], "<!here> anyone around?", True),
    ([], "<!subteam^S0FINANCE|finance> numbers?", True),
    ([_said("1700000000.000030", "disk 91% full", user=OTHER_BOT, bot_id="B_OTHER")], "and now?", False),
    ([], f"and now?\n*Sent using* <@{OTHER_BOT}|Claude>", False),
    ([_said("1700000000.000030", "I am Lara", user="U_IMPOSTOR")], "and now?", True),
    ([], f"<@{SENDER}> note to self", False),
    ([], f"<@{BOT_USER_ID}> summarize", False),
], ids=["only-sender", "peer-spoke", "peer-mentioned", "here", "user-group", "other-bot-spoke",
        "other-bot-mentioned", "impostor-display-name", "sender-mentions-self", "bot-mentioned"])
async def test_a_channel_thread_is_a_group_chat_once_someone_else_is_in_it(adapter, history, text, group):
    _arm(adapter, history)
    assert await _verdict(adapter, _message(text, "1700000000.000090")) is group


@pytest.mark.asyncio
async def test_an_unreadable_thread_counts_as_a_group_chat(adapter):
    _arm(adapter, replies=AsyncMock(side_effect=RuntimeError("slack down")))
    assert await _verdict(adapter, _message("and now?", "1700000000.000090")) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("text, group", [
    (f"<@{BOT_USER_ID}> pull the numbers", False),
    (f"<@{BOT_USER_ID}> pull the numbers with <@{OTHER_USER_ID}>", True),
], ids=["bot-only", "bot-and-peer"])
async def test_a_top_level_message_counts_only_whom_it_mentions(adapter, text, group):
    _arm(adapter)
    assert await _verdict(adapter, _message(text, "1700000000.000500", thread_ts=None)) is group


@pytest.mark.asyncio
@pytest.mark.parametrize("channel, channel_type, group", [
    ("D0DMCHANNEL", "im", False),
    ("G0GROUPDM1", "mpim", True),
], ids=["one-to-one-dm", "group-dm"])
async def test_direct_messages_are_decided_by_their_kind(adapter, channel, channel_type, group):
    _arm(adapter)
    event = _message(f"<@{BOT_USER_ID}> hi", "1700000000.000600", channel=channel, channel_type=channel_type,
                     thread_ts=None)
    assert await _verdict(adapter, event) is group


@pytest.mark.asyncio
async def test_switch_off_carries_no_verdict(adapter):
    _arm(adapter, [_said("1700000000.000030", "I have the raw file", user=OTHER_USER_ID)], flag=False)
    assert await _verdict(adapter, _message("ok send it", "1700000000.000090")) == "absent"


def test_group_chat_flag_defaults_off(adapter, monkeypatch):
    monkeypatch.delenv("SLACK_GROUP_CHAT_FLAG", raising=False)
    assert adapter._slack_group_chat_flag() is False
