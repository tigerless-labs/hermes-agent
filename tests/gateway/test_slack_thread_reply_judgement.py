"""``platforms.slack.extra.thread_reply_judgement``: the bot judges replies in shared threads.

In a thread where someone besides the sender and the bot has spoken or been @mentioned, a message
that does not @mention the bot may stay silent (``reply_expected`` False); with only the sender and
the bot it must be answered (True). Who is in the thread comes from Slack user ids, seeded once from
the thread's history and kept current from each admitted message; an unreadable history means the
message must be answered. The bot's own @mention stays visible so the model can tell, and the
identity line says how to stay silent everywhere but a 1:1 DM. Off, upstream behaviour holds.
Driven through the real ``_handle_slack_message``.
"""
from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway.test_slack_ignore_other_user_mentions import (  # noqa: F401 - fixtures
    BOT_USER_ID, CHANNEL_ID, OTHER_USER_ID, _redirect_cache, adapter,
)

THREAD = "1700000000.000010"
SENDER = "U_HUMAN"
BOT_NAME = "tiger"
ROOT = {"ts": THREAD, "user": SENDER, "text": f"<@{BOT_USER_ID}> pull last week's numbers"}
BOT_REPLY = {"ts": "1700000000.000020", "user": BOT_USER_ID, "bot_id": "B_SELF", "text": "Done."}
PEER_REPLY = {"ts": "1700000000.000030", "user": OTHER_USER_ID, "text": "I have the raw file"}
OTHER_BOT_REPLY = {"ts": "1700000000.000040", "user": "U_BOT_OTHER", "bot_id": "B_OTHER", "text": "alert"}


def _message(text, ts, *, user=SENDER, channel=CHANNEL_ID, channel_type="channel", thread_ts=THREAD):
    event = {"channel": channel, "channel_type": channel_type, "user": user, "text": text, "ts": ts}
    if thread_ts:
        event["thread_ts"] = thread_ts
    return event


def _history(*messages):
    return {"messages": [ROOT, BOT_REPLY, *messages]}


def _arm(adapter, *, history=None, judgement=True, replies=None):
    if judgement:
        adapter.config.extra["thread_reply_judgement"] = True
    adapter._bot_display_name = BOT_NAME
    adapter._mentioned_threads.add(THREAD)
    adapter._app.client.conversations_replies = replies or AsyncMock(return_value=history or _history())
    return adapter._app.client.conversations_replies


async def _admit(adapter, event, *, names=None):
    names = names or {}
    resolve = AsyncMock(side_effect=lambda uid, **_kw: names.get(uid, "human"))
    with patch.object(adapter, "_resolve_user_name", new=resolve), \
            patch.object(adapter, "_fetch_thread_context", new=AsyncMock(return_value=None)), \
            patch.object(adapter, "_fetch_thread_parent_text", new=AsyncMock(return_value="")), \
            patch.object(adapter, "_has_active_session_for_thread", return_value=False):
        await adapter._handle_slack_message(event)
    return adapter.handle_message.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("history, event, expected", [
    (_history(), _message("and the week before?", "2.1"), True),
    (_history(PEER_REPLY), _message("ok send it over", "2.2"), False),
    (_history(), _message(f"<@{OTHER_USER_ID}> can you check?", "2.3"), False),
    (_history(), _message("<!here> anyone around?", "2.4"), False),
    (_history(), _message("<!subteam^S0FINANCE|finance> numbers?", "2.5"), False),
    (_history(PEER_REPLY), _message(f"<@{BOT_USER_ID}> summarize", "2.6"), True),
    (_history(OTHER_BOT_REPLY), _message("and now?", "2.7"), True),
    ({"messages": [dict(ROOT, user=OTHER_USER_ID)]}, _message("thoughts?", "2.8"), False),
], ids=["only-sender", "peer-spoke", "peer-mentioned", "here", "user-group", "bot-mentioned",
        "other-bot-spoke", "peer-started-thread"])
async def test_a_thread_follow_up_may_go_unanswered_only_with_someone_else_there(
        adapter, history, event, expected):
    _arm(adapter, history=history)
    assert (await _admit(adapter, event)).reply_expected is expected


@pytest.mark.asyncio
async def test_a_one_to_one_dm_is_always_answered(adapter):
    _arm(adapter, history=_history(PEER_REPLY))
    dm = _message("hi", "3.1", channel="D0001", channel_type="im", thread_ts=None)
    assert (await _admit(adapter, dm)).reply_expected is True


@pytest.mark.asyncio
async def test_who_is_in_the_thread_is_read_from_slack_once_then_kept_from_each_message(adapter):
    replies = _arm(adapter)
    assert (await _admit(adapter, _message("first", "4.1"))).reply_expected is True
    assert (await _admit(adapter, _message("I can help", "4.2", user=OTHER_USER_ID))).reply_expected is False
    assert (await _admit(adapter, _message("great, go ahead", "4.3"))).reply_expected is False
    assert replies.await_count == 1


@pytest.mark.asyncio
async def test_redteam_an_unreadable_thread_history_means_the_message_must_be_answered(adapter):
    _arm(adapter, replies=AsyncMock(side_effect=RuntimeError("slack is down")))
    assert (await _admit(adapter, _message("status?", "5.1"))).reply_expected is True


@pytest.mark.asyncio
async def test_redteam_a_person_named_like_the_bot_is_still_someone_else(adapter):
    _arm(adapter)
    event = _message(f"<@{OTHER_USER_ID}> can you take this?", "5.2")
    admitted = await _admit(adapter, event, names={OTHER_USER_ID: BOT_NAME})
    assert admitted.reply_expected is False


@pytest.mark.asyncio
async def test_the_bots_own_mention_stays_visible_and_names_it(adapter):
    _arm(adapter, history=_history(PEER_REPLY))
    admitted = await _admit(adapter, _message(f"<@{BOT_USER_ID}> summarize this", "6.1"))
    assert admitted.text.startswith(f"@{BOT_NAME}")
    assert f"<@{BOT_USER_ID}>" not in admitted.text


@pytest.mark.asyncio
async def test_off_keeps_upstream_rules_and_strips_the_bots_own_mention(adapter):
    replies = _arm(adapter, history=_history(PEER_REPLY), judgement=False)
    follow_up = await _admit(adapter, _message("ok send it over", "7.1"))
    mention = await _admit(adapter, _message(f"<@{BOT_USER_ID}> summarize", "7.2"))
    assert follow_up.reply_expected is None
    assert f"@{BOT_NAME}" not in mention.text
    assert replies.await_count == 0
    assert "treat every delivered turn" in mention.channel_prompt


@pytest.mark.asyncio
async def test_the_identity_line_says_how_to_stay_silent_outside_a_one_to_one_dm(adapter):
    _arm(adapter)
    in_thread = await _admit(adapter, _message("and now?", "8.1"))
    again = await _admit(adapter, _message("one more", "8.2"))
    dm = await _admit(adapter, _message("hi", "8.3", channel="D0001", channel_type="im", thread_ts=None))
    assert "NO_REPLY" in in_thread.channel_prompt and "NO_REPLY" not in dm.channel_prompt
    assert f"@{BOT_NAME}" in in_thread.channel_prompt and f"@{BOT_NAME}" in dm.channel_prompt
    assert "treat every delivered turn" not in in_thread.channel_prompt
    assert in_thread.channel_prompt == again.channel_prompt
