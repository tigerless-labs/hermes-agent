"""``platforms.slack.extra.inbound_markdown``: the Slack text the agent reads, the message itself and each
message a thread backfill carries, arrives as Markdown, the inverse of ``format_message``. Bold becomes
``**``, links become ``[label](url)`` and Slack's ``&``/``<``/``>`` escaping is undone. Mentions, channels,
broadcasts and dates stay in Slack's form, and code keeps its characters. Routing, mention detection and
commands still read Slack's own text. With the switch off the text is upstream's. Messages are shaped the
way the Slack client writes them: a bold run is ``*…*`` in ``text`` even beside Chinese text, and a
``rich_text`` block carries the same run with ``style.bold``.
"""

import html
import re
from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway.test_slack_ignore_other_user_mentions import (  # noqa: F401 - fixtures
    BOT_USER_ID, CHANNEL_ID, _redirect_cache, adapter,
)
from plugins.platforms.slack.inbound_markdown import slack_mrkdwn_to_markdown

ZWSP = "​"
SENDER = "U_HUMAN"
DM = "D0SENDER01"


def _bold(text):
    return ("bold", text)


def _composed(*parts):
    """``(text, blocks)`` the way the Slack client writes a message: a bold part is ``*…*`` in ``text``
    and a ``style.bold`` element in the ``rich_text`` block."""
    flat, elements = [], []
    for part in parts:
        if isinstance(part, tuple):
            flat.append(f"*{part[1]}*")
            elements.append({"type": "text", "text": part[1], "style": {"bold": True}})
        else:
            flat.append(part)
            elements.append({"type": "text", "text": part})
    blocks = [{"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": elements}]}]
    return "".join(flat), blocks


COMPOSED = [
    _composed("周报：\n\n", _bold("車險：Form 4，WeChat 0，Purchase 1"), "\n• Form：Google 1、Red 1\n\n",
              _bold("寵物險：Quote 0，Cart 0"), "\n\n再調整一次"),
    _composed("应该是", _bold("Hawaii"), "吗"),
    _composed("• ", _bold("循环逻辑"), "在 cron 里"),
    _composed("第一個", _bold("Auto+Pet"), "，不用改"),
    _composed(_bold("Lara"), " 已上线"),
]

MRKDWN = [text for text, _ in COMPOSED] + [
    "看 <https://example.com/report?a=1&amp;b=2|这份报告> 和 <https://example.com/raw>",
    "a &lt; b &amp;&amp; c &gt; d",
    "&gt; 引用的话\n回复",
    f"<@{BOT_USER_ID}> 帮 <@U0PEER0001> 看 <#C0CHAN0001|general>",
    "<!here> *今天* 开会 <!date^1700000000^{date_short}|Nov 14>",
    "`a*b*c` 与 `x &lt; y`\n```\n*不是粗体* &amp;\n```",
    "*~删掉~* 和 *_斜_*",
]

LEFT_ALONE = [
    "snake_case_name 与 _斜体_",
    "~删除线~ 5~10 分钟",
    "a * b * c",
    "* 第一项\n* 第二项",
    "已经是 **Markdown** 粗体",
]

SLACK_TOKEN = re.compile(r"<[@#!][^>]*>")
CODE = re.compile(r"```[\s\S]*?```|`[^`\n]+`")


def _bold_runs(blocks):
    return [element["text"] for block in blocks for section in block["elements"]
            for element in section["elements"] if element.get("style", {}).get("bold")]


def _markdown_bold_runs(markdown):
    return re.findall(r"\*\*(.+?)\*\*", markdown)


def _as_slack_shows(mrkdwn):
    """What a reader sees: U+200B is invisible, entities show as their characters, and Slack links a bare
    URL the same as ``<url>``."""
    shown = html.unescape(mrkdwn.replace(ZWSP, ""))
    return re.sub(r"<((?:https?|mailto|tel):[^|>]+)>", r"\1", shown)


@pytest.mark.parametrize("text, blocks", COMPOSED)
def test_each_bold_run_the_client_wrote_reads_as_markdown_bold(text, blocks):
    markdown = slack_mrkdwn_to_markdown(text)
    assert _markdown_bold_runs(markdown) == _bold_runs(blocks)
    assert "*" not in re.sub(r"\*\*.+?\*\*", "", markdown)


@pytest.mark.parametrize("text", MRKDWN)
def test_sending_the_markdown_back_shows_what_slack_showed(adapter, text):
    assert _as_slack_shows(adapter.format_message(slack_mrkdwn_to_markdown(text))) == _as_slack_shows(text)


@pytest.mark.parametrize("text", MRKDWN)
def test_mentions_channels_broadcasts_and_dates_stay_slacks(text):
    markdown = slack_mrkdwn_to_markdown(text)
    assert all(token in markdown for token in SLACK_TOKEN.findall(text))


def test_links_read_as_markdown_links_with_slacks_escaping_undone():
    text = "看 <https://example.com/report?a=1&amp;b=2|这份报告> 和 <https://example.com/raw>"
    markdown = slack_mrkdwn_to_markdown(text)
    assert "[这份报告](https://example.com/report?a=1&b=2)" in markdown
    assert "<" not in markdown and "&amp;" not in markdown


@pytest.mark.parametrize("text", [t for t in MRKDWN if CODE.search(t)])
def test_code_keeps_its_characters(text):
    markdown = slack_mrkdwn_to_markdown(text)
    assert CODE.findall(markdown) == [html.unescape(code) for code in CODE.findall(text)]


@pytest.mark.parametrize("text", LEFT_ALONE)
def test_what_is_not_slack_bold_is_left_as_it_is(text):
    assert slack_mrkdwn_to_markdown(text) == text


def test_lara_own_bold_beside_chinese_reads_back_without_its_zero_width_spaces(adapter):
    sent = adapter.format_message("先看**車險：**再看**寵物險**。")
    markdown = slack_mrkdwn_to_markdown(sent)
    assert ZWSP in sent and ZWSP not in markdown
    assert _markdown_bold_runs(markdown) == ["車險：", "寵物險"]


@pytest.mark.parametrize("typed", ["&lt;!channel&gt; 大家看", "&lt;!here|here&gt;", "&lt;!everyone&gt;"])
def test_a_broadcast_typed_as_text_never_goes_out_live(adapter, typed):
    sent = adapter.format_message(slack_mrkdwn_to_markdown(typed))
    assert not re.search(r"<!(?:channel|here|everyone)", sent)


def _dm(text, ts="1700000000.000100", **extra):
    return {"channel": DM, "channel_type": "im", "user": SENDER, "text": text, "ts": ts, **extra}


def _channel(text, ts="1700000000.000200"):
    return {"channel": CHANNEL_ID, "channel_type": "channel", "user": SENDER, "text": text, "ts": ts}


async def _delivered(adapter, event, *, on):
    adapter.config.extra["inbound_markdown"] = on
    adapter.handle_message.reset_mock()
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="human")), \
            patch.object(adapter, "_fetch_thread_context", new=AsyncMock(return_value=None)):
        await adapter._handle_slack_message(event)
    return adapter.handle_message.await_args.args[0] if adapter.handle_message.await_count else None


@pytest.mark.asyncio
@pytest.mark.parametrize("text, blocks", COMPOSED)
async def test_the_message_reaches_the_agent_as_markdown(adapter, text, blocks):
    on = await _delivered(adapter, _dm(text, blocks=blocks), on=True)
    off = await _delivered(adapter, _dm(text, blocks=blocks, ts="1700000000.000101"), on=False)
    assert _markdown_bold_runs(on.text) == _bold_runs(blocks)
    assert off.text == text


@pytest.mark.asyncio
async def test_a_link_in_the_message_is_not_added_twice_from_its_block(adapter):
    url = "https://example.com/report?a=1&b=2"
    text = "看 <https://example.com/report?a=1&amp;b=2|这份报告>"
    blocks = [{"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [
        {"type": "text", "text": "看 "}, {"type": "link", "url": url, "text": "这份报告"}]}]}]
    on = await _delivered(adapter, _dm(text, blocks=blocks), on=True)
    assert on.text.count(url) == 1


@pytest.mark.asyncio
async def test_a_command_keeps_its_arguments_as_typed(adapter):
    on = await _delivered(adapter, _dm("/title *周报*"), on=True)
    off = await _delivered(adapter, _dm("/title *周报*", ts="1700000000.000102"), on=False)
    assert on.text == off.text


@pytest.mark.asyncio
@pytest.mark.parametrize("on", [True, False])
async def test_a_mention_typed_as_text_does_not_summon_the_bot(adapter, on):
    adapter.config.extra["require_mention"] = True
    assert await _delivered(adapter, _channel(f"&lt;@{BOT_USER_ID}&gt; *帮我*看看"), on=on) is None


@pytest.mark.asyncio
async def test_a_real_mention_summons_the_bot_and_the_rest_reads_as_markdown(adapter):
    adapter.config.extra["require_mention"] = True
    on = await _delivered(adapter, _channel(f"<@{BOT_USER_ID}> *帮我*看看"), on=True)
    assert _markdown_bold_runs(on.text) == ["帮我"]


THREAD = "1700000000.000010"


async def _backfill(adapter, messages, *, on):
    adapter.config.extra["inbound_markdown"] = on
    adapter._thread_context_cache.clear()
    adapter._app.client.conversations_replies = AsyncMock(return_value={"messages": messages})
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(side_effect=lambda uid, **_: uid)):
        return await adapter._fetch_thread_context(channel_id=CHANNEL_ID, thread_ts=THREAD, current_ts="999.0")


@pytest.mark.asyncio
async def test_a_thread_backfill_reads_every_message_as_markdown(adapter):
    human, _ = _composed("应该是", _bold("Hawaii"), "吗")
    lara = adapter.format_message("先看**車險：**再说")
    messages = [{"ts": THREAD, "user": SENDER, "text": human},
                {"ts": "1700000000.000020", "user": "U0PEER0001", "text": lara}]
    on = await _backfill(adapter, messages, on=True)
    off = await _backfill(adapter, messages, on=False)
    assert _markdown_bold_runs(on) == ["Hawaii", "車險："]
    assert ZWSP not in on
    assert human in off and lara in off
