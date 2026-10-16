"""Slack mrkdwn read as Markdown (``platforms.slack.extra.inbound_markdown``), the inverse of
``SlackAdapter.format_message``, so a single ``*`` means the same in what the agent reads and in what it
writes. Bold becomes ``**``, links ``[label](url)``, and Slack's ``&``/``<``/``>`` escaping is undone.
Mentions, channels, broadcasts and dates keep Slack's form, and code keeps its characters. Slack's client
writes a bold run as ``*…*`` even beside Chinese text, so only the run's own edges must be non-space;
the U+200B that ``format_message`` puts around emphasis is scaffolding and is dropped."""

from __future__ import annotations

import re

_ZWSP = "​"
_FENCE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_SLACK_ANGLE = re.compile(r"<([^<>\n]+)>")
_LINK_SCHEMES = ("http://", "https://", "mailto:", "tel:")
_ESCAPED = re.compile(r"&(amp|lt|gt);")
_UNESCAPED = {"amp": "&", "lt": "<", "gt": ">"}
_BOLD = re.compile(rf"{_ZWSP}?(?<!\*)\*([^\s*](?:[^*\n]*?[^\s*])?)\*(?!\*){_ZWSP}?")
_KEPT = re.compile(r"\x00(\d+)\x00")


def _unescape(text: str) -> str:
    return _ESCAPED.sub(lambda match: _UNESCAPED[match.group(1)], text)


def _markdown_angle(inner: str) -> str:
    target, _, label = inner.partition("|")
    if not target.startswith(_LINK_SCHEMES):
        return f"<{inner}>"
    url, label = _unescape(target), _unescape(label)
    return f"[{label}]({url})" if label and label != url else url


def slack_mrkdwn_to_markdown(text: str) -> str:
    if not text:
        return text
    kept: list[str] = []

    def keep(value: str) -> str:
        kept.append(value)
        return f"\x00{len(kept) - 1}\x00"

    text = text.replace("\x00", "")
    text = _FENCE.sub(lambda match: keep(_unescape(match.group(0))), text)
    text = _INLINE_CODE.sub(lambda match: keep(_unescape(match.group(0))), text)
    text = _SLACK_ANGLE.sub(lambda match: keep(_markdown_angle(match.group(1))), text)
    text = _BOLD.sub(lambda match: f"**{match.group(1).strip(_ZWSP)}**", _unescape(text))
    return _KEPT.sub(lambda match: kept[int(match.group(1))], text)
