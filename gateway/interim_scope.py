"""Interim assistant messages only in one-to-one conversations
(``display.interim_assistant_messages_one_to_one_only``), and the group-chat verdict as a session value.

The adapter's verdict rides on the message's source (``SessionSource.group_chat``). A turn sends
interim messages only when that verdict is a plain ``False``; a group chat, or no verdict, sends none.
The same verdict is bound per turn as ``HERMES_SESSION_GROUP_CHAT``: ``"1"`` group chat, ``"0"``
one-to-one, ``""`` unknown.
"""

from __future__ import annotations

from typing import Any


def interim_allowed(source: Any, one_to_one_only: bool) -> bool:
    return not one_to_one_only or getattr(source, "group_chat", None) is False


def group_chat_value(source: Any) -> str:
    verdict = getattr(source, "group_chat", None)
    return {True: "1", False: "0"}.get(verdict, "") if isinstance(verdict, bool) else ""
