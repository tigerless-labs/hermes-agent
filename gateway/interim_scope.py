"""Interim assistant messages only in one-to-one conversations
(``display.interim_assistant_messages_one_to_one_only``).

The adapter's verdict on whether a conversation is a group chat rides on the message event
(``metadata["slack_group_chat"]``). It reaches the turn through the persisted user row's display
metadata, so a first turn and a turn taken from the queue each carry their own message's verdict.
Only a plain ``False`` counts as one-to-one.
"""

from __future__ import annotations

from typing import Any, Optional

GROUP_CHAT_KEY = "group_chat"
EVENT_VERDICT_KEY = "slack_group_chat"


def group_chat_metadata(event: Any) -> dict:
    verdict = (getattr(event, "metadata", None) or {}).get(EVENT_VERDICT_KEY)
    return {GROUP_CHAT_KEY: verdict} if isinstance(verdict, bool) else {}


def interim_allowed(display_metadata: Optional[dict], one_to_one_only: bool) -> bool:
    return not one_to_one_only or (display_metadata or {}).get(GROUP_CHAT_KEY) is False
