"""Who may steer a running turn (``display.busy_steer_same_sender``).

A turn acts as the person whose message opened it. A steer lands inside that turn, so only a
follow-up from that same person qualifies: same platform, same scope (workspace) and same user id,
all as the platform's event carries them. Display names are user-editable and never count; a bot
author never counts; a side with no user id never matches.
"""

from __future__ import annotations

from typing import Any


def same_sender(turn_source: Any, follow_up_source: Any) -> bool:
    if turn_source is None or follow_up_source is None or getattr(follow_up_source, "is_bot", False):
        return False
    user_id = getattr(follow_up_source, "user_id", None)
    return bool(user_id) and all(
        getattr(turn_source, field, None) == getattr(follow_up_source, field, None)
        for field in ("platform", "scope_id", "user_id")
    )
