"""A steered message keeps its own reply verdict (``display.busy_steer_reply_expected``).

A steer hands the running agent text only, so upstream loses whether that message needed an answer:
a steer the model reads leaves the turn judged by its opener alone, and one that lands after the
turn's last model call runs as the next turn with no verdict, which counts as addressed. With the
switch on, every accepted steer is recorded with its message on the turn it entered. When that turn
ends, the steers the model read fold into its verdict, and the leftover ones (still in the agent's
leftover steer text) give the next turn theirs, combined as one turn answering several messages is.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from gateway.platforms.event import MessageEvent

Steer = Tuple[str, Any]


def combined(reply_expected: Optional[bool], events: Iterable[Any]) -> Optional[bool]:
    holder = SimpleNamespace(reply_expected=reply_expected)
    for event in events:
        MessageEvent.absorb_reply_expected(holder, event)
    return holder.reply_expected


def split(steers: Sequence[Steer], leftover_text: Optional[str]) -> Tuple[List[Any], List[Any]]:
    """``(read, leftover)`` messages: a steer is leftover while its text is in *leftover_text*."""
    leftover_text = leftover_text or ""
    read: List[Any] = []
    leftover: List[Any] = []
    for text, event in steers:
        (leftover if text and text in leftover_text else read).append(event)
    return read, leftover
