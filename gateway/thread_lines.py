"""How a bot post names the line it belongs to, when each person in a thread has their own session.

With ``thread_sessions_per_user`` on, a thread holds one session per participant: a line. Slack's
``extra.thread_lines`` then reads the thread into every turn of a line and tells the bot's posts
apart by line. A plugin registers one :class:`ThreadLineRecord` (``ctx.register_thread_line_record``)
that owns the record's format: the metadata a post in a line carries, how a posted message's own
record is read back, and how the model reads a post of another line. Only the bot's own messages
are ever read for a record, never message text, so a person typing a label stays ordinary text.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ThreadLineRecord(ABC):
    """Registered by a plugin; the platform-native form of a post's line."""

    @abstractmethod
    def metadata(self, owner: str) -> Dict[str, Any]:
        """The platform's message metadata every post in ``owner``'s line carries; shown to no one."""

    @abstractmethod
    def owner(self, message: Dict[str, Any]) -> Optional[str]:
        """The line owner a posted message's own record names, or None when it carries none."""

    @abstractmethod
    def label(self, owner_name: Optional[str]) -> str:
        """How the model reads a post of another line; ``None`` for a post with no record."""
