"""How a bot post names the line it belongs to, when each person in a thread has their own session.

With ``thread_sessions_per_user`` on, a thread holds one session per participant: a line. Slack's
``extra.thread_lines`` then reads the thread into every turn of a line and tells the bot's posts
apart by line. A plugin registers one :class:`ThreadLineRecord` (``ctx.register_thread_line_record``)
that owns the record's format: the metadata a post in a line carries, how a posted message's own
record is read back, and how the model reads a post of another line. Only the bot's own messages
are ever read for a record, never message text, so a person typing a label stays ordinary text.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from gateway.session import SessionSource


class ThreadLineRecord(ABC):
    """Registered by a plugin; the platform-native form of a post's line."""

    @abstractmethod
    def metadata(self, owner: str) -> dict[str, Any]:
        """The platform's message metadata every post in ``owner``'s line carries; shown to no one."""

    @abstractmethod
    def owner(self, message: dict[str, Any]) -> Optional[str]:
        """The line owner a posted message's own record names, or None when it carries none."""

    @abstractmethod
    def label(self, owner_name: Optional[str]) -> str:
        """How the model reads a post of another line; ``None`` for a post with no record."""


def reads_as_multi_user(
    source: "SessionSource", *, group_sessions_per_user: bool = True, thread_sessions_per_user: bool = False,
) -> bool:
    """Whether a session's turns read as a multi-user thread's, each new message carrying its sender and the
    session context naming no single user: a session its participants share, or, while a plugin registers a
    :class:`ThreadLineRecord`, one person's line in a group chat, which reads the others in its thread."""
    from gateway.session import is_shared_multi_user_session
    from hermes_cli.plugins import get_plugin_thread_line_record
    if is_shared_multi_user_session(source, group_sessions_per_user=group_sessions_per_user,
                                    thread_sessions_per_user=thread_sessions_per_user):
        return True
    return source.chat_type != "dm" and get_plugin_thread_line_record() is not None
