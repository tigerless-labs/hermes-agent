"""Slack files kept once per chat under their Slack file id (``platforms.slack.extra.files_by_id``):
the identity a file keeps across arrivals, the references it can be named by, and whether it is
shared in a conversation. Pure rules; the adapter downloads and caches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

SLACK_FILE_ID = re.compile(r"F[A-Z0-9]{6,32}")
# A file's permalink (``https://<workspace>.slack.com/files/<user>/<file>/<name>``) or its private
# download URL (``https://files.slack.com/files-pri/<team>-<file>/<name>``).
_FILE_LINK = re.compile(
    r"https://(?:[a-z0-9-]+\.)*slack\.com/(?:files/[A-Z0-9]+/|files-pri/[A-Z0-9]+-)(F[A-Z0-9]{6,32})(?:/|$)")

FILE_NOT_A_REFERENCE = "not_a_slack_file"
FILE_NOT_IN_THIS_CONVERSATION = "not_shared_here"
FILE_UNREADABLE = "unreadable"
FILE_NOT_KEPT = "not_kept"


@dataclass(frozen=True)
class ConversationFile:
    """A Slack file brought into this chat's caches, or why it was not."""
    path: Optional[str]
    name: str = ""
    media_type: str = ""
    size: int = 0
    refusal: Optional[str] = None


def slack_file_id(value: Any) -> Optional[str]:
    """*value* when it is a Slack file id, else None."""
    return value if isinstance(value, str) and SLACK_FILE_ID.fullmatch(value) else None


def file_id_of(reference: Any) -> Optional[str]:
    """The file id a person or the agent names a file by: the id itself or one of its Slack links."""
    if not isinstance(reference, str):
        return None
    reference = reference.strip()
    match = _FILE_LINK.match(reference)
    return match.group(1) if match else slack_file_id(reference)


def kept_key(file_obj: Dict[str, Any]) -> Optional[str]:
    """The cache key a file is kept under: its Slack id, plus Slack's edit time for a file that was
    edited (an edit is a new version; an uploaded file never changes under its id)."""
    file_id = slack_file_id(file_obj.get("id"))
    if file_id is None:
        return None
    edited = str(file_obj.get("edit_timestamp") or "")
    return f"slack-{file_id}-e{edited}" if edited.isdigit() and edited.strip("0") else f"slack-{file_id}"


def shared_in(file_obj: Dict[str, Any], channel_id: str) -> bool:
    """Whether Slack lists *channel_id* among the conversations the file is shared in."""
    listed = [*(file_obj.get(field) or [] for field in ("channels", "groups", "ims"))]
    if any(isinstance(ids, list) and channel_id in ids for ids in listed):
        return True
    shares = file_obj.get("shares") or {}
    return isinstance(shares, dict) and any(isinstance(bucket, dict) and channel_id in bucket
                                            for bucket in shares.values())
