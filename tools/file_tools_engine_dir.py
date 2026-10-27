"""``terminal.refuse_engine_dir_writes``: the file tools keep out of the sandbox's engine folder.

Inside a container sandbox the engine's own folder sits at ``/root/.hermes``: the chat's caches there are read-only
copies, and media delivery never fetches from it (``gateway/media_fetch.py``), so a file written there can neither
be changed in place nor attached. Generated images and downloaded chat files live in that tree, and models copied
its layout for their own deliverables. With the switch on, write_file and patch refuse a target in that folder and
say where to write instead. The terminal can still write there; delivery stays the boundary.
"""

import logging
import posixpath

logger = logging.getLogger(__name__)

SANDBOX_ENGINE_DIR = "/root/.hermes"
SANDBOX_HOME = "/root"
ATTACHABLE_DIR = "/tmp"


def refuse_engine_dir_writes() -> bool:
    try:
        from hermes_cli.config import load_config_readonly
        from utils import is_truthy_value

        section = load_config_readonly().get("terminal")
        return isinstance(section, dict) and is_truthy_value(section.get("refuse_engine_dir_writes"), default=False)
    except Exception as error:
        logger.warning("terminal.refuse_engine_dir_writes unreadable (%s); writing as upstream does", error)
        return False


def _sandbox_target(filepath: str, task_id: str) -> str | None:
    from tools.file_tools_paths import _resolve_path_for_task, _uses_container_paths

    if not _uses_container_paths(task_id):
        return None
    raw = filepath.strip()
    if raw == "~" or raw.startswith("~/"):
        return posixpath.normpath(SANDBOX_HOME + raw[1:])
    try:
        return posixpath.normpath(str(_resolve_path_for_task(raw, task_id)))
    except (OSError, ValueError):
        return posixpath.normpath(raw) if posixpath.isabs(raw) else None


def engine_dir_write_refusal(filepath: str, task_id: str = "default") -> str | None:
    """The refusal for a write landing in the sandbox's engine folder, else None."""
    if not filepath or not refuse_engine_dir_writes():
        return None
    target = _sandbox_target(filepath, task_id)
    if target is None or not (target == SANDBOX_ENGINE_DIR or target.startswith(SANDBOX_ENGINE_DIR + "/")):
        return None
    return (f"Not written: {filepath} is in {SANDBOX_ENGINE_DIR}, the engine's own folder in this sandbox. The chat's "
            f"files there are read-only copies, and a file written there can never be attached. Write it under "
            f"{ATTACHABLE_DIR} instead, where a MEDIA: attachment reaches it; to change one of the chat's files, copy "
            f"it to {ATTACHABLE_DIR} first and edit the copy.")
