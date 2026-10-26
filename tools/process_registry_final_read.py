"""``terminal.read_output_after_exit``: read a sandbox job's log once more after it exits.

The sandbox poller reads the log's new bytes, then checks whether the process is alive. Output written between the
two steps — typically the last line of a command that prints and exits, like ``sleep 240; date`` — was dropped:
the poller saw the exit and finished without reading again. With the switch on, the exit branch reads the bytes
past the poller's offset once before finishing; a read that fails or does not line up adds nothing.
"""

import logging
import shlex
from typing import Any

from tools.process_registry_env_log import log_delta_command

logger = logging.getLogger(__name__)


def read_output_after_exit() -> bool:
    try:
        from hermes_cli.config import load_config_readonly
        from utils import is_truthy_value

        section = load_config_readonly().get("terminal")
        return isinstance(section, dict) and is_truthy_value(section.get("read_output_after_exit"), default=False)
    except Exception as error:
        logger.warning("terminal.read_output_after_exit unreadable (%s); finishing without a final read", error)
        return False


def _log_past(env: Any, log_path: str, offset: int) -> str:
    try:
        raw = env.execute(log_delta_command(shlex.quote(log_path), offset), timeout=10).get("output", "")
    except Exception as error:
        logger.info("final read of %s failed (%s)", log_path, error)
        return ""
    header, _, delta = raw.partition("\n")
    try:
        _, used_offset = (int(field) for field in header.split())
    except ValueError:
        return ""
    return delta if used_offset == offset else ""


def append_output_after_exit(registry: Any, session: Any, env: Any, log_path: str, offset: int) -> None:
    """With the switch on, add what the log holds past *offset* to the exited sandbox job's output."""
    if not read_output_after_exit():
        return
    delta = _log_past(env, log_path, offset)
    if not delta:
        return
    with session._lock:
        if session.exited:
            return
        session.output_buffer += delta
        if len(session.output_buffer) > session.max_output_chars:
            session.output_buffer = session.output_buffer[-session.max_output_chars:]
    registry._check_watch_patterns(session, delta)
    registry._emit_output(session, delta)
