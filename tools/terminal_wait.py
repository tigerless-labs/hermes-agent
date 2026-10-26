"""``terminal.wait_in_background``: waiting happens in the background where a completion can wake the model.

Upstream's terminal description has the model hold its turn with a foreground sleep. In a messaging gateway a
background command with notify=true already outlives the turn and wakes the model in the same conversation when it
exits, so with the switch on the description says so, and a foreground command whose sleeps reach
``terminal.foreground_sleep_limit_seconds`` is started that way instead (like an over-cap foreground timeout is).
Sessions that cannot receive a completion (cron, one-shot runners) and delegated children keep the foreground.
"""

import json
import logging
import math
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_FOREGROUND_SLEEP_LIMIT_SECONDS = 60

FOREGROUND_WAIT_GUIDANCE = (
    "Foreground (default) returns as soon as the command finishes. To wait a fixed time, run the wait in the "
    "foreground; never background a sleep or polling loop. Background is only for commands that must keep running "
    "after this call returns;")
BACKGROUND_WAIT_GUIDANCE = (
    "Foreground (default) holds your turn until the command finishes. With background=true and notify=true a command "
    "keeps running after your turn ends and wakes you in this conversation when it exits: run a long command that "
    "way, or a wait before checking on something (a foreground sleep of {limit}s or more is started that way for "
    "you), then end your turn and tell the user you will report back;")
BACKGROUND_PARAMETER_GUIDANCE = (
    "Keep running after this call and after your turn ends; returns a session_id. With notify=true you are woken in "
    "this conversation when it exits.")
PROMOTED_SLEEP_NOTE = (
    "This command sleeps {seconds:g}s, so it was started as a tracked background process with notify_on_complete=true "
    "instead of holding your turn. Do NOT re-run it. Its completion (exit code + output tail) wakes you in this "
    "conversation: tell the user you will report back, and end your turn.")

_UNIT_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}
_SLEEP_CALL = re.compile(r"(?:^|(?<=[\s;&|(){}!]))(?:/usr)?(?:/bin/)?sleep((?:\s+[^\s;&|()<>]+)+)")
_SLEEP_ARGUMENT = re.compile(r"(\d+(?:\.\d*)?|\.\d+)([smhd]?)")


def _terminal_section() -> dict:
    from hermes_cli.config import load_config_readonly

    section = load_config_readonly().get("terminal")
    return section if isinstance(section, dict) else {}


def foreground_sleep_limit() -> Optional[int]:
    """The sleep length that moves a foreground command to the background, or None when the switch is off."""
    from utils import is_truthy_value

    section = _terminal_section()
    if not is_truthy_value(section.get("wait_in_background"), default=False):
        return None
    try:
        limit = int(section.get("foreground_sleep_limit_seconds", DEFAULT_FOREGROUND_SLEEP_LIMIT_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_FOREGROUND_SLEEP_LIMIT_SECONDS
    return limit if limit > 0 else DEFAULT_FOREGROUND_SLEEP_LIMIT_SECONDS


def foreground_sleep_seconds(command: str) -> float:
    """Seconds the command's literal ``sleep`` calls add up to; quoted text and non-literal arguments count zero."""
    from tools.terminal_tool_guards import _strip_quotes

    total = 0.0
    for call in _SLEEP_CALL.finditer(_strip_quotes(command)):
        for argument in call.group(1).split():
            if argument == "infinity":
                return math.inf
            literal = _SLEEP_ARGUMENT.fullmatch(argument)
            if literal is None:
                break
            total += float(literal.group(1)) * _UNIT_SECONDS[literal.group(2)]
    return total


def sleep_to_promote(command: Any) -> Optional[float]:
    """Seconds of sleep that start this foreground command in the background, or None to keep the foreground."""
    if not isinstance(command, str):
        return None
    try:
        limit = foreground_sleep_limit()
        if limit is None:
            return None
        from agent.delegation_context import is_delegated_child_context
        from gateway.session_context import async_delivery_supported

        if not async_delivery_supported() or is_delegated_child_context():
            return None
    except Exception as error:
        logger.warning("terminal.wait_in_background unreadable (%s); keeping the foreground", error)
        return None
    seconds = foreground_sleep_seconds(command)
    return seconds if seconds >= limit else None


def with_promoted_sleep_note(result_json: str, seconds: float) -> str:
    """Attach the sleep promotion note to a spawn result that kept its completion notice (unchanged otherwise)."""
    try:
        data = json.loads(result_json)
    except (TypeError, ValueError):
        return result_json
    if not isinstance(data, dict) or data.get("error") or not data.get("notify_on_complete"):
        return result_json
    data["promoted_from_foreground"] = PROMOTED_SLEEP_NOTE.format(seconds=seconds)
    return json.dumps(data, ensure_ascii=False)


def schema_overrides(description: str, parameters: dict) -> dict:
    """The terminal tool's per-call overrides: background wording when the switch is on, else none."""
    limit = foreground_sleep_limit()
    if limit is None:
        return {}
    properties = parameters["properties"]
    return {
        "description": description.replace(FOREGROUND_WAIT_GUIDANCE, BACKGROUND_WAIT_GUIDANCE.format(limit=limit)),
        "parameters": {**parameters, "properties": {
            **properties, "background": {**properties["background"], "description": BACKGROUND_PARAMETER_GUIDANCE}}},
    }
