"""A messaging deployment can have the model wait in the background (``terminal.wait_in_background``).

Upstream's terminal description has the model hold its turn with a foreground sleep. In a chat gateway the person
then watches a busy indicator for minutes, while a background command with notify=true already outlives the turn
and wakes the model in the same conversation. With the switch on the description says so, and a foreground
command whose sleeps reach ``terminal.foreground_sleep_limit_seconds`` starts in the background with its
completion notice, the way an over-cap foreground timeout already does.
"""

import contextvars
import copy
import json
import math
from unittest.mock import patch

import pytest

import hermes_cli.config as hermes_config
import tools.terminal_tool as terminal_tool
import tools.terminal_wait as terminal_wait
from hermes_cli.config_defaults import DEFAULT_CONFIG
from tools.registry import registry


def _configure(monkeypatch, terminal_section) -> None:
    monkeypatch.setattr(hermes_config, "load_config_readonly", lambda: {"terminal": terminal_section})


def _terminal_definition(monkeypatch, terminal_section) -> dict:
    _configure(monkeypatch, terminal_section)
    monkeypatch.setattr(registry.get_entry("terminal"), "check_fn", None)
    [definition] = registry.get_definitions({"terminal"}, quiet=True)
    return definition["function"]


def test_upstream_still_words_waiting_as_a_foreground_sleep():
    assert terminal_wait.FOREGROUND_WAIT_GUIDANCE in terminal_tool.TERMINAL_TOOL_DESCRIPTION


def test_the_switch_is_off_by_default_with_a_sleep_limit_ready():
    assert DEFAULT_CONFIG["terminal"]["wait_in_background"] is False
    assert DEFAULT_CONFIG["terminal"]["foreground_sleep_limit_seconds"] > 0


@pytest.mark.parametrize("section", [{}, {"wait_in_background": False}, {"wait_in_background": "false"},
                                     {"wait_in_background": "off"}])
def test_off_leaves_upstream_s_terminal_definition_as_it_is(monkeypatch, section):
    definition = _terminal_definition(monkeypatch, section)
    assert definition["description"] == terminal_tool.TERMINAL_TOOL_DESCRIPTION
    assert definition["parameters"] == terminal_tool.TERMINAL_SCHEMA["parameters"]


@pytest.mark.parametrize("value", [True, "true", "on", "1"])
def test_on_words_waiting_as_background_work_that_outlives_the_turn_and_wakes_the_model(monkeypatch, value):
    definition = _terminal_definition(monkeypatch, {"wait_in_background": value, "foreground_sleep_limit_seconds": 45})
    description = definition["description"]
    assert terminal_wait.FOREGROUND_WAIT_GUIDANCE not in description
    assert terminal_wait.BACKGROUND_WAIT_GUIDANCE.format(limit=45) in description
    assert description.index("after your turn ends") < description.index("wakes you") < description.index(
        "end your turn")
    background = definition["parameters"]["properties"]["background"]["description"]
    assert "after your turn ends" in background and "woken" in background


def test_on_changes_nothing_but_the_waiting_guidance(monkeypatch):
    definition = _terminal_definition(monkeypatch, {"wait_in_background": True})
    limit = DEFAULT_CONFIG["terminal"]["foreground_sleep_limit_seconds"]
    restored = definition["description"].replace(terminal_wait.BACKGROUND_WAIT_GUIDANCE.format(limit=limit),
                                                 terminal_wait.FOREGROUND_WAIT_GUIDANCE)
    assert restored == terminal_tool.TERMINAL_TOOL_DESCRIPTION
    others = {key: value for key, value in definition["parameters"]["properties"].items() if key != "background"}
    upstream = {key: value for key, value in terminal_tool.TERMINAL_SCHEMA["parameters"]["properties"].items()
                if key != "background"}
    assert others == upstream
    assert definition["parameters"]["required"] == terminal_tool.TERMINAL_SCHEMA["parameters"]["required"]


def test_on_leaves_the_static_schema_untouched(monkeypatch):
    before = copy.deepcopy(terminal_tool.TERMINAL_SCHEMA)
    _terminal_definition(monkeypatch, {"wait_in_background": True})
    assert terminal_tool.TERMINAL_SCHEMA == before


def test_redteam_an_unreadable_config_falls_back_to_upstream_s_definition(monkeypatch):
    def unreadable():
        raise OSError("config.yaml unreadable")

    monkeypatch.setattr(hermes_config, "load_config_readonly", unreadable)
    monkeypatch.setattr(registry.get_entry("terminal"), "check_fn", None)
    [definition] = registry.get_definitions({"terminal"}, quiet=True)
    assert definition["function"]["description"] == terminal_tool.TERMINAL_TOOL_DESCRIPTION


@pytest.mark.parametrize("section", [None, "on", ["wait_in_background"]])
def test_redteam_a_terminal_section_that_is_not_a_mapping_leaves_upstream_s_definition(monkeypatch, section):
    definition = _terminal_definition(monkeypatch, section)
    assert definition["description"] == terminal_tool.TERMINAL_TOOL_DESCRIPTION


@pytest.mark.parametrize("value", ["soon", None, 0, -5])
def test_a_limit_that_is_not_a_positive_number_keeps_the_default(monkeypatch, value):
    _configure(monkeypatch, {"wait_in_background": True, "foreground_sleep_limit_seconds": value})
    assert terminal_wait.foreground_sleep_limit() == terminal_wait.DEFAULT_FOREGROUND_SLEEP_LIMIT_SECONDS


@pytest.mark.parametrize(("command", "seconds"), [
    ("sleep 90", 90),
    ("sleep 480 && echo done", 480),
    ("date; sleep 240; date", 240),
    ("sleep 1m 30s", 90),
    ("sleep 2m; sleep 30", 150),
    ("sleep 0.5", 0.5),
    ("sleep 1h", 3600),
    ("/bin/sleep 120", 120),
    ("timeout 300 sleep 100", 100),
    ("for i in 1 2 3; do sleep 20; done", 20),
    ("echo hi", 0),
])
def test_a_command_s_literal_sleeps_add_up(command, seconds):
    assert terminal_wait.foreground_sleep_seconds(command) == seconds


def test_sleep_infinity_never_ends():
    assert terminal_wait.foreground_sleep_seconds("sleep infinity") == math.inf


@pytest.mark.parametrize("command", ["echo 'sleep 900'", 'echo "sleep 900"', "nosleep 900", "my_sleep 900",
                                     "sleep $DELAY", "sleep", "grep sleep log.txt"])
def test_redteam_text_that_only_mentions_sleep_counts_nothing(command):
    assert terminal_wait.foreground_sleep_seconds(command) == 0


def _env_config(cwd: str) -> dict:
    return {"env_type": "local", "timeout": 180, "cwd": cwd, "host_cwd": None, "modal_mode": "auto",
            "docker_image": "", "singularity_image": "", "modal_image": "", "daytona_image": ""}


def _run(tmp_path, monkeypatch, command: str, **kwargs) -> dict:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    with patch("tools.terminal_tool._get_env_config", return_value=_env_config(str(tmp_path))), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        return json.loads(terminal_tool.terminal_tool(command=command, **kwargs))


def _stop(result: dict) -> None:
    from tools.process_registry import process_registry

    if result.get("session_id"):
        process_registry.kill_process(result["session_id"], source="test")


def test_on_a_foreground_sleep_at_the_limit_starts_in_the_background_and_says_so(tmp_path, monkeypatch):
    _configure(monkeypatch, {"wait_in_background": True, "foreground_sleep_limit_seconds": 1})
    result = _run(tmp_path, monkeypatch, "sleep 30; echo done")
    try:
        assert result.get("error") is None and result["session_id"].startswith("proc_")
        assert result["notify_on_complete"] is True
        note = result["promoted_from_foreground"]
        assert "30s" in note and "end your turn" in note
    finally:
        _stop(result)


def test_on_a_foreground_sleep_under_the_limit_runs_in_the_foreground(tmp_path, monkeypatch):
    _configure(monkeypatch, {"wait_in_background": True, "foreground_sleep_limit_seconds": 60})
    result = _run(tmp_path, monkeypatch, "sleep 0.1; echo done")
    assert result["output"].strip() == "done" and "session_id" not in result


def test_off_a_long_foreground_sleep_stays_in_the_foreground(tmp_path, monkeypatch):
    _configure(monkeypatch, {"wait_in_background": False, "foreground_sleep_limit_seconds": 1})
    result = _run(tmp_path, monkeypatch, "sleep 1.2; echo done")
    assert result["output"].strip() == "done" and "promoted_from_foreground" not in result


def test_an_explicit_background_call_is_left_as_asked(tmp_path, monkeypatch):
    _configure(monkeypatch, {"wait_in_background": True, "foreground_sleep_limit_seconds": 1})
    result = _run(tmp_path, monkeypatch, "sleep 30", background=True)
    try:
        assert result["session_id"].startswith("proc_") and "promoted_from_foreground" not in result
    finally:
        _stop(result)


def test_a_session_that_cannot_receive_a_completion_keeps_the_foreground(tmp_path, monkeypatch):
    from gateway.session_context import declare_stateless_channel

    _configure(monkeypatch, {"wait_in_background": True, "foreground_sleep_limit_seconds": 1})

    def stateless_call() -> dict:
        declare_stateless_channel()
        return _run(tmp_path, monkeypatch, "sleep 1.2; echo done")

    result = contextvars.copy_context().run(stateless_call)
    assert result["output"].strip() == "done" and "promoted_from_foreground" not in result


def test_a_delegated_child_keeps_the_foreground(tmp_path, monkeypatch):
    _configure(monkeypatch, {"wait_in_background": True, "foreground_sleep_limit_seconds": 1})
    with patch("agent.delegation_context.is_delegated_child_context", return_value=True):
        result = _run(tmp_path, monkeypatch, "sleep 1.2; echo done")
    assert result["output"].strip() == "done" and "promoted_from_foreground" not in result


def test_redteam_an_unreadable_config_keeps_the_foreground(tmp_path, monkeypatch):
    def unreadable():
        raise OSError("config.yaml unreadable")

    monkeypatch.setattr(hermes_config, "load_config_readonly", unreadable)
    result = _run(tmp_path, monkeypatch, "sleep 1.2; echo done")
    assert result["output"].strip() == "done" and "promoted_from_foreground" not in result


def test_redteam_a_sleep_inside_quotes_is_not_a_wait(tmp_path, monkeypatch):
    _configure(monkeypatch, {"wait_in_background": True, "foreground_sleep_limit_seconds": 1})
    result = _run(tmp_path, monkeypatch, "echo 'sleep 900'")
    assert result["output"].strip() == "sleep 900" and "promoted_from_foreground" not in result
