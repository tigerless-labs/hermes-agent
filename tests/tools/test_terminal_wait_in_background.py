"""A messaging deployment can tell the model to wait in the background (``terminal.wait_in_background``).

Upstream's terminal description has the model hold its turn with a foreground sleep. In a chat gateway the person
then watches a busy indicator for minutes; with the switch on, the description says what the background path
already does there: the command outlives the turn and its completion wakes the model in the same conversation.
"""

import copy

import pytest

import hermes_cli.config as hermes_config
import tools.terminal_tool as terminal_tool
from hermes_cli.config_defaults import DEFAULT_CONFIG
from tools.registry import registry


def _terminal_definition(monkeypatch, terminal_section) -> dict:
    monkeypatch.setattr(hermes_config, "load_config_readonly", lambda: {"terminal": terminal_section})
    monkeypatch.setattr(registry.get_entry("terminal"), "check_fn", None)
    [definition] = registry.get_definitions({"terminal"}, quiet=True)
    return definition["function"]


def test_upstream_still_words_waiting_as_a_foreground_sleep():
    assert terminal_tool.FOREGROUND_WAIT_GUIDANCE in terminal_tool.TERMINAL_TOOL_DESCRIPTION


def test_the_switch_is_off_by_default():
    assert DEFAULT_CONFIG["terminal"]["wait_in_background"] is False


@pytest.mark.parametrize("section", [{}, {"wait_in_background": False}, {"wait_in_background": "false"},
                                     {"wait_in_background": "off"}])
def test_off_leaves_upstream_s_terminal_definition_as_it_is(monkeypatch, section):
    definition = _terminal_definition(monkeypatch, section)
    assert definition["description"] == terminal_tool.TERMINAL_TOOL_DESCRIPTION
    assert definition["parameters"] == terminal_tool.TERMINAL_SCHEMA["parameters"]


@pytest.mark.parametrize("value", [True, "true", "on", "1"])
def test_on_words_waiting_as_background_work_that_outlives_the_turn_and_wakes_the_model(monkeypatch, value):
    definition = _terminal_definition(monkeypatch, {"wait_in_background": value})
    description = definition["description"]
    assert terminal_tool.FOREGROUND_WAIT_GUIDANCE not in description
    assert terminal_tool.BACKGROUND_WAIT_GUIDANCE in description
    assert description.index("after your turn ends") < description.index("wakes you") < description.index(
        "end your turn")
    background = definition["parameters"]["properties"]["background"]["description"]
    assert "after your turn ends" in background and "woken" in background


def test_on_changes_nothing_but_the_waiting_guidance(monkeypatch):
    definition = _terminal_definition(monkeypatch, {"wait_in_background": True})
    restored = definition["description"].replace(terminal_tool.BACKGROUND_WAIT_GUIDANCE,
                                                 terminal_tool.FOREGROUND_WAIT_GUIDANCE)
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
