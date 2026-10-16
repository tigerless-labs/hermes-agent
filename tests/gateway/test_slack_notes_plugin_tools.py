"""The Slack platform note can count Slack tools a plugin provides
(``platforms.slack.extra.slack_tools_via_plugin``).

Upstream decides between "you have Slack tools" and "you do NOT have access to Slack-specific
APIs — you cannot search channel history … do not promise" by looking only at the native Slack
toolset and Slack-named MCP servers. A deployment whose Slack tools come from a plugin (a tool hub
read as the person speaking) got the second note, which contradicts the plugin's own guidance, and
the model sometimes refused to read a channel it could read. With the switch on, the check counts
the plugin's tools, so the note and the agent-cache key both say Slack tools are loaded.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.session import (
    SessionContext, SessionSource, _slack_tools_loaded, build_session_context_prompt,
)

TOOLS_NOTE = "have access to Slack-specific tools"
NO_TOOLS_NOTE = "cannot search channel history"


def _config(extra=None, platform="slack"):
    return {"platforms": {platform: {"extra": extra or {}}}} if extra is not None else {}


def _check(config, monkeypatch, *, raises=False):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

    def load(*_a, **_k):
        if raises:
            raise RuntimeError("config unreadable")
        return config

    with patch("tools.mcp_tool_discovery.get_registered_mcp_server_names", return_value=[]), \
            patch("hermes_cli.config.load_config_readonly", side_effect=load):
        loaded = _slack_tools_loaded()
        context = SessionContext(
            source=SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group", user_id="U1",
                                 thread_id="100.1"),
            connected_platforms=[Platform.SLACK], home_channels={}, created_at=datetime.now())
        prompt = build_session_context_prompt(context)
    return loaded, prompt


def test_switch_off_keeps_upstreams_no_slack_tools_note(monkeypatch):
    loaded, prompt = _check(_config({}), monkeypatch)
    assert loaded is False
    assert NO_TOOLS_NOTE in prompt and TOOLS_NOTE not in prompt


def test_plugin_provided_slack_tools_get_the_slack_tools_note(monkeypatch):
    loaded, prompt = _check(_config({"slack_tools_via_plugin": True}), monkeypatch)
    assert loaded is True
    assert TOOLS_NOTE in prompt and NO_TOOLS_NOTE not in prompt


@pytest.mark.parametrize("config", [
    _config({"slack_tools_via_plugin": "true"}),
    _config({"slack_tools_via_plugin": 1}),
    _config({"slack_tools_via_plugin": True}, platform="discord"),
    {"slack_tools_via_plugin": True},
], ids=["switch as text", "switch as number", "switch under another platform", "switch at the top level"])
def test_redteam_only_the_slack_switch_set_to_true_counts(monkeypatch, config):
    loaded, prompt = _check(config, monkeypatch)
    assert loaded is False and NO_TOOLS_NOTE in prompt


def test_redteam_an_unreadable_config_still_fails_closed(monkeypatch):
    loaded, prompt = _check({}, monkeypatch, raises=True)
    assert loaded is False and NO_TOOLS_NOTE in prompt
