"""``send_message.mirror_to_session`` decides whether a sent message is mirrored into a session."""

from __future__ import annotations

import pytest

import tools.send_message_tool as send_message_tool


@pytest.fixture
def mirrored(monkeypatch):
    calls = []
    monkeypatch.setattr("gateway.mirror.mirror_to_session", lambda *args, **kwargs: calls.append((args, kwargs)) or True)
    return calls


def _config(monkeypatch, settings):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: settings)


def test_a_sent_message_is_mirrored_by_default(monkeypatch, mirrored):
    _config(monkeypatch, {})
    assert send_message_tool._mirror_sent_message("slack", "C0TEAM", "hello", None) is True
    assert len(mirrored) == 1


def test_switched_off_a_sent_message_is_mirrored_nowhere(monkeypatch, mirrored):
    _config(monkeypatch, {"send_message": {"mirror_to_session": False}})
    assert send_message_tool._mirror_sent_message("slack", "C0TEAM", "hello", None) is False
    assert mirrored == []


def test_the_switch_is_a_known_config_key():
    from hermes_cli.config import _validate_config_key

    assert _validate_config_key("send_message.mirror_to_session")[0]
