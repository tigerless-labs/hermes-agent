"""``PluginContext.steer_session(content, origin=...)``: the switch, read from the plugin's own profile,
and the hand-off to the live gateway that published the message host."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import hermes_yaml as yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

PLUGIN = "desk"
ORIGIN = {"platform": "slack", "chat_id": "D1", "chat_type": "dm", "thread_id": "170.1", "user_id": "U1",
          "scope_id": "T1"}
WORDS = "only the PDFs"


def _home(path, entry) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(yaml.safe_dump({"plugins": {"entries": {PLUGIN: entry}}}), encoding="utf-8")


def _context(tmp_path, monkeypatch, entry, *, answer=True):
    home = tmp_path / "own"
    _home(home, entry)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    gateway = SimpleNamespace(_steer_plugin_session=MagicMock(return_value=answer))
    manager.set_gateway_message_injector(gateway, MagicMock(return_value=True))
    return PluginContext(PluginManifest(name=PLUGIN, key=PLUGIN, source="user"), manager), manager, gateway


def test_the_live_gateway_is_asked_with_the_plugins_own_home(tmp_path, monkeypatch):
    context, manager, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_steer": True})
    other = tmp_path / "other"
    other.mkdir()
    token = set_hermes_home_override(other)
    try:
        assert context.steer_session(WORDS, origin=ORIGIN) is True
    finally:
        reset_hermes_home_override(token)
    gateway._steer_plugin_session.assert_called_once_with(
        origin=ORIGIN, content=WORDS, plugin_home=manager.home_path, plugin_id=PLUGIN)


def test_a_turn_that_does_not_take_it_is_false(tmp_path, monkeypatch):
    context, _, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_steer": True}, answer=False)
    assert context.steer_session(WORDS, origin=ORIGIN) is False
    gateway._steer_plugin_session.assert_called_once()


@pytest.mark.parametrize("entry", [{}, {"allow_gateway_steer": "true"}, {"allow_gateway_steer": False},
                                   {"allow_gateway_injection": True}, {"allow_gateway_interrupt": True}])
def test_without_the_switch_the_gateway_is_never_asked(tmp_path, monkeypatch, entry):
    context, _, gateway = _context(tmp_path, monkeypatch, entry)
    assert context.steer_session(WORDS, origin=ORIGIN) is False
    gateway._steer_plugin_session.assert_not_called()


def test_the_switch_comes_from_the_plugins_own_profile_not_the_ambient_one(tmp_path, monkeypatch):
    context, _, gateway = _context(tmp_path, monkeypatch, {})
    granting = tmp_path / "granting"
    _home(granting, {"allow_gateway_steer": True})
    token = set_hermes_home_override(granting)
    try:
        assert context.steer_session(WORDS, origin=ORIGIN) is False
    finally:
        reset_hermes_home_override(token)
    gateway._steer_plugin_session.assert_not_called()


def test_an_unreadable_config_denies(tmp_path, monkeypatch):
    context, _, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_steer": True})
    with patch("hermes_cli.plugins_steer.load_config_readonly", side_effect=OSError("config unavailable")):
        assert context.steer_session(WORDS, origin=ORIGIN) is False
    gateway._steer_plugin_session.assert_not_called()


@pytest.mark.parametrize("origin", ["slack:D1", None, ["slack", "D1"]])
def test_an_origin_that_is_not_a_mapping_is_refused(tmp_path, monkeypatch, origin):
    context, _, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_steer": True})
    assert context.steer_session(WORDS, origin=origin) is False
    gateway._steer_plugin_session.assert_not_called()


@pytest.mark.parametrize("content", ["", "   ", None, 42])
def test_content_that_is_not_text_is_refused(tmp_path, monkeypatch, content):
    context, _, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_steer": True})
    assert context.steer_session(content, origin=ORIGIN) is False
    gateway._steer_plugin_session.assert_not_called()


def test_no_live_gateway_is_false(tmp_path, monkeypatch):
    context, manager, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_steer": True})
    manager.clear_gateway_message_injector(gateway)
    assert context.steer_session(WORDS, origin=ORIGIN) is False
    manager.set_gateway_message_injector(object(), MagicMock(return_value=True))
    assert context.steer_session(WORDS, origin=ORIGIN) is False
    gateway._steer_plugin_session.assert_not_called()


def test_a_failing_gateway_is_false(tmp_path, monkeypatch):
    context, _, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_steer": True})
    gateway._steer_plugin_session.side_effect = RuntimeError("loop gone")
    assert context.steer_session(WORDS, origin=ORIGIN) is False
