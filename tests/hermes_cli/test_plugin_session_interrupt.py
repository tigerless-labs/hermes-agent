"""``PluginContext.interrupt_session(origin=...)``: the switch, read from the plugin's own profile, and
the hand-off to the live gateway that published the message host."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import hermes_yaml as yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

PLUGIN = "desk"
ORIGIN = {"platform": "slack", "chat_id": "D1", "chat_type": "dm", "thread_id": "170.1", "user_id": "U1"}


def _home(path, entry) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(yaml.safe_dump({"plugins": {"entries": {PLUGIN: entry}}}), encoding="utf-8")


def _context(tmp_path, monkeypatch, entry, *, answer=True):
    home = tmp_path / "own"
    _home(home, entry)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    gateway = SimpleNamespace(_interrupt_plugin_session=MagicMock(return_value=answer))
    manager.set_gateway_message_injector(gateway, MagicMock(return_value=True))
    return PluginContext(PluginManifest(name=PLUGIN, key=PLUGIN, source="user"), manager), manager, gateway


def test_the_live_gateway_is_asked_with_the_plugins_own_home(tmp_path, monkeypatch):
    context, manager, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_interrupt": True})
    other = tmp_path / "other"
    other.mkdir()
    token = set_hermes_home_override(other)
    try:
        assert context.interrupt_session(origin=ORIGIN) is True
    finally:
        reset_hermes_home_override(token)
    gateway._interrupt_plugin_session.assert_called_once_with(
        origin=ORIGIN, plugin_home=manager.home_path, plugin_id=PLUGIN)


def test_nothing_running_is_false(tmp_path, monkeypatch):
    context, _, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_interrupt": True}, answer=False)
    assert context.interrupt_session(origin=ORIGIN) is False
    gateway._interrupt_plugin_session.assert_called_once()


@pytest.mark.parametrize("entry", [{}, {"allow_gateway_interrupt": "true"}, {"allow_gateway_interrupt": False},
                                   {"allow_gateway_injection": True}])
def test_without_the_switch_the_gateway_is_never_asked(tmp_path, monkeypatch, entry):
    context, _, gateway = _context(tmp_path, monkeypatch, entry)
    assert context.interrupt_session(origin=ORIGIN) is False
    gateway._interrupt_plugin_session.assert_not_called()


def test_the_switch_comes_from_the_plugins_own_profile_not_the_ambient_one(tmp_path, monkeypatch):
    context, _, gateway = _context(tmp_path, monkeypatch, {})
    granting = tmp_path / "granting"
    _home(granting, {"allow_gateway_interrupt": True})
    token = set_hermes_home_override(granting)
    try:
        assert context.interrupt_session(origin=ORIGIN) is False
    finally:
        reset_hermes_home_override(token)
    gateway._interrupt_plugin_session.assert_not_called()


def test_an_unreadable_config_denies(tmp_path, monkeypatch):
    context, _, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_interrupt": True})
    with patch("hermes_cli.plugins_interrupt.load_config_readonly", side_effect=OSError("config unavailable")):
        assert context.interrupt_session(origin=ORIGIN) is False
    gateway._interrupt_plugin_session.assert_not_called()


@pytest.mark.parametrize("origin", ["slack:D1", None, ["slack", "D1"]])
def test_an_origin_that_is_not_a_mapping_is_refused(tmp_path, monkeypatch, origin):
    context, _, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_interrupt": True})
    assert context.interrupt_session(origin=origin) is False
    gateway._interrupt_plugin_session.assert_not_called()


def test_no_live_gateway_is_false(tmp_path, monkeypatch):
    context, manager, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_interrupt": True})
    manager.clear_gateway_message_injector(gateway)
    assert context.interrupt_session(origin=ORIGIN) is False
    manager.set_gateway_message_injector(object(), MagicMock(return_value=True))  # a host that cannot stop
    assert context.interrupt_session(origin=ORIGIN) is False
    gateway._interrupt_plugin_session.assert_not_called()


def test_a_failing_gateway_is_false(tmp_path, monkeypatch):
    context, _, gateway = _context(tmp_path, monkeypatch, {"allow_gateway_interrupt": True})
    gateway._interrupt_plugin_session.side_effect = RuntimeError("loop gone")
    assert context.interrupt_session(origin=ORIGIN) is False
