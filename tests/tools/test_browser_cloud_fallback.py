"""Tests for cloud browser provider runtime fallback to local Chromium.

Covers the fallback logic in _get_session_info() when a cloud provider
is configured but fails at runtime (issue #10883).
"""
import logging
from unittest.mock import Mock

import pytest

import tools.browser_tool as browser_tool
from tools import browser_tool_session as bt_session
from tools import browser_tool_cloud as bt_cloud


def _reset_session_state(monkeypatch):
    """Clear caches so each test starts fresh."""
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(browser_tool, "_cached_cloud_provider", None)
    monkeypatch.setattr(browser_tool, "_cloud_provider_resolved", False)
    monkeypatch.setattr("tools.browser_tool_lifecycle._start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr("tools.browser_tool_lifecycle._update_session_activity", lambda t: None)


class TestCloudProviderRuntimeFallback:
    """Tests for _get_session_info cloud → local fallback."""

    def test_cloud_failure_falls_back_to_local(self, monkeypatch):
        """When cloud provider.create_session raises, fall back to local."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.side_effect = RuntimeError("401 Unauthorized")
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: None)

        session = bt_session._get_session_info("task-1")

        assert session["fallback_from_cloud"] is True
        assert "401 Unauthorized" in session["fallback_reason"]
        assert session["fallback_provider"] == "Mock"
        assert session["features"]["local"] is True
        assert session["cdp_url"] is None


    def test_no_provider_uses_local_directly(self, monkeypatch):
        """When no cloud provider is configured, local mode is used with no fallback markers."""
        _reset_session_state(monkeypatch)

        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: None)

        session = bt_session._get_session_info("task-4")

        assert session["features"]["local"] is True
        assert "fallback_from_cloud" not in session


    def test_cloud_returns_invalid_session_triggers_fallback(self, monkeypatch):
        """Cloud provider returning None or empty dict triggers fallback."""
        _reset_session_state(monkeypatch)

        provider = Mock()
        provider.create_session.return_value = None
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: None)

        session = bt_session._get_session_info("task-7")

        assert session["fallback_from_cloud"] is True
        assert "invalid session" in session["fallback_reason"]


class TestCloudFallbackToLocalSwitch:
    """``browser.cloud_fallback_to_local: false`` — a failed cloud session is an error, never a local browser."""

    @staticmethod
    def _configure(monkeypatch, browser_cfg):
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"browser": browser_cfg})

    @staticmethod
    def _forbid_local(monkeypatch):
        launched = []
        monkeypatch.setattr(bt_session, "_create_local_session",
                            lambda *a, **kw: launched.append(a) or {"session_name": "h_local", "cdp_url": None,
                                                                     "features": {"local": True}})
        return launched

    @pytest.mark.parametrize("failure", [RuntimeError("sandbox unavailable"), None])
    def test_disabled_fallback_raises_and_launches_nothing_locally(self, monkeypatch, failure):
        _reset_session_state(monkeypatch)
        self._configure(monkeypatch, {"cloud_fallback_to_local": False})
        launched = self._forbid_local(monkeypatch)
        provider = Mock()
        if failure is None:
            provider.create_session.return_value = None
        else:
            provider.create_session.side_effect = failure
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: None)

        with pytest.raises(RuntimeError) as raised:
            bt_session._get_session_info("task-nofallback")

        assert launched == []
        assert "task-nofallback" not in browser_tool._active_sessions
        assert "cloud_fallback_to_local" in str(raised.value)
        if failure is not None:
            assert str(failure) in str(raised.value)

    @pytest.mark.parametrize("value", [False, "false", "no", "off", "0", "anything-else"])
    def test_any_non_truthy_value_disables_fallback(self, monkeypatch, value):
        _reset_session_state(monkeypatch)
        self._configure(monkeypatch, {"cloud_fallback_to_local": value})
        launched = self._forbid_local(monkeypatch)
        provider = Mock()
        provider.create_session.side_effect = RuntimeError("boom")
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: None)

        with pytest.raises(RuntimeError):
            bt_session._get_session_info("task-values")
        assert launched == []

    @pytest.mark.parametrize("browser_cfg", [{}, {"cloud_fallback_to_local": True}])
    def test_absent_or_enabled_keeps_upstream_fallback(self, monkeypatch, browser_cfg):
        _reset_session_state(monkeypatch)
        self._configure(monkeypatch, browser_cfg)
        launched = self._forbid_local(monkeypatch)
        provider = Mock()
        provider.create_session.side_effect = RuntimeError("boom")
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: None)

        session = bt_session._get_session_info("task-default")

        assert len(launched) == 1
        assert session["fallback_from_cloud"] is True

    def test_a_working_provider_is_untouched_by_the_switch(self, monkeypatch):
        _reset_session_state(monkeypatch)
        self._configure(monkeypatch, {"cloud_fallback_to_local": False})
        launched = self._forbid_local(monkeypatch)
        provider = Mock()
        provider.create_session.return_value = {"session_name": "sb", "bb_session_id": "sb-1",
                                                "cdp_url": "ws://10.0.0.2:9223/devtools/browser/x",
                                                "features": {"sandbox": True}}
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: provider)
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: None)
        monkeypatch.setattr("tools.browser_tool_cdp._resolve_cdp_override", lambda url: url)
        monkeypatch.setattr("tools.browser_tool_cdp._ensure_cdp_supervisor", lambda task_id: None)

        session = bt_session._get_session_info("task-ok")

        assert launched == []
        assert session["bb_session_id"] == "sb-1"
        assert "fallback_from_cloud" not in session
