"""Off switch for local browser sessions (fork addition; on by default, as upstream)."""

import pytest

from tools import browser_local_gate as gate
from tools import browser_tool_session as session

TASK = "slack-D0C8FU30NCF"


@pytest.fixture
def browser_config(monkeypatch):
    def configure(**browser):
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"browser": browser})
    monkeypatch.setattr(session._real_profile, "_real_profile_cdp", lambda: (None, None))
    monkeypatch.setattr(session._cdp, "_get_cdp_override_raw", lambda: "")
    return configure


class FailingProvider:
    def create_session(self, task_id):
        raise RuntimeError("sandbox browser did not start")


def test_by_default_a_local_session_starts_as_upstream_starts_it(browser_config, monkeypatch):
    browser_config()
    monkeypatch.setattr(session._cloud, "_get_cloud_provider", lambda: None)
    assert session._create_session_for_key(TASK, force_local=False)["features"]["local"] is True


@pytest.mark.parametrize("off", [False, "false", "no", 0])
def test_any_spelling_of_off_turns_local_sessions_off(browser_config, off):
    browser_config(local_sessions=off)
    assert not gate.local_sessions_allowed()


def test_redteam_with_local_sessions_off_a_missing_provider_starts_no_browser_on_the_host(browser_config, monkeypatch):
    browser_config(local_sessions=False)
    monkeypatch.setattr(session._cloud, "_get_cloud_provider", lambda: None)
    with pytest.raises(gate.LocalBrowserRefused):
        session._create_session_for_key(TASK, force_local=False)


def test_redteam_with_local_sessions_off_a_failed_provider_does_not_fall_back_to_the_host(browser_config, monkeypatch):
    browser_config(local_sessions=False, cloud_fallback_to_local=True)
    monkeypatch.setattr(session._cloud, "_get_cloud_provider", lambda: FailingProvider())
    with pytest.raises(RuntimeError) as refused:
        session._create_session_for_key(TASK, force_local=False)
    assert isinstance(refused.value.__cause__, RuntimeError) or isinstance(refused.value, gate.LocalBrowserRefused)
    assert "local" in str(refused.value).lower()


def test_redteam_with_local_sessions_off_the_private_url_sidecar_starts_no_browser_on_the_host(browser_config):
    browser_config(local_sessions=False)
    with pytest.raises(gate.LocalBrowserRefused):
        session._create_session_for_key(f"{TASK}::local", force_local=True)
