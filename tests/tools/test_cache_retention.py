"""Cache retention (``terminal.cache_max_age_hours``).

Uploads, screenshots, oversized tool results and session temp files are swept by age. One setting
decides that age for every sweep — the gateway's hourly housekeeping, the once-per-process sweeps of
CLI-only installs, the browser's own screenshot sweep — so a longer retention is not undercut by a
sweep that still assumes a day, and a shorter one shrinks how long fetched data sits on disk.
"""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import pytest

from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_constants import DEFAULT_CACHE_MAX_AGE_HOURS, cache_max_age_hours

ENV = "TERMINAL_CACHE_MAX_AGE_HOURS"

HOUSEKEEPING_CLEANUPS = (
    ("gateway.platforms.base", "cleanup_image_cache"),
    ("gateway.platforms.base", "cleanup_document_cache"),
    ("gateway.platforms.base", "cleanup_audio_cache"),
    ("gateway.platforms.base", "cleanup_video_cache"),
    ("gateway.platforms.base", "cleanup_screenshot_cache"),
    ("tools.tool_result_storage", "cleanup_spillover_cache"),
    ("tools.environments.local", "cleanup_terminal_temp_cache"),
    ("tools.bot_mode_dm", "cleanup_bot_dm_cache"),
    ("tools.bot_relay", "cleanup_bot_relay_artifacts"),
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir()
    return tmp_path / "hermes"


def _aged(path: Path, hours: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    then = time.time() - hours * 3600
    os.utime(path, (then, then))
    return path


def test_the_setting_is_a_terminal_key_bridged_like_the_cache_scope() -> None:
    assert DEFAULT_CONFIG["terminal"]["cache_max_age_hours"] == DEFAULT_CACHE_MAX_AGE_HOURS
    assert TERMINAL_CONFIG_ENV_MAP["cache_max_age_hours"] == ENV


def test_retention_keeps_the_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert cache_max_age_hours() == DEFAULT_CACHE_MAX_AGE_HOURS


def test_a_configured_retention_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV, str(DEFAULT_CACHE_MAX_AGE_HOURS * 3))
    assert cache_max_age_hours() == DEFAULT_CACHE_MAX_AGE_HOURS * 3


@pytest.mark.parametrize("raw", ["", "0", "-3", "1.5", "abc", "nan", "inf", "1e9"])
def test_an_invalid_retention_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(ENV, raw)
    assert cache_max_age_hours() == DEFAULT_CACHE_MAX_AGE_HOURS


def test_housekeeping_sweeps_every_cache_with_the_configured_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV, "6")
    seen: dict[str, float] = {}
    for module_name, name in HOUSEKEEPING_CLEANUPS:
        def record(max_age_hours, _name=name):
            seen[_name] = max_age_hours
            return 0
        monkeypatch.setattr(importlib.import_module(module_name), name, record)
    from gateway.run import _housekeeping_media_caches
    _housekeeping_media_caches()
    assert set(seen) == {name for _, name in HOUSEKEEPING_CLEANUPS}
    assert set(seen.values()) == {6}


def test_a_platform_cache_sweep_follows_the_retention_by_default(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.platforms.base import cleanup_document_cache, get_document_cache_dir
    monkeypatch.setenv(ENV, "1")
    stale = _aged(get_document_cache_dir() / "doc_stale.pdf", 2)
    fresh = _aged(get_document_cache_dir() / "doc_fresh.pdf", 0)
    assert cleanup_document_cache() == 1
    assert not stale.exists() and fresh.exists()


def test_a_longer_retention_keeps_what_a_day_old_sweep_would_drop(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.tool_result_storage import cleanup_spillover_cache, get_spillover_dir
    monkeypatch.setenv(ENV, str(DEFAULT_CACHE_MAX_AGE_HOURS * 3))
    kept = _aged(get_spillover_dir(chat_scoped=False) / "result_two_days", DEFAULT_CACHE_MAX_AGE_HOURS * 2)
    gone = _aged(get_spillover_dir(chat_scoped=False) / "result_four_days", DEFAULT_CACHE_MAX_AGE_HOURS * 4)
    assert cleanup_spillover_cache() == 1
    assert kept.exists() and not gone.exists()


def test_the_browser_screenshot_sweep_follows_the_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools import browser_tool_lifecycle
    monkeypatch.setenv(ENV, "1")
    stale = _aged(tmp_path / "browser_screenshot_old.png", 2)
    fresh = _aged(tmp_path / "browser_screenshot_new.png", 0)
    browser_tool_lifecycle._cleanup_old_screenshots(tmp_path)
    assert not stale.exists() and fresh.exists()


def test_the_terminal_temp_sweep_follows_the_retention(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.environments import local
    monkeypatch.setenv(ENV, "1")
    root = home / "cache" / "terminal"
    stale = _aged(root / "hermes_bg_old.log", 2)
    fresh = _aged(root / "hermes_bg_new.log", 0)
    local.cleanup_terminal_temp_cache()
    assert not stale.exists() and fresh.exists()
