"""``compression.blocked_notices``: a deployment keeps the compression-blocked warning out of chat.

The warning tells the reader to run ``/new`` or ``/compress``; on a shared deployment the people in the chat
neither own the session nor can act on it. Switched off, chat surfaces drop it and the gateway log keeps it.
Unset or true keeps upstream's always-visible failure notice.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

import gateway.run as gateway_run
from agent.conversation_compression import CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE
from gateway.run import _prepare_gateway_status_message

CHAT_PLATFORMS = ["slack", "telegram", "discord", "matrix"]

BLOCKED_WARNINGS = [
    CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(tokens=134_340, threshold=85_000, reason="structural_backoff:291"),
    CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(tokens=90_000, threshold=85_000, reason="cooldown:42"),
    CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(tokens=200_000, threshold=100_000, reason="ineffective"),
]

OTHER_FAILURE_NOTICES = [
    "⚠ Compression aborted: auth failure. No messages were dropped — conversation continues unchanged. "
    "Run /compress to retry, or /new to start a fresh session.",
    "Compression aborted: 30 messages preserved",
    "Here is the answer you asked for.",
]


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: object) -> None:
    (tmp_path / "config.yaml").write_text(config if isinstance(config, str) else yaml.safe_dump(config),
                                          encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)


@pytest.fixture
def switched_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(tmp_path, monkeypatch, {"compression": {"blocked_notices": False}})


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
@pytest.mark.parametrize("message", BLOCKED_WARNINGS, ids=lambda m: m[-40:])
def test_switched_off_chat_surfaces_drop_the_blocked_warning(switched_off, platform, message):
    assert _prepare_gateway_status_message(platform, "warn", message) is None


@pytest.mark.parametrize("message", BLOCKED_WARNINGS, ids=lambda m: m[-40:])
def test_switched_off_the_log_keeps_the_blocked_warning(switched_off, caplog, message):
    with caplog.at_level(logging.INFO):
        _prepare_gateway_status_message("slack", "warn", message)
    kept = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert any(message in record.getMessage() for record in kept)


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
@pytest.mark.parametrize("message", OTHER_FAILURE_NOTICES, ids=lambda m: m[:32])
def test_switched_off_other_messages_are_untouched(switched_off, platform, message):
    assert _prepare_gateway_status_message(platform, "warn", message) == message


@pytest.mark.parametrize("message", BLOCKED_WARNINGS, ids=lambda m: m[-40:])
def test_switched_off_raw_text_surfaces_still_get_it(switched_off, message):
    assert _prepare_gateway_status_message("local", "warn", message) == message


@pytest.mark.parametrize("config", [
    {},
    {"compression": {}},
    {"compression": {"blocked_notices": True}},
    {"compression": {"blocked_notices": "off"}},
    {"compression": {"blocked_notices": 0}},
    {"compression": "not-a-mapping"},
    "::: not yaml :::",
], ids=["no-config", "no-key", "true", "string", "zero", "bad-section", "bad-yaml"])
@pytest.mark.parametrize("message", BLOCKED_WARNINGS[:1], ids=lambda m: m[-40:])
def test_anything_but_false_keeps_upstreams_visible_warning(tmp_path, monkeypatch, config, message):
    _configure(tmp_path, monkeypatch, config)
    assert _prepare_gateway_status_message("slack", "warn", message) == message
