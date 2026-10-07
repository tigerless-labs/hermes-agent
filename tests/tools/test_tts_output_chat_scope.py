"""With chat-scoped caches (``terminal.docker_cache_scope: chat``) the speech a model asks for lands in the
current chat's audio cache — the exchange area allocates the place, the model's path keeps only its file name."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_constants import chat_scope_slug
from tools.tts_tool import _resolve_output_base


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    (tmp_path / "hermes").mkdir()
    return tmp_path / "hermes"


@pytest.fixture
def in_chat():
    tokens = set_session_vars(platform="slack", chat_id="C1")
    yield home
    clear_session_vars(tokens)


@pytest.mark.parametrize("asked", ["/etc/cron.d/evil.mp3", "/var/lib/lara/hermes/plugins/evil.mp3", "evil.mp3",
                                   "~/.ssh/evil.mp3"])
def test_with_chat_scoped_caches_a_model_path_lands_in_the_chats_audio_cache(home, in_chat, monkeypatch, asked):
    monkeypatch.setenv("TERMINAL_DOCKER_CACHE_SCOPE", "chat")
    path, error = _resolve_output_base(asked, "edge", None, False)
    assert error is None
    assert path == home / "cache" / "audio" / "chats" / chat_scope_slug("slack", "C1") / "evil.mp3"


def test_with_chat_scoped_caches_a_session_outside_any_chat_writes_nowhere(home, monkeypatch):
    monkeypatch.setenv("TERMINAL_DOCKER_CACHE_SCOPE", "chat")
    path, error = _resolve_output_base("/tmp/out.mp3", "edge", None, False)
    assert path is None and "chat" in error


def test_with_chat_scoped_caches_traversal_is_still_refused(home, in_chat, monkeypatch):
    monkeypatch.setenv("TERMINAL_DOCKER_CACHE_SCOPE", "chat")
    path, error = _resolve_output_base("audio/../../etc/x.mp3", "edge", None, False)
    assert path is None and "traversal" in error


def test_shared_caches_keep_the_path_the_caller_gave(home, in_chat, tmp_path):
    asked = tmp_path / "elsewhere" / "out.mp3"
    path, error = _resolve_output_base(str(asked), "edge", None, False)
    assert error is None and path == asked
