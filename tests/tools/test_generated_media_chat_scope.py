"""Generated media under chat-scoped caches (``terminal.docker_cache_scope: chat``).

Image and video backends materialise their output in ``$HERMES_HOME/cache/<kind>``. With caches scoped by
chat, that output lands only in the current chat's own cache, where only that chat can deliver, read or
mount it. A context outside any chat writes nothing unless it borrows a chat (a background run working
for that chat); the session's own chat wins over a borrowed one. Shared caches keep the upstream path.
"""

from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import provider_media
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_constants import borrowed_chat_cache_scope, chat_scope_slug

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
_B64 = base64.b64encode(_PNG).decode()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    (tmp_path / "hermes").mkdir()
    return tmp_path / "hermes"


@pytest.fixture
def scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERMINAL_DOCKER_CACHE_SCOPE", "chat")


class _Chat:
    def __init__(self, platform: str, chat_id: str) -> None:
        self.platform, self.chat_id = platform, chat_id

    def __enter__(self):
        self.tokens = set_session_vars(platform=self.platform, chat_id=self.chat_id)
        return self

    def __exit__(self, *_):
        clear_session_vars(self.tokens)


def _chat_cache(home: Path, kind: str, chat_id: str) -> Path:
    return home / "cache" / kind / "chats" / chat_scope_slug("slack", chat_id)


def _written_files(home: Path) -> list[Path]:
    return [path for path in (home / "cache").rglob("*") if path.is_file()] if (home / "cache").exists() else []


def test_shared_caches_keep_the_upstream_path(home: Path) -> None:
    with _Chat("slack", "C1"):
        saved = provider_media.save_b64("images", _B64, prefix="meta", extension="png")
    assert saved.parent == home / "cache" / "images"


@pytest.mark.parametrize("kind", ["images", "videos"])
def test_generated_media_land_in_the_chats_own_cache(home: Path, scoped: None, kind: str) -> None:
    with _Chat("slack", "C1"):
        saved = provider_media.save_b64(kind, _B64, prefix="gen", extension="png")
    assert saved.parent == _chat_cache(home, kind, "C1")
    assert saved.read_bytes() == _PNG


def test_redteam_a_context_outside_any_chat_writes_nothing(home: Path, scoped: None) -> None:
    with pytest.raises(PermissionError):
        provider_media.save_b64("images", _B64, prefix="gen", extension="png")
    assert _written_files(home) == []


def test_a_borrowed_chat_scopes_a_context_outside_any_chat(home: Path, scoped: None) -> None:
    with borrowed_chat_cache_scope("slack", "C9"):
        saved = provider_media.save_b64("images", _B64, prefix="gen", extension="png")
    assert saved.parent == _chat_cache(home, "images", "C9")
    with pytest.raises(PermissionError):
        provider_media.save_b64("images", _B64, prefix="gen", extension="png")


def test_the_sessions_own_chat_beats_a_borrowed_one(home: Path, scoped: None) -> None:
    with borrowed_chat_cache_scope("slack", "C9"), _Chat("slack", "C1"):
        saved = provider_media.save_b64("images", _B64, prefix="gen", extension="png")
    assert saved.parent == _chat_cache(home, "images", "C1")


def test_a_borrowed_chat_has_no_effect_on_shared_caches(home: Path) -> None:
    with borrowed_chat_cache_scope("slack", "C9"):
        saved = provider_media.save_b64("images", _B64, prefix="gen", extension="png")
    assert saved.parent == home / "cache" / "images"


def test_redteam_another_chat_can_neither_deliver_nor_read_a_generated_image(
        home: Path, scoped: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.platforms import base
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (home / "cache" / "images",))
    monkeypatch.setattr(base, "_profile_cache_roots", lambda: [])
    with _Chat("slack", "C1"):
        saved = provider_media.save_b64("images", _B64, prefix="gen", extension="png").resolve()
        mine = [root.resolve() for root in base._media_delivery_allowed_roots()]
    with _Chat("slack", "C2"):
        theirs = [root.resolve() for root in base._media_delivery_allowed_roots()]
        denied = [path.resolve() for path in base._media_delivery_denied_paths()]
    assert any(root in saved.parents for root in mine)
    assert not any(root in saved.parents for root in theirs)
    assert any(path in saved.parents for path in denied)


@pytest.mark.asyncio
async def test_redteam_vision_in_another_chat_cannot_read_a_generated_image(
        home: Path, scoped: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.image_source as image_source
    monkeypatch.setattr(image_source, "_get_active_env", lambda task_id: None)
    monkeypatch.setattr(image_source, "_ensure_container_env", lambda task_id: None)
    with _Chat("slack", "C1"):
        saved = provider_media.save_b64("images", _B64, prefix="gen", extension="png")
        read = await image_source.resolve_image_source(str(saved), image_source.ResolveContext(task_id="t1"))
        assert read.data == _PNG
    with _Chat("slack", "C2"), pytest.raises(image_source.SourceNotFound):
        await image_source.resolve_image_source(str(saved), image_source.ResolveContext(task_id="t2"))


def test_the_sandbox_sees_the_generated_image_in_the_chats_mounted_cache(
        home: Path, scoped: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools import image_generation_tool
    from tools.credential_files import get_cache_directory_mounts

    env = SimpleNamespace(agent_visible_cache_base=lambda: "/root/.hermes")
    monkeypatch.setattr(image_generation_tool, "_active_terminal_env", lambda task_id: env)
    with _Chat("slack", "C1"):
        saved = provider_media.save_b64("images", _B64, prefix="gen", extension="png")
        mounts = get_cache_directory_mounts()
        result = json.loads(image_generation_tool._postprocess_image_generate_result(
            json.dumps({"success": True, "image": str(saved)}), task_id="t1"))
    mount = next(m for m in mounts if Path(m["host_path"]) in saved.parents)
    assert result["agent_visible_image"] == f"{mount['container_path']}/{saved.name}"


def test_a_meta_image_outside_any_chat_is_an_error_not_a_file(home: Path, scoped: None,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    meta_plugin = importlib.import_module("plugins.image_gen.meta-ai")
    monkeypatch.setenv("META_MODEL_API_KEY", "test-key")
    fake_client = MagicMock()
    fake_client.images.generate.return_value = SimpleNamespace(
        data=[SimpleNamespace(b64_json=_B64, url=None, revised_prompt=None)])
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    with patch.dict("sys.modules", {"openai": fake_openai}):
        result = meta_plugin.MetaImageGenProvider().generate("a cat")
    assert result["success"] is False
    assert result["error_type"] == "io_error"
    assert _written_files(home) == []
