"""Caches scoped by chat (``terminal.docker_cache_scope: chat``).

Uploads, screenshots, web pages and oversized tool results live in host cache dirs that are
bind-mounted read-only into every sandbox container. With the switch on, every chat writes and
mounts only ``<cache dir>/chats/<platform>-<chat>``: a container never sees another chat's files,
media delivery refuses them, and a session outside any chat mounts no cache at all.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_constants import chat_scope_slug, get_hermes_dir
from tools.credential_files import (
    from_agent_visible_cache_path, get_cache_directory_mounts, map_cache_path_to_container)

CONTAINER_BASE = "/root/.hermes"


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


def _hosts(mounts) -> list[Path]:
    return [Path(m["host_path"]) for m in mounts]


def test_caches_stay_shared_unless_switched_on(home: Path) -> None:
    with _Chat("slack", "C1"):
        assert get_hermes_dir("cache/documents", "document_cache") == home / "cache" / "documents"
        assert home / "cache" / "documents" in _hosts(get_cache_directory_mounts())


def test_a_chat_writes_its_cache_files_under_its_own_directory(home: Path, scoped: None) -> None:
    with _Chat("slack", "C1"):
        assert get_hermes_dir("cache/documents", "document_cache") == (
            home / "cache" / "documents" / "chats" / chat_scope_slug("slack", "C1"))
        assert get_hermes_dir("skills", "skills") == home / "skills"


def test_a_chat_container_mounts_only_its_own_cache(home: Path, scoped: None) -> None:
    with _Chat("slack", "C1"):
        mounts = get_cache_directory_mounts()
    slug = chat_scope_slug("slack", "C1")
    assert mounts and all(Path(m["host_path"]).name == slug and Path(m["host_path"]).parent.name == "chats"
                          for m in mounts)
    assert all(m["container_path"].endswith(f"/chats/{slug}") for m in mounts)


def test_redteam_two_chats_never_mount_each_others_cache(home: Path, scoped: None) -> None:
    with _Chat("slack", "C1"):
        first = set(_hosts(get_cache_directory_mounts()))
    with _Chat("slack", "C2"):
        second = set(_hosts(get_cache_directory_mounts()))
    assert first and second and not first & second
    assert not any(a in b.parents or b in a.parents for a in first for b in second)


def test_redteam_a_session_outside_any_chat_mounts_no_cache(home: Path, scoped: None) -> None:
    with _Chat("cron", ""):
        assert get_cache_directory_mounts() == []
    assert get_cache_directory_mounts() == []


@pytest.mark.parametrize("chat_id", ["../../etc", "C1/../C2", "..", "/abs", "C1\x00x"])
def test_redteam_a_chat_id_cannot_escape_the_chats_directory(home: Path, scoped: None, chat_id: str) -> None:
    with _Chat("slack", chat_id):
        directory = get_hermes_dir("cache/documents", "document_cache")
    assert directory.resolve().parent == (home / "cache" / "documents" / "chats").resolve()


def test_host_paths_translate_the_same_inside_and_outside_a_chat(home: Path, scoped: None) -> None:
    host = home / "cache" / "documents" / "chats" / chat_scope_slug("slack", "C1") / "doc_x.pdf"
    outside = map_cache_path_to_container(str(host))
    with _Chat("slack", "C1"):
        inside = map_cache_path_to_container(str(host))
        mounted = [m["container_path"] for m in get_cache_directory_mounts()]
    assert outside == inside and any(inside.startswith(root + "/") for root in mounted)


def test_redteam_another_chats_container_path_never_maps_back_to_a_host_file(home: Path, scoped: None) -> None:
    other = f"{CONTAINER_BASE}/cache/documents/chats/{chat_scope_slug('slack', 'C2')}/doc_y.pdf"
    mine = f"{CONTAINER_BASE}/cache/documents/chats/{chat_scope_slug('slack', 'C1')}/doc_x.pdf"
    with _Chat("slack", "C1"):
        assert from_agent_visible_cache_path(other) == other
        assert from_agent_visible_cache_path(mine).startswith(str(home))


def _cached_upload(home: Path, name: str) -> Path:
    path = home / "cache" / "documents" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF")
    return path


def test_an_upload_moves_into_its_chats_cache_when_it_arrives(home: Path, scoped: None) -> None:
    from gateway.cache_scope import scope_inbound_media
    upload = _cached_upload(home, "doc_abc_report.pdf")
    outside = home.parent / "elsewhere.txt"
    outside.write_text("x")
    event = SimpleNamespace(media_urls=[str(upload), str(outside)])
    scope_inbound_media(event, SimpleNamespace(platform=SimpleNamespace(value="slack"), chat_id="C1"))
    moved = Path(event.media_urls[0])
    assert moved.parent == home / "cache" / "documents" / "chats" / chat_scope_slug("slack", "C1")
    assert moved.read_bytes() == b"%PDF" and not upload.exists()
    assert event.media_urls[1] == str(outside)


def test_uploads_stay_put_when_caches_are_shared(home: Path) -> None:
    from gateway.cache_scope import scope_inbound_media
    upload = _cached_upload(home, "doc_abc_report.pdf")
    event = SimpleNamespace(media_urls=[str(upload)])
    scope_inbound_media(event, SimpleNamespace(platform=SimpleNamespace(value="slack"), chat_id="C1"))
    assert event.media_urls == [str(upload)] and upload.exists()


def test_redteam_media_delivery_refuses_another_chats_recent_upload(home: Path, scoped: None,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.platforms import base
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (home / "cache" / "documents",))
    monkeypatch.setattr(base, "_profile_cache_roots", lambda: [])
    mine = home / "cache" / "documents" / "chats" / chat_scope_slug("slack", "C1") / "doc_mine.pdf"
    theirs = home / "cache" / "documents" / "chats" / chat_scope_slug("slack", "C2") / "doc_theirs.pdf"
    for path in (mine, theirs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF")
    with _Chat("slack", "C1"):
        allowed = [root.resolve() for root in base._media_delivery_allowed_roots()]
        denied = [path.resolve() for path in base._media_delivery_denied_paths()]
    assert any(root in mine.resolve().parents for root in allowed)
    assert not any(root in theirs.resolve().parents for root in allowed)
    assert any(path in theirs.resolve().parents for path in denied)


def test_cleanup_sweeps_every_chats_cache(home: Path, scoped: None) -> None:
    from gateway.platforms.base import cleanup_document_cache
    from tools.tool_result_storage import cleanup_spillover_cache
    old = time.time() - 48 * 3600
    stale = []
    for sub in ("documents", "spillover"):
        path = home / "cache" / sub / "chats" / chat_scope_slug("slack", "C1") / f"stale_{sub}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        os.utime(path, (old, old))
        stale.append(path)
    assert cleanup_document_cache(max_age_hours=24) >= 1 and cleanup_spillover_cache(max_age_hours=24) >= 1
    assert not any(path.exists() for path in stale)


def test_the_memory_partition_and_the_cache_scope_name_a_chat_the_same_way(home: Path, monkeypatch) -> None:
    from tools import memory_tool
    monkeypatch.setattr(memory_tool, "get_memory_dir", lambda: home / "memories")
    partition = memory_tool.memory_partition({"partition_by_chat": True}, "slack", "C1", "group")
    assert partition.directory.name == chat_scope_slug("slack", "C1")


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")


def _image_in_chat_cache(platform: str, chat_id: str, name: str) -> Path:
    with _Chat(platform, chat_id):
        path = get_hermes_dir("cache/images", "image_cache") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG)
    return path


@pytest.mark.asyncio
async def test_vision_reads_only_the_current_chats_cache_off_the_host(home: Path, scoped: None,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.image_source as image_source

    mine, theirs = _image_in_chat_cache("slack", "C1", "mine.png"), _image_in_chat_cache("slack", "C2", "theirs.png")
    monkeypatch.setattr(image_source, "_get_active_env", lambda task_id: None)
    monkeypatch.setattr(image_source, "_ensure_container_env", lambda task_id: None)
    with _Chat("slack", "C1"):
        read = await image_source.resolve_image_source(str(mine), image_source.ResolveContext(task_id="t1"))
        assert read.origin == "file" and read.data == _PNG
        with pytest.raises(image_source.SourceNotFound):
            await image_source.resolve_image_source(str(theirs), image_source.ResolveContext(task_id="t1"))
    with pytest.raises(image_source.SourceNotFound):
        await image_source.resolve_image_source(str(mine), image_source.ResolveContext(task_id="t1"))


def test_delivery_and_vision_narrow_the_caches_by_the_same_rule(home: Path, scoped: None) -> None:
    from gateway.platforms import base
    from hermes_constants import chat_scoped_roots

    roots = [home / "cache" / "images", home / "cache" / "documents"]
    with _Chat("slack", "C1"):
        assert chat_scoped_roots(roots) == [root / "chats" / chat_scope_slug("slack", "C1") for root in roots]
        assert base._chat_scoped_cache_roots is chat_scoped_roots
    assert chat_scoped_roots(roots) == []
