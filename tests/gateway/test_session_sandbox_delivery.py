"""Reply delivery validates as the session that produced the reply, and strict delivery can fetch from
that session's own sandbox (``gateway.trust_session_sandbox``).

A turn's reply is delivered after the turn's session vars were cleared. Validation then saw no chat, so a
chat-scoped cache allowed nothing, and no session, so the sandbox fetch looked for the shared ``default``
container. Strict deployments also refused every sandbox fetch: files an agent built in its per-session
container never reached the chat.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

import tools.terminal_tool as terminal_tool
import tools.terminal_tool_backends as backends
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, get_session_env
from hermes_constants import chat_scope_slug
from tools.environments.base import FileFetchError

CHAT, OTHER_CHAT = "C1", "C2"
SESSION_KEY = "agent:main:slack:channel:C1:U1"
OWN, OTHER = "20261005_213333_aaaaaaaa", "20261005_213334_bbbbbbbb"
REPORT = "/tmp/report.xlsx"
SESSION_VARS = ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_KEY", "HERMES_SESSION_ID")


class _Sandbox:
    """A live sandbox: a small filesystem, a symlink table, and every path it was asked to hand over."""

    def __init__(self, files: dict[str, bytes], *, links: dict[str, str] | None = None,
                 session_scoped: bool = True, home: str | None = "/root") -> None:
        self.files, self.links = dict(files), dict(links or {})
        self._session_scoped, self._remote_home = session_scoped, home
        self.fetched: list[str] = []

    def fetch_realpath(self, remote_path: str) -> str:
        return self.links.get(remote_path, remote_path)

    def fetch_file(self, remote_path: str, local_dest: Path, *, max_bytes: int) -> None:
        self.fetched.append(remote_path)
        if remote_path not in self.files:
            raise FileFetchError("missing")
        Path(local_dest).write_bytes(self.files[remote_path])


class _Store:
    def __init__(self, ids: dict[str, str]) -> None:
        self.ids, self.lookups = ids, 0

    def peek_session_id(self, session_key: str) -> str | None:
        self.lookups += 1
        return self.ids.get(session_key)

    def load_transcript(self, session_id: str) -> list:
        return []


class _SlackAdapter(BasePlatformAdapter):
    def __init__(self, store: _Store) -> None:
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.set_session_store(store)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "channel"}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from gateway.platforms import base
    hermes = tmp_path / "hermes"
    hermes.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setenv("TERMINAL_DOCKER_CACHE_SCOPE", "chat")
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
    monkeypatch.delenv("HERMES_MEDIA_TRUST_SESSION_SANDBOX", raising=False)
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (hermes / "cache" / "documents",))
    monkeypatch.setattr(base, "_profile_cache_roots", lambda: [])
    return hermes


@pytest.fixture
def sandboxes(monkeypatch: pytest.MonkeyPatch) -> dict:
    registry: dict = {}
    monkeypatch.setattr(terminal_tool, "_active_environments", registry)
    return registry


@pytest.fixture
def strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")


@pytest.fixture
def trusted(strict: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_MEDIA_TRUST_SESSION_SANDBOX", "1")


def _chat_documents(home: Path, chat: str = CHAT) -> Path:
    return home / "cache" / "documents" / "chats" / chat_scope_slug("slack", chat)


def _deliver(reply: str, ids: dict[str, str] | _Store, chat: str = CHAT) -> tuple[list[str], dict[str, str]]:
    """Deliver ``reply`` the way the gateway does once a turn has ended (its session vars cleared);
    returns the delivered host paths and the session vars right after delivery."""
    adapter = _SlackAdapter(ids if isinstance(ids, _Store) else _Store(ids))
    event = MessageEvent(text="make me a sheet",
                         source=SessionSource(platform=Platform.SLACK, chat_id=chat, chat_type="channel", user_id="U1"))

    async def after_the_turn():
        clear_session_vars([])
        extracted = await adapter._extract_response_content(reply, event, SESSION_KEY, is_ephemeral_response=False)
        return [path for path, _ in extracted.media_files], {name: get_session_env(name) for name in SESSION_VARS}

    return asyncio.run(after_the_turn())


def test_a_reply_delivers_its_own_chats_cached_file_after_the_turn_cleared_its_session(home: Path, strict: None) -> None:
    mine, theirs = _chat_documents(home) / "doc_mine.pdf", _chat_documents(home, OTHER_CHAT) / "doc_theirs.pdf"
    for path in (mine, theirs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF")

    delivered, _ = _deliver(f"Here: MEDIA:{mine} and MEDIA:{theirs}", {SESSION_KEY: OWN})

    assert delivered == [str(mine.resolve())]


def test_a_reply_fetches_from_the_sandbox_of_the_session_that_produced_it(home: Path, sandboxes: dict) -> None:
    sandboxes[OWN] = _Sandbox({REPORT: b"own"})
    sandboxes["default"] = _Sandbox({REPORT: b"shared"}, session_scoped=False)

    delivered, _ = _deliver(f"MEDIA:{REPORT}", {SESSION_KEY: OWN})

    assert [Path(path).read_bytes() for path in delivered] == [b"own"]
    assert sandboxes["default"].fetched == []


def test_delivery_leaves_the_cleared_session_as_it_found_it(home: Path, sandboxes: dict,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_SESSION_ID", OTHER)
    sandboxes[OWN] = _Sandbox({REPORT: b"own"})

    _, after = _deliver(f"MEDIA:{REPORT}", {SESSION_KEY: OWN})

    assert after == {name: "" for name in SESSION_VARS}
    assert os.environ["HERMES_SESSION_ID"] == OTHER


def test_the_session_store_is_asked_only_when_a_file_must_come_from_a_sandbox_and_then_once(
        home: Path, sandboxes: dict, trusted: None) -> None:
    cached = _chat_documents(home) / "doc_mine.pdf"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"%PDF")
    sandboxes[OWN] = _Sandbox({REPORT: b"own", "/root/out/summary.csv": b"a,b"})
    store = _Store({SESSION_KEY: OWN})

    _deliver("Plain words, no file.", store)
    _deliver(f"MEDIA:{cached}", store)
    assert store.lookups == 0

    delivered, _ = _deliver(f"MEDIA:{REPORT} MEDIA:/root/out/summary.csv", store)
    assert len(delivered) == 2 and store.lookups == 1


def test_strict_delivery_fetches_nothing_from_a_sandbox_unless_switched_on(home: Path, sandboxes: dict,
                                                                           strict: None) -> None:
    sandboxes[OWN] = _Sandbox({REPORT: b"own"})

    delivered, _ = _deliver(f"MEDIA:{REPORT}", {SESSION_KEY: OWN})

    assert delivered == []
    assert sandboxes[OWN].fetched == []


def test_switched_on_strict_delivery_brings_the_sessions_own_file_into_its_chats_cache(
        home: Path, sandboxes: dict, trusted: None) -> None:
    sandboxes[OWN] = _Sandbox({REPORT: b"own", "/root/out/summary.csv": b"a,b"})

    delivered, _ = _deliver(f"MEDIA:{REPORT} MEDIA:/root/out/summary.csv", {SESSION_KEY: OWN})

    assert [Path(path).read_bytes() for path in delivered] == [b"own", b"a,b"]
    assert all(_chat_documents(home).resolve() in Path(path).parents for path in delivered)
    assert [Path(path).name for path in delivered] == [Path(REPORT).name, "summary.csv"]


def test_a_fetched_file_keeps_the_name_a_person_sees_and_loses_what_could_disguise_it(
        home: Path, sandboxes: dict, trusted: None) -> None:
    readable = "/tmp/报告 (终版) 2026-10.xlsx"
    sandboxes[OWN] = _Sandbox({readable: b"a", "/tmp/invoice‮xcod.pdf": b"b", "/tmp/plan​ .csv": b"c"},
                              links={"/tmp/invoice.pdf": "/tmp/invoice‮xcod.pdf",
                                     "/tmp/plan.csv": "/tmp/plan​ .csv"})

    delivered, _ = _deliver(f"MEDIA:{readable} MEDIA:/tmp/invoice.pdf MEDIA:/tmp/plan.csv", {SESSION_KEY: OWN})

    names = [Path(path).name for path in delivered]
    assert names[0] == Path(readable).name
    assert all(char.isprintable() for name in names for char in name)
    assert [name.replace("_", "") for name in names[1:]] == ["invoicexcod.pdf", "plan.csv"]


def test_two_copies_of_one_name_never_overwrite_each_other(home: Path, sandboxes: dict, trusted: None) -> None:
    sandboxes[OWN] = _Sandbox({REPORT: b"first"})
    [first], _ = _deliver(f"MEDIA:{REPORT}", {SESSION_KEY: OWN})
    sandboxes[OWN].files[REPORT] = b"second"
    [second], _ = _deliver(f"MEDIA:{REPORT}", {SESSION_KEY: OWN})

    assert Path(first).name == Path(second).name and first != second
    assert (Path(first).read_bytes(), Path(second).read_bytes()) == (b"first", b"second")


def test_a_fetch_that_fails_leaves_no_folder_behind(home: Path, sandboxes: dict, trusted: None) -> None:
    sandboxes[OWN] = _Sandbox({})

    delivered, _ = _deliver(f"MEDIA:{REPORT}", {SESSION_KEY: OWN})

    assert delivered == [] and sandboxes[OWN].fetched == [REPORT]
    assert not list(_chat_documents(home).glob("remote_*")) if _chat_documents(home).exists() else True


def test_the_cache_sweep_removes_old_fetched_copies_and_their_folders(home: Path, sandboxes: dict,
                                                                      trusted: None) -> None:
    import contextvars
    import time

    from gateway.platforms.base import cleanup_document_cache
    sandboxes[OWN] = _Sandbox({REPORT: b"old", "/tmp/new.csv": b"new"})
    [old], _ = _deliver(f"MEDIA:{REPORT}", {SESSION_KEY: OWN})
    [new], _ = _deliver("MEDIA:/tmp/new.csv", {SESSION_KEY: OWN})
    stale = time.time() - 3 * 3600
    os.utime(old, (stale, stale))

    removed = contextvars.Context().run(cleanup_document_cache, max_age_hours=1)

    assert removed == 1
    assert not Path(old).parent.exists() and Path(new).exists()


def test_redteam_switched_on_strict_delivery_never_reads_another_sessions_sandbox(
        home: Path, sandboxes: dict, trusted: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_SESSION_ID", OTHER)
    sandboxes[OTHER] = _Sandbox({REPORT: b"other"})

    delivered, _ = _deliver(f"MEDIA:{REPORT}", {SESSION_KEY: OWN})

    assert delivered == []
    assert sandboxes[OTHER].fetched == []


def test_redteam_strict_delivery_outside_a_reply_never_takes_its_session_from_the_process_env(
        home: Path, sandboxes: dict, trusted: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.media_fetch import fetch_remote_media
    from gateway.session_context import reset_session_vars
    monkeypatch.setenv("HERMES_SESSION_ID", OTHER)
    sandboxes[OTHER] = _Sandbox({REPORT: b"other"})

    async def unbound():
        reset_session_vars()
        return fetch_remote_media(REPORT)

    assert asyncio.run(unbound()) is None
    assert sandboxes[OTHER].fetched == []


def test_redteam_without_a_session_id_strict_delivery_fetches_from_no_sandbox(
        home: Path, sandboxes: dict, trusted: None) -> None:
    sandboxes["default"] = _Sandbox({REPORT: b"shared"})
    sandboxes[""] = _Sandbox({REPORT: b"blank"})

    delivered, _ = _deliver(f"MEDIA:{REPORT}", {})

    assert delivered == []
    assert sandboxes["default"].fetched == sandboxes[""].fetched == []


def test_redteam_strict_delivery_never_trusts_a_sandbox_shared_beyond_the_session(
        home: Path, sandboxes: dict, trusted: None) -> None:
    sandboxes[OWN] = _Sandbox({REPORT: b"own"}, session_scoped=False)

    delivered, _ = _deliver(f"MEDIA:{REPORT}", {SESSION_KEY: OWN})

    assert delivered == []
    assert sandboxes[OWN].fetched == []


def test_redteam_the_engines_mounts_and_credentials_never_leave_the_sandbox(
        home: Path, sandboxes: dict, trusted: None) -> None:
    held_back = ["/root/.hermes/auth.json", "/root/.hermes/skills/notes/SKILL.md", "/root/.hermes/my_token.json",
                 "/root/.ssh/id_rsa", "/etc/passwd", "/root/out/innocent.txt"]
    sandboxes[OWN] = _Sandbox({**{path: b"SECRET" for path in held_back}, "/root/report.xlsx": b"own"},
                              links={"/root/out/innocent.txt": "/root/.hermes/my_token.json"})

    delivered, _ = _deliver(" ".join(f"MEDIA:{path}" for path in [*held_back, "/root/report.xlsx"]),
                            {SESSION_KEY: OWN})

    assert [Path(path).read_bytes() for path in delivered] == [b"own"]
    assert sandboxes[OWN].fetched == ["/root/report.xlsx"]


def test_a_sandbox_with_an_unknown_home_keeps_its_home_directory_back(home: Path, sandboxes: dict,
                                                                      trusted: None) -> None:
    sandboxes[OWN] = _Sandbox({"/root/report.xlsx": b"own", REPORT: b"own"}, home=None)

    delivered, _ = _deliver(f"MEDIA:/root/report.xlsx MEDIA:{REPORT}", {SESSION_KEY: OWN})

    assert [Path(path).read_bytes() for path in delivered] == [b"own"]
    assert sandboxes[OWN].fetched == [REPORT]


def test_a_per_session_docker_sandbox_owns_its_home(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Docker:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setattr(backends, "_DockerEnvironment", _Docker)
    monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda cc: None)
    cc = {"container_persistent": False}

    own = backends._build_docker_env(image="img", cwd="/root", timeout=1, cc=cc, task_id=OWN, host_cwd=None)
    shared = backends._build_docker_env(image="img", cwd="/root", timeout=1, cc=cc, task_id="default", host_cwd=None)

    assert own._session_scoped and own._remote_home == "/root"
    assert getattr(shared, "_remote_home", None) is None


def test_the_switch_is_off_by_default_known_to_the_config_and_bridged_to_the_env(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway import media_policy
    from hermes_cli.config import _validate_config_key
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    monkeypatch.setenv("HERMES_MEDIA_TRUST_SESSION_SANDBOX", "")
    assert DEFAULT_CONFIG["gateway"]["trust_session_sandbox"] is False
    assert _validate_config_key("gateway.trust_session_sandbox")[0]
    assert not media_policy.media_delivery_trust_session_sandbox()

    media_policy.apply_media_policy_env({"gateway": {"trust_session_sandbox": True}})
    assert media_policy.media_delivery_trust_session_sandbox()

    monkeypatch.setattr(media_policy, "_routed_gateway_cfg", lambda: {"trust_session_sandbox": False})
    assert not media_policy.media_delivery_trust_session_sandbox()
