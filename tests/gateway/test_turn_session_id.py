"""Each turn binds its session id with the other session vars (``gateway.bind_session_id_per_turn``).

The gateway resets the session vars at every turn without a session id; only building an agent publishes
it. On a thread's later turns the agent comes from the cache, so tools ran with an empty session id: the
sandbox fetch found no session and a file the session made in its own container could not be sent mid-turn.
"""

from __future__ import annotations

import contextvars
import threading
from pathlib import Path

import pytest
import yaml

import gateway.run as gateway_run
import tools.terminal_tool as terminal_tool
from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.session import SessionContext, SessionSource
from gateway.session_context import (
    clear_session_vars, delivering_session_id, get_session_env, set_current_session_id)
from hermes_constants import chat_scope_slug
from tools.environments.base import FileFetchError
from tools.thread_context import propagate_context_to_thread

CHAT, THREAD, USER = "C1", "1792089908.934399", "U1"
OWN, OTHER = "20261015_184508_aaaaaaaa", "20261015_184514_bbbbbbbb"
REPORT = "/tmp/test_chart.png"


class _Sandbox:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files, self.fetched = dict(files), []
        self._session_scoped, self._remote_home = True, "/root"

    def fetch_realpath(self, remote_path: str) -> str:
        return remote_path

    def fetch_file(self, remote_path: str, local_dest: Path, *, max_bytes: int) -> None:
        self.fetched.append(remote_path)
        if remote_path not in self.files:
            raise FileFetchError("missing")
        Path(local_dest).write_bytes(self.files[remote_path])


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from gateway.platforms import base
    hermes = tmp_path / "hermes"
    hermes.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setenv("TERMINAL_DOCKER_CACHE_SCOPE", "chat")
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_SESSION_SANDBOX", "1")
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (hermes / "cache" / "documents",))
    monkeypatch.setattr(base, "_profile_cache_roots", lambda: [])
    return hermes


@pytest.fixture
def switched_on(home: Path) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({"gateway": {"bind_session_id_per_turn": True}}))


@pytest.fixture
def sandboxes(monkeypatch: pytest.MonkeyPatch) -> dict:
    registry = {OWN: _Sandbox({REPORT: b"own chart"}), OTHER: _Sandbox({REPORT: b"other chart"})}
    monkeypatch.setattr(terminal_tool, "_active_environments", registry)
    return registry


def _runner() -> gateway_run.GatewayRunner:
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    return runner


def _context(session_id: str = OWN, chat: str = CHAT, user: str = USER) -> SessionContext:
    source = SessionSource(platform=Platform.SLACK, chat_id=chat, chat_type="channel", thread_id=THREAD,
                           user_id=user, user_name="Ryan")
    return SessionContext(source=source, connected_platforms=[Platform.SLACK], home_channels={},
                          session_key=f"agent:main:slack:channel:{chat}:{THREAD}", session_id=session_id)


def _a_later_turn(context: SessionContext) -> None:
    """Turn one built the agent, which published its id; the next turn reuses that agent from the cache."""
    set_current_session_id(context.session_id)
    clear_session_vars([])
    _runner()._set_session_env(context)


def _in_a_tool_thread(read):
    seen = {}
    worker = threading.Thread(target=propagate_context_to_thread(lambda: seen.update(value=read())))
    worker.start()
    worker.join()
    return seen["value"]


def _sent_mid_turn(path: str = REPORT) -> tuple[list[str], list[dict]]:
    dropped: list[dict] = []
    kept = BasePlatformAdapter.filter_media_delivery_paths([(path, False)], dropped=dropped)
    return [delivered for delivered, _ in kept], dropped


def test_the_switch_is_off_by_default_and_known_to_the_config() -> None:
    from hermes_cli.config import _validate_config_key
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["gateway"]["bind_session_id_per_turn"] is False
    assert _validate_config_key("gateway.bind_session_id_per_turn")[0]


def test_switched_off_a_later_turn_keeps_upstreams_empty_session_id(home: Path) -> None:
    _a_later_turn(_context())
    assert delivering_session_id() == "" and get_session_env("HERMES_SESSION_ID") == ""


def test_switched_on_a_later_turn_binds_its_own_session_id(switched_on: None) -> None:
    _a_later_turn(_context())
    assert delivering_session_id() == OWN and get_session_env("HERMES_SESSION_ID") == OWN


def test_every_session_var_a_tool_reads_on_a_later_turn_is_this_turns(switched_on: None) -> None:
    context = _context()
    _a_later_turn(context)
    expected = {"HERMES_SESSION_PLATFORM": "slack", "HERMES_SESSION_CHAT_ID": context.source.chat_id,
                "HERMES_SESSION_THREAD_ID": context.source.thread_id, "HERMES_SESSION_USER_ID": context.source.user_id,
                "HERMES_SESSION_USER_NAME": context.source.user_name, "HERMES_SESSION_KEY": context.session_key,
                "HERMES_SESSION_ID": context.session_id}
    assert _in_a_tool_thread(lambda: {name: get_session_env(name) for name in expected}) == expected
    assert _in_a_tool_thread(delivering_session_id) == context.session_id


def test_switched_on_a_tool_sends_a_file_from_the_sessions_own_sandbox_on_a_later_turn(
        switched_on: None, home: Path, sandboxes: dict) -> None:
    _a_later_turn(_context())
    delivered, dropped = _in_a_tool_thread(_sent_mid_turn)
    assert dropped == [] and len(delivered) == 1
    copy = Path(delivered[0])
    assert copy.read_bytes() == b"own chart" and copy.name == Path(REPORT).name
    assert copy.is_relative_to(home / "cache" / "documents" / "chats" / chat_scope_slug("slack", CHAT))
    assert sandboxes[OTHER].fetched == []


def test_switched_off_the_same_file_is_dropped_with_its_reason(home: Path, sandboxes: dict) -> None:
    _a_later_turn(_context())
    delivered, dropped = _in_a_tool_thread(_sent_mid_turn)
    assert delivered == [] and [entry["path"] for entry in dropped] == [REPORT] and dropped[0]["reason"]
    assert sandboxes[OWN].fetched == sandboxes[OTHER].fetched == []


def test_switched_on_a_background_process_started_on_a_later_turn_belongs_to_its_session(
        switched_on: None) -> None:
    from tools.process_registry import ProcessRegistry

    _a_later_turn(_context())
    process = _in_a_tool_thread(lambda: ProcessRegistry._new_session("sleep 60", OWN, None, "", "/root"))
    assert process.parent_session_id == OWN


def test_the_binding_ends_with_the_turn(switched_on: None) -> None:
    _a_later_turn(_context())
    clear_session_vars([])
    assert delivering_session_id() == "" and get_session_env("HERMES_SESSION_ID") == ""


def test_redteam_two_sessions_at_once_each_reach_only_their_own_sandbox(
        switched_on: None, sandboxes: dict) -> None:
    sandboxes[OTHER].files[REPORT] = b"other chart"
    results: dict = {}
    both_bound = threading.Barrier(2)

    def turn(context: SessionContext) -> None:
        _a_later_turn(context)
        both_bound.wait()
        delivered, _ = _sent_mid_turn()
        results[context.session_id] = [Path(path).read_bytes() for path in delivered]

    workers = [threading.Thread(target=contextvars.copy_context().run, args=(turn, context))
               for context in (_context(OWN), _context(OTHER, chat="C2", user="U2"))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert results == {OWN: [b"own chart"], OTHER: [b"other chart"]}
    assert sandboxes[OWN].fetched == sandboxes[OTHER].fetched == [REPORT]


def test_redteam_another_sessions_id_in_the_process_env_changes_nothing(
        switched_on: None, sandboxes: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_SESSION_ID", OTHER)
    _a_later_turn(_context())
    delivered, _ = _in_a_tool_thread(_sent_mid_turn)
    assert [Path(path).read_bytes() for path in delivered] == [b"own chart"]
    assert sandboxes[OTHER].fetched == []


def test_redteam_a_turn_without_a_session_record_fetches_from_no_sandbox(
        switched_on: None, sandboxes: dict) -> None:
    clear_session_vars([])
    _runner()._set_session_env(_context(session_id=""))
    delivered, dropped = _in_a_tool_thread(_sent_mid_turn)
    assert delivered == [] and len(dropped) == 1
    assert sandboxes[OWN].fetched == sandboxes[OTHER].fetched == []
