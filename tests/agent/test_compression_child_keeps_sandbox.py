"""A compression child keeps its session's sandbox (``terminal.compression_child_keeps_sandbox``).

Out-of-place compression moves a session onto a new id, while its sandbox stays registered under the old one.
Tools then look the sandbox up by the new id: a file made before the split cannot be fetched, and the next
turn, keyed by the new id, starts a fresh container. On, the child id resolves to the parent's container
through the same alias a delegated child uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.terminal_tool as terminal_tool
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import SessionDB
from tests.agent.test_compression_worker_isolation import _build_agent_with_db
from tests.gateway.test_turn_session_id import REPORT, _Sandbox, _in_a_tool_thread, _sent_mid_turn, home  # noqa: F401
from tools.terminal_tool_lifecycle import get_active_env

PARENT, OTHER = "20261015_190000_parent00", "20261015_190000_other000"
MESSAGES = [{"role": "user", "content": f"m{i}"} for i in range(20)]


@pytest.fixture
def aliases(monkeypatch: pytest.MonkeyPatch) -> dict:
    registry: dict = {}
    monkeypatch.setattr(terminal_tool, "_container_aliases", registry)
    return registry


@pytest.fixture
def sandboxes(monkeypatch: pytest.MonkeyPatch) -> dict:
    registry = {PARENT: _Sandbox({REPORT: b"made before the split"}), OTHER: _Sandbox({REPORT: b"another session"})}
    monkeypatch.setattr(terminal_tool, "_active_environments", registry)
    return registry


@pytest.fixture
def switched_on(home: Path, aliases: dict, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    monkeypatch.setenv("TERMINAL_COMPRESSION_CHILD_KEEPS_SANDBOX", "true")


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    store = SessionDB(db_path=tmp_path / "state.db")
    store.create_session(PARENT, source="slack")
    return store


def _split(db: SessionDB, session_id: str = PARENT):
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = False
    agent._cached_system_prompt = "sys"
    agent._compress_context(MESSAGES, "sys", approx_tokens=120_000)
    assert agent.session_id != session_id
    return agent


def test_the_switch_is_off_by_default_and_known_to_the_config() -> None:
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP, _validate_config_key
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["terminal"]["compression_child_keeps_sandbox"] is False
    assert TERMINAL_CONFIG_ENV_MAP["compression_child_keeps_sandbox"] == "TERMINAL_COMPRESSION_CHILD_KEEPS_SANDBOX"
    assert _validate_config_key("terminal.compression_child_keeps_sandbox")[0]


def test_switched_on_the_child_reaches_the_parents_sandbox(switched_on: None, db: SessionDB, sandboxes: dict) -> None:
    child = _split(db).session_id
    assert terminal_tool._resolve_container_task_id(child) == PARENT
    assert get_active_env(child) is sandboxes[PARENT]


def test_switched_on_a_second_split_still_reaches_the_first_sandbox(
        switched_on: None, db: SessionDB, sandboxes: dict) -> None:
    grandchild = _split(db, _split(db).session_id).session_id
    assert get_active_env(grandchild) is sandboxes[PARENT]


def test_switched_on_an_agent_that_adopts_the_live_child_reaches_the_parents_sandbox(
        switched_on: None, db: SessionDB, sandboxes: dict, aliases: dict) -> None:
    from agent.conversation_compression import _adopt_live_compression_child

    child = _split(db).session_id
    aliases.clear()
    stale = _build_agent_with_db(db, PARENT)
    assert _adopt_live_compression_child(stale, db, PARENT) is not None
    assert stale.session_id == child and get_active_env(child) is sandboxes[PARENT]


def test_switched_on_a_file_made_before_the_split_is_sent_after_it(
        switched_on: None, db: SessionDB, sandboxes: dict, home: Path) -> None:  # noqa: F811
    tokens = set_session_vars(platform="slack", chat_id="C1", session_id=PARENT)
    try:
        _split(db)
        delivered, dropped = _in_a_tool_thread(_sent_mid_turn)
    finally:
        clear_session_vars(tokens)
    assert dropped == [] and [Path(path).read_bytes() for path in delivered] == [b"made before the split"]
    assert sandboxes[OTHER].fetched == []


def test_switched_off_the_child_keeps_upstreams_own_key(
        home: Path, aliases: dict, db: SessionDB, sandboxes: dict) -> None:  # noqa: F811
    child = _split(db).session_id
    assert terminal_tool._resolve_container_task_id(child) == child
    assert get_active_env(child) is None and aliases == {}


def test_redteam_in_place_compaction_registers_nothing(switched_on: None, db: SessionDB, aliases: dict) -> None:
    agent = _build_agent_with_db(db, PARENT)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    agent._compress_context(MESSAGES, "sys", approx_tokens=120_000)
    assert agent.session_id == PARENT and aliases == {}


def test_redteam_a_child_without_a_parent_id_never_lands_on_the_shared_default_container(
        switched_on: None, aliases: dict) -> None:
    from agent.conversation_compression import _keep_sandbox_for_compression_child

    _keep_sandbox_for_compression_child("20261015_190000_orphan00", "")
    assert aliases == {} and terminal_tool._resolve_container_task_id("20261015_190000_orphan00") != "default"


def test_redteam_another_session_never_reaches_the_split_sessions_sandbox(
        switched_on: None, db: SessionDB, sandboxes: dict, aliases: dict) -> None:
    child = _split(db).session_id
    assert aliases == {child: PARENT}
    assert get_active_env(OTHER) is sandboxes[OTHER]
    assert get_active_env("20261015_190000_unrelated") is None
