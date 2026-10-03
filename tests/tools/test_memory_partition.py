"""Built-in memory partitioned by chat (``memory.partition_by_chat``).

One gateway instance can serve many people in many chats. With the switch on, each chat gets its
own MEMORY.md, the person profile (USER.md) is kept only in direct messages, and a session that
belongs to no chat (cron, background) gets no built-in memory. The memory guidance then tells the model
its memory belongs to this chat. With the switch off nothing changes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agent_init import _init_memory
from agent.prompt_builder import ACROSS_SESSIONS_SCOPE, THIS_CHAT_SCOPE, build_memory_guidance
from agent.system_prompt import _tool_guidance_block
from tools import memory_tool
from tools.memory_tool import MemoryStore, memory_partition


@pytest.fixture
def memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "memories"
    monkeypatch.setattr(memory_tool, "get_memory_dir", lambda: root)
    return root


def _partitioned(**overrides) -> dict:
    return {"memory": {"partition_by_chat": True, **overrides}}


def _agent(chat_id: str | None, chat_type: str | None) -> SimpleNamespace:
    return SimpleNamespace(_chat_id=chat_id, _chat_type=chat_type, enabled_toolsets=None, disabled_toolsets=None,
                           tools=[], valid_tool_names={"memory"})


def test_a_store_bound_to_a_directory_reads_and_writes_only_there(memory_root: Path, tmp_path: Path) -> None:
    bound = MemoryStore(memory_dir=tmp_path / "chat-a")
    bound.load_from_disk()
    bound.add("memory", "the release ships on Friday")
    assert "Friday" in (tmp_path / "chat-a" / "MEMORY.md").read_text()
    assert not (memory_root / "MEMORY.md").exists()
    reloaded = MemoryStore(memory_dir=tmp_path / "chat-a")
    reloaded.load_from_disk()
    assert any("Friday" in entry for entry in reloaded.memory_entries)


def test_partitioning_is_off_unless_switched_on(memory_root: Path) -> None:
    assert memory_partition({}, "slack", "C1", "group") is None
    assert memory_partition({"partition_by_chat": False}, "slack", "C1", "group") is None


def test_each_chat_gets_its_own_directory_under_the_memory_root(memory_root: Path) -> None:
    first = memory_partition({"partition_by_chat": True}, "slack", "C1", "group")
    second = memory_partition({"partition_by_chat": True}, "slack", "C2", "group")
    assert first.directory != second.directory
    assert first.directory.parent == second.directory.parent and memory_root in first.directory.parents


def test_channels_keep_notes_only_and_direct_messages_keep_the_profile_too(memory_root: Path) -> None:
    channel = memory_partition({"partition_by_chat": True}, "slack", "C1", "group")
    direct = memory_partition({"partition_by_chat": True}, "slack", "D1", "dm")
    assert (channel.notes, channel.profile) == (True, False)
    assert (direct.notes, direct.profile) == (True, True)


@pytest.mark.parametrize("platform,chat_id", [("cron", None), ("slack", None), ("slack", ""), (None, None)])
def test_a_session_outside_any_chat_gets_no_memory(memory_root: Path, platform, chat_id) -> None:
    partition = memory_partition({"partition_by_chat": True}, platform, chat_id, None)
    assert (partition.notes, partition.profile, partition.directory) == (False, False, None)


@pytest.mark.parametrize("chat_id", ["../../etc", "C1/../../x", "..", "/abs/path", "C1\x00x"])
def test_redteam_a_chat_id_cannot_escape_the_partition_root(memory_root: Path, chat_id: str) -> None:
    partition = memory_partition({"partition_by_chat": True}, "slack", chat_id, "group")
    root = (memory_root / "chats").resolve()
    assert partition.directory.resolve().parent == root


def test_the_agent_binds_its_store_to_its_chat(memory_root: Path) -> None:
    agent = _agent("C1", "group")
    _init_memory(agent, _partitioned(), False, "slack")
    assert agent._memory_store.memory_dir == memory_partition(_partitioned()["memory"], "slack", "C1", "group").directory
    assert agent._memory_enabled and not agent._user_profile_enabled


def test_a_direct_message_agent_keeps_the_profile(memory_root: Path) -> None:
    agent = _agent("D1", "dm")
    _init_memory(agent, _partitioned(), False, "slack")
    assert agent._memory_enabled and agent._user_profile_enabled


def test_redteam_two_channels_never_see_each_others_notes(memory_root: Path) -> None:
    first, second = _agent("C1", "group"), _agent("C2", "group")
    _init_memory(first, _partitioned(), False, "slack")
    first._memory_store.add("memory", "the budget for C1 is 40k")
    _init_memory(second, _partitioned(), False, "slack")
    assert not any("40k" in entry for entry in second._memory_store.memory_entries)
    assert not (memory_root / "MEMORY.md").exists()


def test_a_cron_agent_gets_no_built_in_memory_when_partitioned(memory_root: Path) -> None:
    agent = _agent(None, None)
    _init_memory(agent, _partitioned(), False, "cron")
    assert agent._memory_store is None and not agent._memory_enabled and not agent._user_profile_enabled


def test_the_config_switch_still_narrows_inside_a_partition(memory_root: Path) -> None:
    agent = _agent("D1", "dm")
    _init_memory(agent, _partitioned(user_profile_enabled=False), False, "slack")
    assert agent._memory_enabled and not agent._user_profile_enabled


def test_without_the_switch_the_agent_uses_the_shared_store(memory_root: Path) -> None:
    agent = _agent("C1", "group")
    _init_memory(agent, {"memory": {}}, False, "slack")
    assert agent._memory_store.memory_dir is None and agent._user_profile_enabled


def test_without_the_switch_the_guidance_is_the_unpartitioned_one(memory_root: Path) -> None:
    agent = _agent("C1", "group")
    _init_memory(agent, {"memory": {}}, False, "slack")
    assert _tool_guidance_block(agent) == build_memory_guidance(True, True, skill_manage_available=False)
    assert ACROSS_SESSIONS_SCOPE in _tool_guidance_block(agent)


@pytest.mark.parametrize("chat_id,chat_type,overrides", [
    ("C1", "group", {}), ("D1", "dm", {}), ("D1", "dm", {"memory_enabled": False})])
def test_a_partitioned_session_is_told_its_memory_belongs_to_this_chat(memory_root: Path, chat_id, chat_type,
                                                                       overrides) -> None:
    agent = _agent(chat_id, chat_type)
    _init_memory(agent, _partitioned(**overrides), False, "slack")
    guidance = _tool_guidance_block(agent)
    assert THIS_CHAT_SCOPE in guidance and ACROSS_SESSIONS_SCOPE not in guidance
    assert guidance == build_memory_guidance(agent._memory_enabled, agent._user_profile_enabled,
                                             skill_manage_available=False, chat_scoped=True)


def test_redteam_a_session_outside_any_chat_gets_no_memory_guidance(memory_root: Path) -> None:
    agent = _agent(None, None)
    _init_memory(agent, _partitioned(), False, "cron")
    assert _tool_guidance_block(agent) is None
