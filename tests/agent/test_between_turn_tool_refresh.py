"""A plugin that registers or drops tools while a session is cached reaches that session's next
turn: the turn start re-derives the tool snapshot whenever the registry moved past it."""

from types import SimpleNamespace

import pytest

from agent import turn_context
from tools import mcp_tool_agent
from tools.registry import registry


@pytest.fixture
def refreshes(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_tool_agent, "refresh_agent_mcp_tools",
                        lambda agent, **options: calls.append(options) or set())
    return calls


def _agent(generation):
    return SimpleNamespace(_tool_snapshot_generation=generation)


def test_a_registry_that_moved_past_the_snapshot_refreshes_it_keeping_the_prefix(monkeypatch, refreshes):
    monkeypatch.setattr(registry, "_generation", 7)
    turn_context._refresh_mcp_tools_between_turns(_agent(6))
    options, = refreshes
    assert options["preserve_prefix"] is True and options["content_aware"] is True


def test_an_unmoved_registry_leaves_the_snapshot_alone(monkeypatch, refreshes):
    monkeypatch.setattr(registry, "_generation", 7)
    turn_context._refresh_mcp_tools_between_turns(_agent(7))
    assert refreshes == []


def test_an_agent_that_opted_out_of_refreshes_keeps_its_snapshot(monkeypatch, refreshes):
    monkeypatch.setattr(registry, "_generation", 9)
    agent = _agent(1)
    agent._skip_mcp_refresh = True
    turn_context._refresh_mcp_tools_between_turns(agent)
    assert refreshes == []
