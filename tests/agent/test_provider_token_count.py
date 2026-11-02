"""``compression.provider_token_count``: the pre-request pressure check takes the provider's own count.

Each response carries real usage, but what was appended since (tool results, a new message) and a transcript a
compaction just rewrote are estimated, and the estimate runs at about half the real size on CJK-with-JSON
payloads. With the switch on, when the usage anchor does not cover the request, a provider that can count
(only Anthropic, through ``messages.count_tokens``) counts the request about to be sent; the figure is treated as
real usage. Every failure (other providers, errors, timeouts, non-positive or non-integer answers) keeps the
estimate. Counting never consumes the one-shot settings that belong to the real request. Switched off, nothing
is counted.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from agent.provider_token_count import counted_request_pressure
from agent.usage_anchor import capture_usage_anchor

TIMEOUT = 2.5
HISTORY = [
    {"role": "user", "content": "搜一下大家关于压缩的讨论"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "slack__search", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "第一条：阈值放在八万五千之下。" * 200},
]


class FakeCounter:
    def __init__(self, answer=48_321, error: Exception | None = None):
        self.answer, self.error, self.calls = answer, error, []

    def count_tokens(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(input_tokens=self.answer)


def _configure(tmp_path, monkeypatch, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.fixture
def switched_on(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch,
               f"compression:\n  provider_token_count: true\n  provider_token_count_timeout_seconds: {TIMEOUT}\n")


def _anthropic_agent(counter: FakeCounter, *, api_mode: str = "anthropic_messages"):
    from agent.transports.anthropic import AnthropicTransport

    agent = SimpleNamespace(
        api_mode=api_mode, model="claude-haiku-5-5", max_tokens=4096, reasoning_config=None,
        _is_anthropic_oauth=False, _anthropic_base_url="https://api.anthropic.com", _oauth_1m_beta_disabled=False,
        context_compressor=SimpleNamespace(context_length=1_000_000), _usage_anchor=None,
        _anthropic_client=SimpleNamespace(messages=counter),
        _ephemeral_max_output_tokens=777, _ephemeral_reasoning_off=True,
    )
    agent._get_transport = AnthropicTransport
    agent._prepare_anthropic_messages_for_api = lambda messages: messages
    agent._anthropic_preserve_dots = lambda: False
    return agent


def _api_messages():
    return [{"role": "system", "content": "你是 Lara。"}, *copy.deepcopy(HISTORY)]


def _count(agent, messages=None, tools=None):
    return counted_request_pressure(agent, messages or copy.deepcopy(HISTORY), _api_messages(), tools or [])


def test_switched_off_nothing_is_counted(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, "compression:\n  threshold_tokens: 85000\n")
    counter = FakeCounter()
    assert _count(_anthropic_agent(counter)) is None
    assert counter.calls == []


def test_without_an_anchor_the_request_about_to_be_sent_is_counted(switched_on):
    counter = FakeCounter()
    assert _count(_anthropic_agent(counter)) == counter.answer
    sent = counter.calls[0]
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["system"] and sent["messages"]
    assert sent["timeout"] == TIMEOUT
    assert "max_tokens" not in sent


def test_an_anchor_that_covers_the_request_is_not_recounted(switched_on):
    counter = FakeCounter()
    agent = _anthropic_agent(counter)
    agent._usage_anchor = capture_usage_anchor(60_000, 500, copy.deepcopy(HISTORY))
    assert _count(agent) is None
    assert counter.calls == []


def test_an_anchor_with_messages_appended_since_is_recounted(switched_on):
    counter = FakeCounter()
    agent = _anthropic_agent(counter)
    agent._usage_anchor = capture_usage_anchor(60_000, 500, copy.deepcopy(HISTORY[:1]))
    assert _count(agent) == counter.answer


def test_a_stale_anchor_after_a_rewrite_is_recounted(switched_on):
    counter = FakeCounter()
    agent = _anthropic_agent(counter)
    rewritten = copy.deepcopy(HISTORY)
    agent._usage_anchor = capture_usage_anchor(60_000, 500, rewritten)
    rewritten[0]["content"] = "[压缩后的摘要]"
    assert counted_request_pressure(agent, rewritten, _api_messages(), []) == counter.answer


def test_counting_leaves_the_real_request_s_one_shot_settings(switched_on):
    counter = FakeCounter()
    agent = _anthropic_agent(counter)
    _count(agent)
    assert (agent._ephemeral_max_output_tokens, agent._ephemeral_reasoning_off) == (777, True)


def test_deferred_tools_are_counted_as_the_wire_sends_them(switched_on):
    counter = FakeCounter()
    loaded = {"type": "function", "function": {"name": "hub_tool", "description": "d",
                                               "parameters": {"type": "object", "properties": {}}}}
    deferred = {"type": "function", "function": {"name": "slack__search", "defer_loading": True,
                                                 "parameters": {"type": "object", "properties": {}}}}
    _count(_anthropic_agent(counter), tools=[loaded, deferred])
    assert [tool["name"] for tool in counter.calls[0]["tools"]] == ["hub_tool"]


@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses", "bedrock_converse"])
def test_redteam_providers_that_cannot_count_keep_the_estimate(switched_on, api_mode):
    counter = FakeCounter()
    assert _count(_anthropic_agent(counter, api_mode=api_mode)) is None
    assert counter.calls == []


@pytest.mark.parametrize("answer", [0, -5, "48321", None, True, 3.5])
def test_redteam_an_answer_that_is_not_a_positive_count_keeps_the_estimate(switched_on, answer):
    assert _count(_anthropic_agent(FakeCounter(answer=answer))) is None


@pytest.mark.parametrize("error", [TimeoutError("slow"), RuntimeError("boom"), ValueError("bad request")])
def test_redteam_a_failing_count_keeps_the_estimate(switched_on, error):
    assert _count(_anthropic_agent(FakeCounter(error=error))) is None


def test_redteam_an_agent_without_a_client_keeps_the_estimate(switched_on):
    agent = _anthropic_agent(FakeCounter())
    agent._anthropic_client = None
    assert _count(agent) is None


@pytest.mark.parametrize("timeout", ["0", "-1", "abc", "null", "true"])
def test_redteam_an_invalid_timeout_counts_nothing(tmp_path, monkeypatch, timeout):
    _configure(tmp_path, monkeypatch,
               f"compression:\n  provider_token_count: true\n  provider_token_count_timeout_seconds: {timeout}\n")
    counter = FakeCounter()
    assert _count(_anthropic_agent(counter)) is None
    assert counter.calls == []


def test_assembled_request_pressure_is_the_count_and_counts_as_real(switched_on, tmp_path, monkeypatch):
    from tests.agent.test_anthropic_preserved_thinking_replay import (
        _agent, _assemble_and_wire, _patch_assembly_loop, _session_db,
    )
    import agent.turn_request_assembly as assembly

    agent = _agent(_session_db(tmp_path, "s1"))
    _patch_assembly_loop(monkeypatch)
    monkeypatch.setattr(assembly, "counted_request_pressure", lambda *args: 91_234, raising=False)
    assembled, _ = _assemble_and_wire(agent, [{"role": "user", "content": "继续"}])
    assert assembled.request_pressure_tokens == 91_234
    assert agent._request_pressure_anchored is True


def test_assembled_request_pressure_keeps_the_estimate_when_nothing_is_counted(switched_on, tmp_path, monkeypatch):
    from tests.agent.test_anthropic_preserved_thinking_replay import (
        _agent, _assemble_and_wire, _patch_assembly_loop, _session_db,
    )
    import agent.turn_request_assembly as assembly

    agent = _agent(_session_db(tmp_path, "s1"))
    _patch_assembly_loop(monkeypatch)
    monkeypatch.setattr(assembly, "counted_request_pressure", lambda *args: None, raising=False)
    assembled, _ = _assemble_and_wire(agent, [{"role": "user", "content": "继续"}])
    assert assembled.request_pressure_tokens == assembled.approx_tokens
    assert agent._request_pressure_anchored is False
