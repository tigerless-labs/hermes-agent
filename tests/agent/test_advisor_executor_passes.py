"""An advisor request is billed per executor pass (fork, Lara).

When the executor consults the advisor inside one request, Anthropic bills each executor pass on its own and
reports the top-level usage as their sum, with the per-pass breakdown in ``usage.iterations``. The summed prompt is
not a prompt that ever went over the wire: priced as one request it crossed Haiku's long-context tier although each
pass stayed under it, and as the context size it made compaction and the usage anchor see twice the real prompt.
Each executor pass is priced at its own length and the costs add up; the context engines read the first pass, the
request that was built from the transcript, through the same ``initial_prompt_tokens`` field Gemini's long decode
uses. A request without advisor passes is accounted exactly as before.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from types import SimpleNamespace

import pytest

from agent.anthropic_advisor import account_advisor_usage, executor_passes_cost
from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

MODEL, ADVISOR = "claude-haiku-5-5", "claude-sonnet-5-5"
FIRST = {"type": "message", "input_tokens": 1_237, "cache_read_input_tokens": 66_129, "cache_creation_input_tokens": 0,
         "output_tokens": 300}
ADVICE = {"type": "advisor_message", "model": ADVISOR, "input_tokens": 82_102, "output_tokens": 1_961,
          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
SECOND = {"type": "message", "input_tokens": 931, "cache_read_input_tokens": 67_364, "cache_creation_input_tokens": 0,
          "output_tokens": 972}


def _prompt(entry: dict) -> int:
    return entry["input_tokens"] + entry["cache_read_input_tokens"] + entry["cache_creation_input_tokens"]


def _reply(iterations: list[dict] | None) -> SimpleNamespace:
    passes = [e for e in iterations or [FIRST] if e["type"] == "message"]
    usage = SimpleNamespace(input_tokens=sum(e["input_tokens"] for e in passes),
                            cache_read_input_tokens=sum(e["cache_read_input_tokens"] for e in passes),
                            cache_creation_input_tokens=sum(e["cache_creation_input_tokens"] for e in passes),
                            output_tokens=sum(e["output_tokens"] for e in passes))
    if iterations is not None:
        usage.iterations = iterations
    return SimpleNamespace(usage=usage, content=[], id=None, model=MODEL, stop_reason="end_turn")


def _canonical(entry: dict) -> CanonicalUsage:
    return CanonicalUsage(input_tokens=entry["input_tokens"], output_tokens=entry["output_tokens"],
                          cache_read_tokens=entry["cache_read_input_tokens"],
                          cache_write_tokens=entry["cache_creation_input_tokens"])


def _cost(usage: CanonicalUsage) -> Decimal:
    return estimate_usage_cost(MODEL, usage, provider="anthropic").amount_usd


AGENT = SimpleNamespace(session_id=None, log_prefix="", _session_db=None, _anthropic_base_url=None)


def test_the_context_engines_read_the_first_pass(caplog):
    reply = _reply([FIRST, ADVICE, SECOND])
    with caplog.at_level(logging.INFO):
        account_advisor_usage(reply, AGENT)
    assert reply.usage.initial_prompt_tokens == _prompt(FIRST)
    assert reply.usage.initial_cached_tokens == FIRST["cache_read_input_tokens"]
    assert any(f"{_prompt(FIRST)}+{_prompt(SECOND)}" in r.getMessage() for r in caplog.records)


def test_each_pass_is_priced_at_its_own_length_and_the_costs_add_up():
    reply = _reply([FIRST, ADVICE, SECOND])
    account_advisor_usage(reply, AGENT)
    priced = executor_passes_cost(MODEL, reply.usage, provider="anthropic")
    assert priced.amount_usd == _cost(_canonical(FIRST)) + _cost(_canonical(SECOND))


def test_passes_under_the_tier_cost_less_than_their_sum_priced_as_one_request():
    reply = _reply([FIRST, ADVICE, SECOND])
    account_advisor_usage(reply, AGENT)
    summed = CanonicalUsage(input_tokens=reply.usage.input_tokens, output_tokens=reply.usage.output_tokens,
                            cache_read_tokens=reply.usage.cache_read_input_tokens)
    assert _prompt(FIRST) < 100_000 < summed.prompt_tokens
    assert executor_passes_cost(MODEL, reply.usage, provider="anthropic").amount_usd < _cost(summed)


def test_a_pass_past_the_tier_still_pays_the_long_rate():
    big_first = {**FIRST, "cache_read_input_tokens": 120_000}
    big_second = {**SECOND, "cache_read_input_tokens": 121_000}
    reply = _reply([big_first, ADVICE, big_second])
    account_advisor_usage(reply, AGENT)
    long_rate = _cost(_canonical(big_first)) + _cost(_canonical(big_second))
    assert executor_passes_cost(MODEL, reply.usage, provider="anthropic").amount_usd == long_rate


@pytest.mark.parametrize("iterations", [None, [FIRST], [FIRST, ADVICE]], ids=["no breakdown", "one pass", "no second pass"])
def test_a_request_without_two_executor_passes_is_accounted_as_before(iterations):
    reply = _reply(iterations)
    account_advisor_usage(reply, AGENT)
    assert getattr(reply.usage, "initial_prompt_tokens", None) is None
    assert executor_passes_cost(MODEL, reply.usage, provider="anthropic") is None


@pytest.mark.parametrize("bad", [{"type": "message"}, {"type": "message", "input_tokens": "lots"}, "not a dict"],
                         ids=["missing counts", "non-numeric", "not a mapping"])
def test_redteam_a_malformed_pass_keeps_upstream_accounting(bad):
    reply = _reply([FIRST, ADVICE, SECOND])
    reply.usage.iterations = [FIRST, ADVICE, bad, SECOND]
    account_advisor_usage(reply, AGENT)
    assert getattr(reply.usage, "initial_prompt_tokens", None) is None
    assert executor_passes_cost(MODEL, reply.usage, provider="anthropic") is None


def _anthropic_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(api_key="k", base_url="https://api.anthropic.com", provider="anthropic", api_mode="anthropic_messages",
                   model=MODEL, session_id="t", platform="cli", quiet_mode=True, skip_context_files=True,
                   skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])


def test_the_recorded_cost_and_the_compressor_follow_the_passes(tmp_path, monkeypatch):
    from agent import turn_usage

    agent = _anthropic_agent(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setattr(agent.context_compressor, "update_from_response", lambda usage: seen.update(usage))
    reply = _reply([FIRST, ADVICE, SECOND])
    account_advisor_usage(reply, AGENT)
    before = agent.session_estimated_cost_usd
    try:
        turn_usage.record_response_usage(agent, reply, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                                         api_duration=0.2, compression_attempts=0, max_compression_attempts=3)
    finally:
        agent.close()
    assert seen["prompt_tokens"] == _prompt(FIRST)
    expected = _cost(_canonical(FIRST)) + _cost(_canonical(SECOND))
    assert agent.session_estimated_cost_usd - before == pytest.approx(float(expected))
