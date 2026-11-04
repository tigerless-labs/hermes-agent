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

from agent.anthropic_advisor import PAUSED, account_advisor_usage, advisor_tool, executor_passes_cost, resume_paused
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


def _reply(iterations: list[dict] | None, stop_reason: str = "end_turn") -> SimpleNamespace:
    passes = [e for e in iterations or [FIRST] if e["type"] == "message"]
    usage = SimpleNamespace(input_tokens=sum(e["input_tokens"] for e in passes),
                            cache_read_input_tokens=sum(e["cache_read_input_tokens"] for e in passes),
                            cache_creation_input_tokens=sum(e["cache_creation_input_tokens"] for e in passes),
                            output_tokens=sum(e["output_tokens"] for e in passes))
    if iterations is not None:
        usage.iterations = iterations
    return SimpleNamespace(usage=usage, content=[], id=None, model=MODEL, stop_reason=stop_reason)


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
    line = "in/read/write/out=" + " ".join(
        f"{e['input_tokens']}/{e['cache_read_input_tokens']}/{e['cache_creation_input_tokens']}/{e['output_tokens']}"
        for e in (FIRST, SECOND))
    assert any(line in r.getMessage() for r in caplog.records)


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


THIRD = {"type": "message", "input_tokens": 512, "cache_read_input_tokens": 68_336, "cache_creation_input_tokens": 1_024,
         "output_tokens": 640}
KWARGS = {"messages": [{"role": "user", "content": "how many new orders?"}],
          "tools": [advisor_tool({"advisor": {"model": ADVISOR, "max_uses": 2}})]}


def _resumed(first_part: SimpleNamespace, *later_parts: SimpleNamespace) -> SimpleNamespace:
    pending = list(later_parts)

    def send(_kwargs):
        part = pending.pop(0)
        account_advisor_usage(part, AGENT)
        return part

    account_advisor_usage(first_part, AGENT)
    return resume_paused(send, KWARGS, first_part, AGENT)


def _passes_cost(*entries: dict) -> Decimal:
    return sum((_cost(_canonical(e)) for e in entries), Decimal(0))


def test_every_part_of_a_resumed_request_is_priced_and_the_costs_add_up():
    joined = _resumed(_reply([FIRST, ADVICE], PAUSED), _reply([SECOND]))
    assert executor_passes_cost(MODEL, joined.usage, provider="anthropic").amount_usd == _passes_cost(FIRST, SECOND)


def test_a_resumed_part_with_its_own_passes_keeps_each_of_them():
    joined = _resumed(_reply([FIRST, ADVICE], PAUSED), _reply([SECOND, ADVICE, THIRD]))
    assert executor_passes_cost(MODEL, joined.usage, provider="anthropic").amount_usd == _passes_cost(FIRST, SECOND, THIRD)


def test_a_resumed_request_counts_every_part_and_reads_the_first_part_as_its_context():
    joined = _resumed(_reply([FIRST, ADVICE], PAUSED), _reply([SECOND, ADVICE, THIRD]))
    for wire in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"):
        assert getattr(joined.usage, wire) == sum(e[wire] for e in (FIRST, SECOND, THIRD))
    assert joined.usage.initial_prompt_tokens == _prompt(FIRST)
    assert joined.usage.initial_cached_tokens == FIRST["cache_read_input_tokens"]


def test_the_joined_passes_are_logged_last_for_the_golden_split(caplog):
    with caplog.at_level(logging.INFO):
        _resumed(_reply([FIRST, ADVICE], PAUSED), _reply([SECOND]))
    lines = [r.getMessage() for r in caplog.records if "Executor passes" in r.getMessage()]
    assert lines[-1].endswith(" ".join(
        f"{e['input_tokens']}/{e['cache_read_input_tokens']}/{e['cache_creation_input_tokens']}/{e['output_tokens']}"
        for e in (FIRST, SECOND)))


@pytest.mark.parametrize("broken", ["missing input", "non-numeric", "no usage"])
def test_redteam_a_part_whose_usage_cannot_be_read_leaves_the_last_part_accounted_alone(broken):
    first = _reply([FIRST, ADVICE], PAUSED)
    if broken == "missing input":
        del first.usage.input_tokens
    elif broken == "non-numeric":
        first.usage.output_tokens = "lots"
    else:
        first.usage = None
    last = _reply([SECOND])
    before = vars(last.usage).copy()
    joined = _resumed(first, last)
    assert vars(joined.usage) == before
    assert executor_passes_cost(MODEL, joined.usage, provider="anthropic") is None


def test_the_recorded_cost_of_a_resumed_request_covers_every_part(tmp_path, monkeypatch):
    from agent import turn_usage

    agent = _anthropic_agent(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setattr(agent.context_compressor, "update_from_response", lambda usage: seen.update(usage))
    joined = _resumed(_reply([FIRST, ADVICE], PAUSED), _reply([SECOND]))
    before = agent.session_estimated_cost_usd
    try:
        turn_usage.record_response_usage(agent, joined, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                                         api_duration=0.2, compression_attempts=0, max_compression_attempts=3)
    finally:
        agent.close()
    assert seen["prompt_tokens"] == _prompt(FIRST)
    assert agent.session_estimated_cost_usd - before == pytest.approx(float(_passes_cost(FIRST, SECOND)))
