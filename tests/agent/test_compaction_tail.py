"""``compression.tail_budget_real_tokens`` / ``tail_tool_rounds`` / ``tail_demote_to_file``: what a compaction keeps.

Upstream budgets the verbatim tail in chars/4, which on CJK-with-JSON payloads keeps about twice the budget, and
never demotes the newest six tool rounds. With the switches on, the tail budget is a real-token cap converted
through this session's own real-to-estimate ratio, measured between consecutive provider counts (no extra
calls); jumps from a compaction or a changed tool list are not samples. Only the newest ``tail_tool_rounds`` stay
whole; older long results move to the chat's file and leave a one-line path, the way Claude Code keeps large
files as paths. Unset keeps upstream exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.compaction_tail import calibrated_tail_budget, record_provider_count, tail_tool_rounds
from agent.context_compressor import _LEAN_TAIL_KEEP_TOOL_ROUNDS, ContextCompressor

REAL_CAP = 20_000
UPSTREAM = 25_000


def _configure(tmp_path, monkeypatch, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.fixture
def switched_on(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch,
               f"compression:\n  tail_budget_real_tokens: {REAL_CAP}\n  tail_tool_rounds: 1\n  tail_demote_to_file: true\n"
               "tool_budget:\n  inline_tools: [hub_read_skill]\n")


@pytest.fixture
def switched_off(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, "compression:\n  threshold_tokens: 85000\n")


def _compressor() -> ContextCompressor:
    compressor = ContextCompressor("claude-haiku-5-5", provider="anthropic", base_url="https://api.anthropic.com",
                                   api_mode="anthropic_messages", quiet_mode=True, config_context_length=1_000_000)
    compressor._session_id = "s_tail"
    return compressor


def _agent(compressor: ContextCompressor):
    return SimpleNamespace(context_compressor=compressor)


def _messages(chars: int) -> list[dict]:
    return [{"role": "system", "content": "你是 Lara。"}, {"role": "user", "content": "表" * chars}]


TOOLS = [{"type": "function", "function": {"name": "airtable__list", "parameters": {"type": "object"}}}]


def _sample(agent, counted: int, chars: int, tools=TOOLS) -> None:
    record_provider_count(agent, counted, _messages(chars), tools)


# --- the session's real-to-estimate ratio ------------------------------------------------------------------------

def test_the_ratio_is_real_growth_over_estimated_growth_between_counts(switched_on):
    compressor = _compressor()
    agent = _agent(compressor)
    _sample(agent, 50_000, 4_000)
    _sample(agent, 70_000, 24_000)
    _sample(agent, 90_000, 44_000)
    from agent.model_metadata import estimate_messages_tokens_rough
    estimated_growth = estimate_messages_tokens_rough(_messages(44_000)) - estimate_messages_tokens_rough(_messages(4_000))
    assert calibrated_tail_budget(compressor, UPSTREAM) == pytest.approx(REAL_CAP * estimated_growth / 40_000, rel=0.01)


def test_a_compaction_or_a_changed_tool_list_is_not_a_sample(switched_on):
    compressor = _compressor()
    agent = _agent(compressor)
    _sample(agent, 50_000, 4_000)
    _sample(agent, 30_000, 2_000)
    _sample(agent, 90_000, 6_000, tools=[*TOOLS, {"type": "function", "function": {"name": "loaded__tool"}}])
    assert calibrated_tail_budget(compressor, UPSTREAM) == min(UPSTREAM, REAL_CAP)


def test_before_enough_growth_is_measured_the_cap_is_taken_as_estimated_tokens(switched_on):
    compressor = _compressor()
    agent = _agent(compressor)
    _sample(agent, 50_000, 4_000)
    _sample(agent, 50_400, 4_400)
    assert calibrated_tail_budget(compressor, UPSTREAM) == min(UPSTREAM, REAL_CAP)


def test_a_denser_session_keeps_a_shorter_tail_and_never_more_than_upstream(switched_on):
    budgets = []
    for real_growth in (20_000, 40_000, 80_000):
        compressor = _compressor()
        _sample(_agent(compressor), 50_000, 4_000)
        _sample(_agent(compressor), 50_000 + real_growth, 44_000)
        budgets.append(calibrated_tail_budget(compressor, UPSTREAM))
    assert budgets == sorted(budgets, reverse=True)
    assert all(0 < budget <= UPSTREAM for budget in budgets)


@pytest.mark.parametrize("counted", [-5, 0, "90000", None, True])
def test_redteam_a_bad_count_is_not_a_sample(switched_on, counted):
    compressor = _compressor()
    agent = _agent(compressor)
    _sample(agent, 50_000, 4_000)
    record_provider_count(agent, counted, _messages(44_000), TOOLS)
    _sample(agent, 50_100, 44_100)
    assert calibrated_tail_budget(compressor, UPSTREAM) == min(UPSTREAM, REAL_CAP)


def test_the_compressor_s_tail_budget_follows_the_calibration(switched_on):
    compressor = _compressor()
    agent = _agent(compressor)
    _sample(agent, 50_000, 4_000)
    _sample(agent, 130_000, 44_000)
    assert compressor.tail_token_budget < REAL_CAP


def test_switched_off_the_tail_budget_is_upstream_s(switched_off):
    compressor = _compressor()
    upstream = compressor.tail_token_budget
    agent = _agent(compressor)
    _sample(agent, 50_000, 4_000)
    _sample(agent, 130_000, 44_000)
    assert compressor.tail_token_budget == upstream
    assert tail_tool_rounds(_LEAN_TAIL_KEEP_TOOL_ROUNDS) == _LEAN_TAIL_KEEP_TOOL_ROUNDS


@pytest.mark.parametrize("value", ["0", "-1", "abc", "null", "true"])
def test_redteam_an_invalid_cap_keeps_upstream_s_budget(tmp_path, monkeypatch, value):
    _configure(tmp_path, monkeypatch, f"compression:\n  tail_budget_real_tokens: {value}\n  tail_tool_rounds: {value}\n")
    compressor = _compressor()
    assert calibrated_tail_budget(compressor, UPSTREAM) == UPSTREAM
    assert tail_tool_rounds(_LEAN_TAIL_KEEP_TOOL_ROUNDS) == _LEAN_TAIL_KEEP_TOOL_ROUNDS


# --- older tail tool results move to the chat's file --------------------------------------------------------------

def _round(call_id: str, tool: str, content: str) -> list[dict]:
    return [{"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": tool, "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call_id, "tool_name": tool, "content": content}]


def _tail() -> list[dict]:
    return [{"role": "user", "content": "分析一下选题表"},
            *_round("call_old", "airtable__list", "旧记录" * 2_000),
            *_round("call_skill", "hub_read_skill", "技能正文" * 1_000),
            *_round("call_short", "airtable__get", "短"),
            *_round("call_new", "airtable__list", "新记录" * 2_000)]


def test_only_the_newest_round_stays_whole_and_older_long_results_leave_a_path(switched_on):
    from tools.tool_result_storage import get_spillover_dir

    messages = _tail()
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    by_call = {m.get("tool_call_id"): m["content"] for m in demoted if m.get("role") == "tool"}
    assert by_call["call_new"] == messages[-1]["content"]
    assert by_call["call_short"] == "短"
    assert by_call["call_skill"] == messages[4]["content"]
    old = by_call["call_old"]
    assert len(old) < 400 and "airtable__list" in old
    assert (get_spillover_dir() / "call_old.txt").read_text(encoding="utf-8") == messages[2]["content"]


def test_a_result_already_saved_keeps_its_own_path(switched_on):
    from tools.tool_result_storage import maybe_persist_tool_result

    saved = maybe_persist_tool_result(content="大" * 40_000, tool_name="slack__search", tool_use_id="call_saved",
                                      threshold=0)
    path_line = next(line for line in saved.splitlines() if line.startswith("Full output saved to: "))
    messages = [{"role": "user", "content": "继续"}, *_round("call_saved", "slack__search", saved),
                *_round("call_new", "airtable__list", "新" * 10)]
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    stub = next(m["content"] for m in demoted if m.get("tool_call_id") == "call_saved")
    assert path_line.removeprefix("Full output saved to: ") in stub and len(stub) < len(saved)


def test_redteam_outside_any_chat_the_upstream_hint_stays(switched_on, monkeypatch):
    import tools.tool_result_storage as storage

    monkeypatch.setattr(storage, "_outside_any_chat", lambda: True)
    messages = _tail()
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    old = next(m["content"] for m in demoted if m.get("tool_call_id") == "call_old")
    assert "session_search" in old


def test_switched_off_upstream_keeps_six_rounds_and_its_hint(switched_off):
    messages = _tail()
    assert _compressor()._demote_stale_tail_tools(messages, 0) == messages
