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


# --- her own long tool inputs, skill bodies, demotion before the tail is measured ---------------------------------

import json
from unittest.mock import patch

SKILL_URI = "skill://department/marketing/blog-writing/SKILL.md"


@pytest.fixture
def inputs_and_skills_on(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch,
               "compression:\n  tail_budget_real_tokens: 20000\n  tail_tool_rounds: 1\n  tail_demote_to_file: true\n"
               "  tail_demote_tool_inputs: true\n  tail_skill_tools: [hub_read_skill]\n"
               "  tail_skill_tokens: 1000\n  tail_skills_total_tokens: 1500\n"
               "tool_budget:\n  inline_tools: [hub_read_skill]\n")


def _call(call_id: str, tool: str, arguments: dict | str, result: str) -> list[dict]:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return [{"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": tool, "arguments": args}}]},
            {"role": "tool", "tool_call_id": call_id, "tool_name": tool, "content": result}]


ARTICLE = "保险知识正文。" * 2_000


def _writing_turn() -> list[dict]:
    return [{"role": "user", "content": "重写这篇博客"},
            *_call("call_skill_1", "hub_read_skill", {"uri": SKILL_URI}, "写作规范第一版。" * 1_500),
            *_call("call_write_1", "execute_code", {"code": f"open('a.md','w').write('''{ARTICLE}''')"}, "ok"),
            *_call("call_skill_2", "hub_read_skill", {"uri": SKILL_URI}, "写作规范第二版。" * 1_500),
            *_call("call_other_skill", "hub_read_skill", {"uri": "skill://department/marketing/seo/SKILL.md"},
                   "SEO 规范。" * 1_500),
            *_call("call_write_2", "execute_code", {"code": f"open('b.md','w').write('''{ARTICLE}''')"}, "ok")]


def _args(messages, call_id):
    return next(call["function"]["arguments"] for m in messages if m.get("role") == "assistant"
                for call in m.get("tool_calls") or [] if call["id"] == call_id)


def _content(messages, call_id):
    return next(m["content"] for m in messages if m.get("tool_call_id") == call_id)


def test_older_long_inputs_leave_a_path_and_the_newest_call_stays(inputs_and_skills_on):
    from tools.tool_result_storage import get_spillover_dir

    messages = _writing_turn()
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    moved = json.loads(_args(demoted, "call_write_1"))
    assert set(moved) == {"moved_at_compaction", "chars"} and moved["chars"] == len(_args(messages, "call_write_1"))
    assert (get_spillover_dir() / "call_write_1.input.txt").read_text(encoding="utf-8") == _args(messages, "call_write_1")
    assert _args(demoted, "call_write_2") == _args(messages, "call_write_2")
    assert _args(demoted, "call_skill_1") == _args(messages, "call_skill_1")


def test_a_skill_keeps_only_its_newest_copy_within_the_caps(inputs_and_skills_on):
    from agent.model_metadata import estimate_tokens_rough

    messages = _writing_turn()
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    older, newer = _content(demoted, "call_skill_1"), _content(demoted, "call_skill_2")
    assert SKILL_URI in older and len(older) < 400
    assert newer.startswith("写作规范第二版") and estimate_tokens_rough(newer) <= 1_000 + 50
    kept = [_content(demoted, c) for c in ("call_skill_2", "call_other_skill")]
    assert sum(estimate_tokens_rough(text) for text in kept) <= 1_500 + 100


def test_past_the_total_the_oldest_skill_is_dropped_first(inputs_and_skills_on):
    messages = _writing_turn()
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    oldest_distinct, newest_distinct = _content(demoted, "call_skill_2"), _content(demoted, "call_other_skill")
    assert newest_distinct.startswith("SEO 规范")
    assert oldest_distinct.startswith("写作规范第二版")
    assert "read it again" in oldest_distinct


def test_a_skill_read_in_the_newest_round_stays_whole(inputs_and_skills_on):
    messages = [*_writing_turn(), *_call("call_skill_now", "hub_read_skill", {"uri": SKILL_URI}, "最新规范。" * 3_000)]
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    assert _content(demoted, "call_skill_now") == _content(messages, "call_skill_now")


@pytest.mark.parametrize("arguments", ["{not json" + "x" * 5_000, "[" + "1," * 3_000 + "1]"], ids=["not-json", "not-object"])
def test_redteam_unparseable_or_odd_inputs_still_move_whole(inputs_and_skills_on, arguments):
    from tools.tool_result_storage import get_spillover_dir

    messages = [{"role": "user", "content": "go"}, *_call("call_odd", "execute_code", arguments, "ok"),
                *_call("call_new", "execute_code", {"code": "1"}, "ok")]
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    assert json.loads(_args(demoted, "call_odd"))["chars"] == len(arguments)
    assert (get_spillover_dir() / "call_odd.input.txt").read_text(encoding="utf-8") == arguments


def test_redteam_a_call_without_an_id_is_left_alone(inputs_and_skills_on):
    messages = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": "execute_code", "arguments": json.dumps({"code": ARTICLE})}}]},
                *_call("call_new", "execute_code", {"code": "1"}, "ok")]
    assert _compressor()._demote_stale_tail_tools(messages, 0)[1] == messages[1]


def test_redteam_outside_any_chat_inputs_and_skills_stay(inputs_and_skills_on, monkeypatch):
    import tools.tool_result_storage as storage

    monkeypatch.setattr(storage, "_outside_any_chat", lambda: True)
    messages = _writing_turn()
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    assert _args(demoted, "call_write_1") == _args(messages, "call_write_1")


def test_switched_off_inputs_and_skills_are_upstream_s(switched_on):
    messages = _writing_turn()
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    assert _args(demoted, "call_write_1") == _args(messages, "call_write_1")
    assert _content(demoted, "call_skill_1") == _content(messages, "call_skill_1")


def _history_before_the_writing_turn() -> list[dict]:
    return [{"role": "system", "content": "你是 Lara。"}, {"role": "user", "content": "先看看选题"},
            {"role": "assistant", "content": "好的,选题如下。" * 50}]


def test_the_window_is_measured_after_the_tail_is_demoted(inputs_and_skills_on):
    compressor = _compressor()
    seen = {}
    real_window = compressor._compress_window

    def window(messages):
        seen["args"] = _args(messages, "call_write_1")
        return real_window(messages)

    with patch.object(compressor, "_compress_window", side_effect=window), \
            patch.object(compressor, "_generate_summary", return_value=None):
        compressor.compress([*_history_before_the_writing_turn(), *_writing_turn()], current_tokens=120_000)
    assert set(json.loads(seen["args"])) == {"moved_at_compaction", "chars"}


def test_switched_off_the_window_sees_the_upstream_transcript(switched_off):
    compressor = _compressor()
    seen = {}
    real_window = compressor._compress_window

    def window(messages):
        seen["args"] = _args(messages, "call_write_1")
        return real_window(messages)

    messages = [*_history_before_the_writing_turn(), *_writing_turn()]
    with patch.object(compressor, "_compress_window", side_effect=window), \
            patch.object(compressor, "_generate_summary", return_value=None):
        compressor.compress(messages, current_tokens=120_000)
    assert seen["args"] == _args(messages, "call_write_1")


def test_a_rewritten_skill_carries_no_stale_copy_to_send(inputs_and_skills_on):
    messages = _writing_turn()
    for message in messages:
        if message.get("tool_call_id") == "call_skill_1":
            message["api_content"] = message["content"]
    demoted = _compressor()._demote_stale_tail_tools(messages, 0)
    rewritten = next(m for m in demoted if m.get("tool_call_id") == "call_skill_1")
    assert "api_content" not in rewritten
