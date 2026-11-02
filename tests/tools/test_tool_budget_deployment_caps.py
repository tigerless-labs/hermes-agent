"""``tool_budget.result_size_chars`` / ``turn_budget_chars`` / ``inline_tools``: a deployment's own caps.

Upstream scales the persistence thresholds to the model's window, so on a 1M-token model one result may carry
100K chars and one batch 200K into the request. A deployment that keeps its prompts under a price line sets lower
caps; they only ever lower the window-scaled values. ``inline_tools`` names tools whose results always stay inline
(tool definitions a search loads, instructions the model must follow whole), in the per-result check and in the
per-batch budget alike. Unset, invalid or empty keys keep upstream's budget exactly.
"""

from __future__ import annotations

import pytest

from tools.budget_config import (
    DEFAULT_MCP_RESULT_SIZE_CHARS,
    DEFAULT_RESULT_SIZE_CHARS,
    DEFAULT_TURN_BUDGET_CHARS,
    BudgetConfig,
    budget_for_context_window,
)
from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, enforce_turn_budget, maybe_persist_tool_result

MILLION = 1_000_000
HUB_TOOL = "slack__slack_search_public_and_private"
INLINE_TOOL = "tool_search"
RESULT_CAP, TURN_CAP = 30_000, 40_000


def _configure(tmp_path, monkeypatch, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.fixture
def capped(tmp_path, monkeypatch) -> BudgetConfig:
    _configure(tmp_path, monkeypatch,
               f"tool_budget:\n  result_size_chars: {RESULT_CAP}\n  turn_budget_chars: {TURN_CAP}\n"
               f"  inline_tools: [{INLINE_TOOL}, hub_read_skill]\n")
    return budget_for_context_window(MILLION)


def test_without_the_keys_the_budget_is_upstreams(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, "tool_budget:\n  mcp_result_size_chars: 50000\n")
    budget = budget_for_context_window(MILLION)
    assert (budget.default_result_size, budget.turn_budget) == (DEFAULT_RESULT_SIZE_CHARS, DEFAULT_TURN_BUDGET_CHARS)
    assert budget.resolve_threshold(INLINE_TOOL) == DEFAULT_RESULT_SIZE_CHARS


def test_caps_lower_the_window_scaled_budget(capped):
    assert capped.default_result_size == RESULT_CAP < DEFAULT_RESULT_SIZE_CHARS
    assert capped.turn_budget == TURN_CAP < DEFAULT_TURN_BUDGET_CHARS
    assert capped.resolve_threshold(HUB_TOOL) == RESULT_CAP


def test_the_result_cap_also_bounds_mcp_tools(capped):
    assert capped.resolve_threshold("mcp_any_server_tool") <= RESULT_CAP < DEFAULT_MCP_RESULT_SIZE_CHARS


def test_caps_never_raise_a_smaller_window_scaled_budget(tmp_path, monkeypatch):
    small_window = 16_384
    upstream = budget_for_context_window(small_window)
    _configure(tmp_path, monkeypatch,
               f"tool_budget:\n  result_size_chars: {upstream.default_result_size * 4}\n"
               f"  turn_budget_chars: {upstream.turn_budget * 4}\n")
    capped = budget_for_context_window(small_window)
    assert (capped.default_result_size, capped.turn_budget) == (upstream.default_result_size, upstream.turn_budget)


def test_caps_apply_when_the_window_is_unknown(capped, tmp_path, monkeypatch):
    budget = budget_for_context_window(None)
    assert (budget.default_result_size, budget.turn_budget) == (RESULT_CAP, TURN_CAP)


@pytest.mark.parametrize("value", ["0", "-5", "abc", "[1, 2]", "null", "true"])
def test_invalid_caps_keep_upstreams_budget(tmp_path, monkeypatch, value):
    _configure(tmp_path, monkeypatch,
               f"tool_budget:\n  result_size_chars: {value}\n  turn_budget_chars: {value}\n  inline_tools: {value}\n")
    budget = budget_for_context_window(MILLION)
    assert (budget.default_result_size, budget.turn_budget) == (DEFAULT_RESULT_SIZE_CHARS, DEFAULT_TURN_BUDGET_CHARS)
    assert budget.resolve_threshold(INLINE_TOOL) == DEFAULT_RESULT_SIZE_CHARS


def test_a_malformed_section_keeps_upstreams_budget(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, "tool_budget: not-a-mapping\n")
    budget = budget_for_context_window(MILLION)
    assert (budget.default_result_size, budget.turn_budget) == (DEFAULT_RESULT_SIZE_CHARS, DEFAULT_TURN_BUDGET_CHARS)


def test_inline_tools_never_persist_and_others_keep_their_thresholds(capped):
    assert capped.resolve_threshold(INLINE_TOOL) == float("inf")
    assert capped.resolve_threshold("hub_read_skill") == float("inf")
    assert capped.resolve_threshold("read_file") == float("inf")
    assert capped.resolve_threshold(HUB_TOOL) == RESULT_CAP


def test_a_result_just_over_the_cap_leaves_only_a_preview(capped):
    content = "x" * (RESULT_CAP + 1)
    replaced = maybe_persist_tool_result(content=content, tool_name=HUB_TOOL, tool_use_id="tc_over", config=capped)
    assert PERSISTED_OUTPUT_TAG in replaced
    assert len(replaced) < RESULT_CAP


def test_a_result_at_the_cap_stays_inline(capped):
    content = "x" * RESULT_CAP
    assert maybe_persist_tool_result(content=content, tool_name=HUB_TOOL, tool_use_id="tc_at", config=capped) == content


def test_an_inline_tool_over_the_cap_stays_whole(capped):
    content = "y" * (RESULT_CAP * 2)
    assert maybe_persist_tool_result(content=content, tool_name=INLINE_TOOL, tool_use_id="tc_inline",
                                     config=capped) == content


def _batch(*sizes_and_names: tuple[int, str]) -> list[dict]:
    return [{"role": "tool", "name": name, "tool_name": name, "tool_call_id": f"tc_{i}", "content": "z" * size}
            for i, (size, name) in enumerate(sizes_and_names)]


def test_the_batch_budget_spills_others_before_touching_an_inline_tool(capped):
    batch = _batch((RESULT_CAP - 1, INLINE_TOOL), (RESULT_CAP - 2, HUB_TOOL), (RESULT_CAP - 3, HUB_TOOL))
    enforce_turn_budget(batch, env=None, config=capped)
    assert PERSISTED_OUTPUT_TAG not in batch[0]["content"]
    assert all(PERSISTED_OUTPUT_TAG in message["content"] for message in batch[1:])


def test_parallel_results_under_the_per_result_cap_are_bounded_together(capped):
    batch = _batch((RESULT_CAP - 1, HUB_TOOL), (RESULT_CAP - 1, HUB_TOOL))
    enforce_turn_budget(batch, env=None, config=capped)
    assert sum(len(message["content"]) for message in batch) <= TURN_CAP


def test_without_inline_tools_the_batch_budget_is_upstreams():
    upstream = BudgetConfig(turn_budget=TURN_CAP)
    batch = _batch((RESULT_CAP - 1, INLINE_TOOL), (RESULT_CAP - 2, HUB_TOOL))
    enforce_turn_budget(batch, env=None, config=upstream)
    assert PERSISTED_OUTPUT_TAG in batch[0]["content"]
