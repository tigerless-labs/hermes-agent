"""The rough request estimate counts a deferred tool by its name only.

A tool declared with ``defer_loading`` stays off the wire until a search loads it (``plain_tools``), or goes out
parameters-only inside a client tool search the model has not run yet (Responses). Counting its full schema made
the estimate right after a compaction several times the provider's count, so the progress check called a
compaction that worked "insufficient" and skipped the passes that would have brought the prompt under threshold.
Tools without the mark are estimated exactly as before.
"""

from __future__ import annotations

import copy

from agent.model_metadata import estimate_request_tokens_rough


def _tool(name: str, *, deferred: bool = False) -> dict:
    properties = {f"field_{i}": {"type": "string", "description": f"what field {i} of {name} holds " * 4}
                  for i in range(12)}
    function = {"name": name, "description": f"{name} does a thing on the platform " * 6,
                "parameters": {"type": "object", "properties": properties}}
    if deferred:
        function = {"name": name, "parameters": function["parameters"], "defer_loading": True,
                    "namespace": {"name": name.split("__")[0], "description": "a platform"}}
    return {"type": "function", "function": function}


def _tools(count: int, *, deferred: bool) -> list[dict]:
    return [_tool(f"platform__tool_{i}", deferred=deferred) for i in range(count)]


MESSAGES = [{"role": "user", "content": "hello"}]


def _estimate(tools: list[dict]) -> int:
    return estimate_request_tokens_rough(MESSAGES, system_prompt="system", tools=tools)


def test_deferred_tools_estimate_below_the_same_tools_loaded():
    deferred, loaded = _tools(40, deferred=True), _tools(40, deferred=False)
    assert _estimate(deferred) < _estimate(loaded)


def test_a_deferred_tool_adds_no_more_than_its_name():
    base_tools = _tools(1, deferred=False)
    with_deferred_tools = [*base_tools, _tool("platform__a_long_deferred_tool_name", deferred=True)]
    assert 0 <= _estimate(with_deferred_tools) - _estimate(base_tools) <= len("platform__a_long_deferred_tool_name")


def test_deferred_declarations_grow_the_estimate_by_names_not_schemas():
    few_deferred, many_deferred = _tools(5, deferred=True), _tools(500, deferred=True)
    few_loaded, many_loaded = _tools(5, deferred=False), _tools(500, deferred=False)
    extra_names = sum(len(tool["function"]["name"]) for tool in many_deferred[5:])
    deferred_growth = _estimate(many_deferred) - _estimate(few_deferred)
    assert deferred_growth <= extra_names < _estimate(many_loaded) - _estimate(few_loaded)


def test_unmarked_tools_estimate_as_before():
    tools = _tools(10, deferred=False)
    marked_false = copy.deepcopy(tools)
    for tool in marked_false:
        tool["function"]["defer_loading"] = False
    assert _estimate(tools) == _estimate(marked_false) > _estimate([])


def test_loading_a_deferred_tool_raises_the_estimate():
    deferred = _tools(10, deferred=True)
    one_loaded = [*deferred[:9], _tool("platform__tool_9", deferred=False)]
    assert _estimate(one_loaded) > _estimate(deferred)

