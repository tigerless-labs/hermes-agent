"""Client-executed tool search on the Responses API: a plugin declares deferred tools in
namespaces plus one search tool; the model loads definitions by name and calls them natively.
The tool array never changes while tools load (prompt-cache prefix), every search and its
result replays exactly, and transports without tool search fall back to appending what was
loaded to the end of the tool array."""

import json
from types import SimpleNamespace

import pytest

from agent.responses_tool_search import plain_tools
from agent.transports.codex import ResponsesApiTransport

SEARCH = "load_tools"
NAMESPACE = {"name": "notes", "description": "The notes platform, as this person."}
ARGUMENTS = {"type": "object", "properties": {"note_id": {"type": "integer"}}, "required": ["note_id"]}
NAMES = {"type": "object", "properties": {"names": {"type": "array", "items": {"type": "string"}}},
         "required": ["names"]}


def _function(name, *, description=None, parameters=None, **marks):
    function = {"name": name, "parameters": parameters or ARGUMENTS, **marks}
    if description is not None:
        function["description"] = description
    return {"type": "function", "function": function}


def _deferred(name, namespace=NAMESPACE):
    marks = {"defer_loading": True}
    if namespace is not None:
        marks["namespace"] = namespace
    return _function(name, **marks)


def _full(name, description="Read one note by its number."):
    return {"name": name, "description": description, "parameters": ARGUMENTS}


TOOLS = [
    _function(SEARCH, description="Load tools by their names.", parameters=NAMES, tool_search=True),
    _function("terminal", description="Run a command."),
    _deferred("notes__read_note"),
    _deferred("notes__list_notes"),
    _deferred("loose_tool", namespace=None),
]


def _call(call_id, name, arguments):
    return {"id": call_id, "call_id": call_id, "response_item_id": f"fc_{call_id.removeprefix('call_')}",
            "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _history(found=None, search_result=None):
    search_arguments = {"names": ["notes.read_note"]}
    result = search_result if search_result is not None else json.dumps({"tools": found or [_full("notes__read_note")]})
    return [
        {"role": "user", "content": "What does note 7 say?"},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_search1", SEARCH, search_arguments)]},
        {"role": "tool", "tool_call_id": "call_search1", "content": result},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_read1", "notes__read_note", {"note_id": 7})]},
        {"role": "tool", "tool_call_id": "call_read1", "content": "Buy lemons."},
    ]


def _build(messages, tools=TOOLS):
    transport = ResponsesApiTransport()
    return transport, transport.build_kwargs(model="muse-spark-1.3", messages=messages, tools=tools,
                                             base_url="https://api.meta.ai/v1", session_id="s1")


def _by_type(items, kind):
    return [item for item in items if item.get("type") == kind]


def _function_names(kwargs):
    names = []
    for tool in kwargs["tools"]:
        if tool["type"] == "namespace":
            names += [f"{tool['name']}.{function['name']}" for function in tool["tools"]]
        elif tool["type"] == "function":
            names.append(tool["name"])
    return names


def test_deferred_tools_go_out_in_their_namespaces_beside_one_client_search():
    _, kwargs = _build([{"role": "user", "content": "hi"}])
    search, = _by_type(kwargs["tools"], "tool_search")
    assert search["execution"] == "client" and search["parameters"] == NAMES and search["description"]
    namespace, = _by_type(kwargs["tools"], "namespace")
    assert {key: namespace[key] for key in ("name", "description")} == NAMESPACE
    assert {function["name"] for function in namespace["tools"]} == {"read_note", "list_notes"}
    assert all(function["defer_loading"] is True and "description" not in function for function in namespace["tools"])
    loose, = [tool for tool in _by_type(kwargs["tools"], "function") if tool["name"] == "loose_tool"]
    assert loose["defer_loading"] is True
    terminal, = [tool for tool in _by_type(kwargs["tools"], "function") if tool["name"] == "terminal"]
    assert "defer_loading" not in terminal
    assert SEARCH not in _function_names(kwargs)


def test_no_plugin_mark_reaches_the_wire():
    _, kwargs = _build([{"role": "user", "content": "hi"}])
    wire = json.dumps(kwargs["tools"])
    assert '"namespace": {' not in wire and '"tool_search": true' not in wire


def test_the_tool_array_and_cache_key_stay_the_same_while_tools_load():
    _, before = _build([{"role": "user", "content": "What does note 7 say?"}])
    _, after = _build(_history())
    assert before["tools"] == after["tools"]
    assert before["prompt_cache_key"] == after["prompt_cache_key"]


def _output_response(*items):
    return SimpleNamespace(status="completed", output=list(items))


def test_a_search_and_a_namespaced_call_come_back_as_the_plugins_tools():
    transport, _ = _build([{"role": "user", "content": "hi"}])
    normalized = transport.normalize_response(_output_response(
        SimpleNamespace(type="tool_search_call", id="tsc_call_1", call_id="call_1", execution="client",
                        arguments={"names": ["notes.read_note"]}, status="completed"),
        SimpleNamespace(type="function_call", id="fc_2", call_id="call_2", name="notes.read_note",
                        arguments='{"note_id": 7}', status="completed"),
        SimpleNamespace(type="function_call", id="fc_3", call_id="call_3", name="list_notes", namespace="notes",
                        arguments="{}", status="completed"),
    ))
    calls = [(call.name, json.loads(call.arguments)) for call in normalized.tool_calls]
    assert calls == [(SEARCH, {"names": ["notes.read_note"]}), ("notes__read_note", {"note_id": 7}),
                     ("notes__list_notes", {})]
    assert normalized.tool_calls[0].provider_data["call_id"] == "call_1"


def test_a_server_executed_search_is_not_handed_to_the_plugin():
    transport, _ = _build([{"role": "user", "content": "hi"}])
    normalized = transport.normalize_response(_output_response(
        SimpleNamespace(type="tool_search_call", id="tsc_1", call_id="call_1", execution="server",
                        arguments={"query": "notes"}, status="completed"),
        SimpleNamespace(type="message", role="assistant", status="completed", id="msg_1",
                        content=[SimpleNamespace(type="output_text", text="done")]),
    ))
    assert not normalized.tool_calls


def test_a_search_and_its_result_replay_as_the_search_items_the_model_saw():
    _, kwargs = _build(_history())
    items = kwargs["input"]
    call, = _by_type(items, "tool_search_call")
    output, = _by_type(items, "tool_search_output")
    assert call["call_id"] == output["call_id"]
    assert call["execution"] == output["execution"] == "client"
    assert call["arguments"] == {"names": ["notes.read_note"]}
    namespace, = output["tools"]
    assert {key: namespace[key] for key in ("type", "name", "description")} == {"type": "namespace", **NAMESPACE}
    loaded, = namespace["tools"]
    assert loaded["name"] == "read_note" and loaded["defer_loading"] is True
    assert {key: loaded[key] for key in ("description", "parameters")} == {
        key: _full("notes__read_note")[key] for key in ("description", "parameters")}
    assert items.index(call) < items.index(output)


def test_a_namespaced_call_replays_under_its_dotted_wire_name():
    _, kwargs = _build(_history())
    call, = _by_type(kwargs["input"], "function_call")
    result, = _by_type(kwargs["input"], "function_call_output")
    assert call["name"] == "notes.read_note" and call["call_id"] == result["call_id"]
    assert json.loads(call["arguments"]) == {"note_id": 7}


def test_the_built_request_passes_preflight_unchanged():
    transport, kwargs = _build(_history())
    checked = transport.preflight_kwargs(kwargs)
    assert checked["tools"] == kwargs["tools"]
    assert checked["input"] == kwargs["input"]


def test_a_found_tool_without_a_namespace_loads_as_a_deferred_function():
    _, kwargs = _build(_history(found=[_full("loose_tool")]))
    output, = _by_type(kwargs["input"], "tool_search_output")
    loaded, = output["tools"]
    assert loaded["type"] == "function" and loaded["name"] == "loose_tool" and loaded["defer_loading"] is True


@pytest.mark.parametrize("found", [
    [_full("terminal")],
    [_full("notes__never_declared")],
    [_full("other__read_note")],
    ["not a definition"],
])
def test_redteam_a_search_result_loads_only_tools_declared_deferred(found):
    _, kwargs = _build(_history(found=found))
    output, = _by_type(kwargs["input"], "tool_search_output")
    assert output["tools"] == []


@pytest.mark.parametrize("result", ["not json", json.dumps(["list"]), json.dumps({"tools": "x"}), ""])
def test_redteam_an_unreadable_search_result_loads_nothing_and_keeps_the_pair(result):
    _, kwargs = _build(_history(search_result=result))
    call, = _by_type(kwargs["input"], "tool_search_call")
    output, = _by_type(kwargs["input"], "tool_search_output")
    assert output["tools"] == [] and output["call_id"] == call["call_id"]


@pytest.mark.parametrize("name", ["evil.read_note", "notes.missing", "terminal.x", "notes.read_note.x"])
def test_redteam_a_dotted_name_outside_the_declarations_maps_to_no_tool(name):
    transport, _ = _build([{"role": "user", "content": "hi"}])
    normalized = transport.normalize_response(_output_response(
        SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name=name, arguments="{}",
                        status="completed")))
    called, = normalized.tool_calls
    assert called.name not in {tool["function"]["name"] for tool in TOOLS}


def test_redteam_a_namespace_field_cannot_redirect_a_call_to_a_plain_tool():
    transport, _ = _build([{"role": "user", "content": "hi"}])
    normalized = transport.normalize_response(_output_response(
        SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name="terminal", namespace="notes",
                        arguments="{}", status="completed")))
    called, = normalized.tool_calls
    assert called.name != "terminal"


def test_without_a_search_tool_the_responses_wire_falls_back_to_plain_loading():
    tools = [tool for tool in TOOLS if tool["function"]["name"] != SEARCH]
    _, kwargs = _build([{"role": "user", "content": "hi"}], tools=tools)
    assert not _by_type(kwargs["tools"], "namespace") and not _by_type(kwargs["tools"], "tool_search")
    assert _function_names(kwargs) == ["terminal"]


def _names(tools):
    return [tool["function"]["name"] for tool in tools]


def test_plain_loading_declares_the_search_and_the_eager_tools_and_lists_what_can_load():
    tools = plain_tools(TOOLS, [{"role": "user", "content": "hi"}])
    assert _names(tools) == [SEARCH, "terminal"]
    search = tools[0]["function"]
    for name in ("notes__read_note", "notes__list_notes", "loose_tool"):
        assert name in search["description"]
    assert NAMESPACE["description"] in search["description"]
    assert not any(mark in tool["function"] for tool in tools for mark in ("defer_loading", "namespace", "tool_search"))


def test_plain_loading_appends_what_was_loaded_in_load_order_and_keeps_the_prefix():
    first = _history(found=[_full("notes__read_note")])
    loaded_once = plain_tools(TOOLS, first)
    assert _names(loaded_once) == [SEARCH, "terminal", "notes__read_note"]
    assert loaded_once[-1]["function"]["description"] == _full("notes__read_note")["description"]
    second = first + [
        {"role": "assistant", "content": "", "tool_calls": [_call("call_search2", SEARCH, {"names": ["x"]})]},
        {"role": "tool", "tool_call_id": "call_search2",
         "content": json.dumps({"tools": [_full("loose_tool"), _full("notes__read_note", "Changed.")]})},
    ]
    loaded_twice = plain_tools(TOOLS, second)
    assert loaded_twice[:len(loaded_once)] == loaded_once
    assert _names(loaded_twice) == [SEARCH, "terminal", "notes__read_note", "loose_tool"]


@pytest.mark.parametrize("found", [[_full("terminal", "Hijacked.")], [_full("never_declared")]])
def test_redteam_plain_loading_appends_only_tools_declared_deferred(found):
    tools = plain_tools(TOOLS, _history(found=found))
    assert _names(tools) == [SEARCH, "terminal"]
    assert tools[1]["function"]["description"] == "Run a command."


def test_redteam_only_results_of_the_search_tool_load_anything():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_t1", "terminal", {"note_id": 1})]},
        {"role": "tool", "tool_call_id": "call_t1", "content": json.dumps({"tools": [_full("notes__read_note")]})},
    ]
    assert _names(plain_tools(TOOLS, history)) == [SEARCH, "terminal"]


def test_plain_loading_leaves_unmarked_tools_untouched():
    unmarked = [_function("terminal", description="Run a command.")]
    assert plain_tools(unmarked, []) == unmarked


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages", "bedrock_converse"])
def test_interfaces_without_tool_search_get_the_plain_tool_array(monkeypatch, api_mode):
    from agent import chat_completion_helpers as helpers

    sent = {}

    def capture(agent, api_messages, tools_for_api, *rest):
        sent["tools"] = tools_for_api
        return {}

    for builder in ("_build_chat_completions_kwargs", "_build_anthropic_kwargs", "_build_bedrock_kwargs"):
        monkeypatch.setattr(helpers, builder, capture)
    monkeypatch.setattr(helpers, "_reasoning_config_for_wire", lambda agent: None)
    monkeypatch.setattr(helpers, "effective_request_overrides", lambda agent: {})
    monkeypatch.setattr(helpers, "_prompt_cache_scope_for_agent", lambda agent: "scope")
    agent = SimpleNamespace(api_mode=api_mode, tools=TOOLS)
    history = _history(found=[_full("notes__read_note")])
    helpers._build_api_kwargs_for_mode(agent, history)
    assert sent["tools"] == plain_tools(TOOLS, history)


def test_the_responses_interface_gets_the_declarations_untouched(monkeypatch):
    from agent import chat_completion_helpers as helpers

    sent = {}
    monkeypatch.setattr(helpers, "_build_codex_kwargs",
                        lambda agent, api_messages, tools_for_api, *rest: sent.setdefault("tools", tools_for_api))
    monkeypatch.setattr(helpers, "_reasoning_config_for_wire", lambda agent: None)
    monkeypatch.setattr(helpers, "effective_request_overrides", lambda agent: {})
    monkeypatch.setattr(helpers, "_prompt_cache_scope_for_agent", lambda agent: "scope")
    helpers._build_api_kwargs_for_mode(SimpleNamespace(api_mode="codex_responses", tools=TOOLS), _history())
    assert sent["tools"] is TOOLS
