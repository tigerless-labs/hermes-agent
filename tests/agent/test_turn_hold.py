"""``PluginContext.hold_turn``: a tool holds its turn for a person's answer.

A plugin tool that leaves its call waiting on a person (an approval card, a sign-in) asks for the
turn to be held. The round's other calls still run; then the turn ends: the model is not called
again, the reply is a silence marker the gateway never posts, and the result says the turn is held,
so a turn that must be answered may end without words. Only a plugin whose own profile config
allows it can hold a turn, and only a top-level turn is held.
"""
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import turn_hold
from gateway.response_filters import is_intentional_silence_response
from hermes_cli.plugins_turn_hold import ALLOW_TURN_HOLD, hold_plugin_turn
from run_agent import AIAgent

SESSION = "turn-hold-session"


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": name, "description": f"{name} tool",
                                              "parameters": {"type": "object", "properties": {}}}}
            for name in names]


def _call(name):
    return SimpleNamespace(id=f"call_{uuid.uuid4().hex[:8]}", type="function",
                           function=SimpleNamespace(name=name, arguments="{}"))


def _response(content="", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    finish = "tool_calls" if tool_calls else "stop"
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           model="test/model", usage=None)


@pytest.fixture(autouse=True)
def _no_requests_left_over():
    turn_hold.forget(SESSION)
    yield
    turn_hold.forget(SESSION)


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs("send_mail", "read_mail")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id=SESSION, api_key="test-key", base_url="https://example.invalid/v1",
            provider="openai-compat", model="test/model", max_iterations=6, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    return instance


def _script(agent, *answers):
    asked, queue = [], iter(answers)

    def model_call(api_kwargs):
        asked.append(api_kwargs.get("messages") or api_kwargs.get("input") or [])
        return next(queue)

    agent._interruptible_api_call = model_call
    return asked


def _run(agent, message, tools, *, required=True):
    """``tools`` maps a tool name to what its handler does with the calling session's id."""
    ran = []

    def handle(function_name, function_args, *_a, session_id=None, **_kw):
        ran.append(function_name)
        return json.dumps(tools[function_name](session_id))

    agent._reply_required = required
    with patch("model_tools.handle_function_call", side_effect=handle), \
            patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        return agent.run_conversation(message), ran


def _waits(session_id):
    turn_hold.request(session_id)
    return {"status": "waiting_for_approval"}


def _runs(_session_id):
    return {"status": "ok"}


def test_a_held_turn_ends_after_its_tool_round_without_another_model_call(agent):
    asked = _script(agent, _response(tool_calls=[_call("send_mail")]), _response("It waits for your approval."))
    result, ran = _run(agent, "@lara send the report", {"send_mail": _waits})
    assert len(asked) == 1 and ran == ["send_mail"]
    assert result[turn_hold.RESULT_KEY] is True and result["completed"] and not result.get("failed")
    assert is_intentional_silence_response(result["final_response"])


def test_a_held_turn_leaves_the_call_and_its_answer_in_the_transcript_and_closes_on_an_assistant_row(agent):
    _script(agent, _response(tool_calls=[_call("send_mail")]))
    result, _ = _run(agent, "@lara send the report", {"send_mail": _waits})
    roles = [message["role"] for message in result["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert json.loads(result["messages"][2]["content"]) == {"status": "waiting_for_approval"}
    assert result["messages"][-1]["content"] == result["final_response"]


def test_the_rounds_other_calls_still_run_before_a_held_turn_ends(agent):
    asked = _script(agent, _response(tool_calls=[_call("read_mail"), _call("send_mail")]))
    result, ran = _run(agent, "@lara read then send", {"send_mail": _waits, "read_mail": _runs})
    assert sorted(ran) == ["read_mail", "send_mail"] and len(asked) == 1
    assert [message["role"] for message in result["messages"]].count("tool") == 2


def test_a_turn_no_tool_held_goes_on_to_the_models_reply(agent):
    asked = _script(agent, _response(tool_calls=[_call("read_mail")]), _response("Two new mails."))
    result, _ = _run(agent, "@lara any mail?", {"read_mail": _runs})
    assert len(asked) == 2 and result["final_response"] == "Two new mails."
    assert turn_hold.RESULT_KEY not in result


def test_the_turn_after_a_held_one_runs_as_usual(agent):
    asked = _script(agent, _response(tool_calls=[_call("send_mail")]), _response("Sent."))
    held, _ = _run(agent, "@lara send the report", {"send_mail": _waits})
    resumed, _ = _run(agent, "[approved: sent]", {})
    assert held[turn_hold.RESULT_KEY] is True
    assert resumed["final_response"] == "Sent." and turn_hold.RESULT_KEY not in resumed and len(asked) == 2


def test_redteam_a_request_left_from_outside_a_turn_does_not_end_the_next_turn(agent):
    turn_hold.request(SESSION)
    asked = _script(agent, _response(tool_calls=[_call("read_mail")]), _response("Two new mails."))
    result, _ = _run(agent, "@lara any mail?", {"read_mail": _runs})
    assert len(asked) == 2 and result["final_response"] == "Two new mails."


def test_redteam_a_request_for_another_session_does_not_hold_this_turn(agent):
    asked = _script(agent, _response(tool_calls=[_call("send_mail")]), _response("Done."))
    result, _ = _run(agent, "@lara send", {"send_mail": lambda _sid: _waits("someone-elses-session")})
    turn_hold.forget("someone-elses-session")
    assert len(asked) == 2 and result["final_response"] == "Done."


def test_redteam_a_subagents_turn_is_not_held(agent):
    agent._parent_session_id = "the-parent-session"
    asked = _script(agent, _response(tool_calls=[_call("send_mail")]), _response("It waits for approval."))
    result, _ = _run(agent, "send the report", {"send_mail": _waits})
    assert len(asked) == 2 and result["final_response"] == "It waits for approval."
    assert turn_hold.RESULT_KEY not in result


def test_redteam_the_model_writing_the_marker_itself_is_not_a_held_turn(agent):
    _script(agent, _response(turn_hold.HELD_REPLY), _response(turn_hold.HELD_REPLY))
    result, _ = _run(agent, "@lara status?", {})
    assert turn_hold.RESULT_KEY not in result and not turn_hold.held(result)


def _plugin(tmp_path: Path, plugin_id="lara"):
    return SimpleNamespace(plugin_id=plugin_id, _manager=SimpleNamespace(home_path=tmp_path))


def _allowing(setting):
    return patch("hermes_cli.plugins_turn_hold.load_config_readonly",
                 return_value={"plugins": {"entries": {"lara": setting}}})


def test_a_plugin_its_profile_allows_holds_the_turn_of_the_session_it_names(tmp_path):
    with _allowing({ALLOW_TURN_HOLD: True}):
        assert hold_plugin_turn(_plugin(tmp_path), SESSION) is True
    assert turn_hold.take(SimpleNamespace(session_id=SESSION, _parent_session_id=None)) is True


@pytest.mark.parametrize("setting", [{}, {ALLOW_TURN_HOLD: False}, {ALLOW_TURN_HOLD: "true"},
                                     {"allow_gateway_interrupt": True}])
def test_redteam_a_plugin_its_profile_does_not_allow_holds_nothing(tmp_path, setting):
    with _allowing(setting):
        assert hold_plugin_turn(_plugin(tmp_path), SESSION) is False
    assert turn_hold.take(SimpleNamespace(session_id=SESSION, _parent_session_id=None)) is False


@pytest.mark.parametrize("session_id", ["", None, 7])
def test_redteam_a_hold_without_a_session_id_holds_nothing(tmp_path, session_id):
    with _allowing({ALLOW_TURN_HOLD: True}):
        assert hold_plugin_turn(_plugin(tmp_path), session_id) is False


def test_redteam_an_unreadable_profile_config_denies_the_hold(tmp_path):
    with patch("hermes_cli.plugins_turn_hold.load_config_readonly", side_effect=OSError("unreadable")):
        assert hold_plugin_turn(_plugin(tmp_path), SESSION) is False
    assert turn_hold.take(SimpleNamespace(session_id=SESSION, _parent_session_id=None)) is False


def test_the_plugin_context_offers_the_hold():
    from hermes_cli.plugins import PluginContext

    ctx = MagicMock(spec=PluginContext)
    with patch("hermes_cli.plugins_turn_hold.hold_plugin_turn", return_value=True) as hold:
        assert PluginContext.hold_turn(ctx, SESSION) is True
    hold.assert_called_once_with(ctx, SESSION)
