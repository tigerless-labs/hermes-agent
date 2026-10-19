"""Anthropic's advisor tool on the main conversation (fork addition, off unless ``advisor.model`` is set)."""

import copy
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import anthropic_advisor as advisor
from agent.anthropic_adapter import _common_betas_for_base_url, build_anthropic_kwargs
from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.transports.anthropic import AnthropicTransport

NATIVE = "https://api.anthropic.com"
EXECUTOR = "claude-haiku-5-5"
CONFIG = {"advisor": {"model": "claude-opus-5-5", "max_uses": 3, "max_tokens": 2048, "caching": "5m"}}
LOOKUP = {"type": "function", "function": {"name": "lookup", "description": "Count orders.",
                                           "parameters": {"type": "object", "properties": {}}}}
ADVICE = {"type": "advisor_tool_result", "tool_use_id": "srvtoolu_1",
          "content": {"type": "advisor_redacted_result", "encrypted_content": "opaque-advice"}}
CONSULT = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "advisor", "input": {}}
THOUGHT = {"type": "thinking", "thinking": "which orders count?", "signature": "sig-1"}


def built(model=EXECUTOR, base_url=NATIVE):
    return build_anthropic_kwargs(model=model, messages=[{"role": "user", "content": "how many new orders?"}],
                                  tools=[LOOKUP], max_tokens=1000, reasoning_config=None, base_url=base_url)


def betas(kwargs):
    return {b for b in (kwargs.get("extra_headers") or {}).get("anthropic-beta", "").split(",") if b}


def block(data):
    return SimpleNamespace(**data, model_dump=lambda **_: copy.deepcopy(data))


def message(blocks, stop_reason, iterations=()):
    return SimpleNamespace(content=[block(b) for b in blocks], stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5, iterations=list(iterations)))


def test_without_an_advisor_model_the_request_is_exactly_what_upstream_builds():
    kwargs = built()
    before = copy.deepcopy(kwargs)
    advisor.apply_advisor(kwargs, model=EXECUTOR, base_url=NATIVE, is_oauth=False, tool=advisor.advisor_tool({}))
    assert kwargs == before


def test_an_advisor_rides_as_a_server_tool_with_its_beta_and_the_client_betas_kept():
    kwargs = built()
    advisor.apply_advisor(kwargs, model=EXECUTOR, base_url=NATIVE, is_oauth=False, tool=advisor.advisor_tool(CONFIG))
    tool = next(t for t in kwargs["tools"] if t.get("type") == advisor.ADVISOR_TOOL_TYPE)
    assert tool["model"] == CONFIG["advisor"]["model"] and tool["name"] == advisor.ADVISOR_TOOL_NAME
    assert tool["max_uses"] == CONFIG["advisor"]["max_uses"] and tool["max_tokens"] == CONFIG["advisor"]["max_tokens"]
    assert tool["caching"] == {"type": "ephemeral", "ttl": CONFIG["advisor"]["caching"]}
    assert any(t.get("name") == "lookup" for t in kwargs["tools"])
    assert advisor.ADVISOR_BETA in betas(kwargs)
    assert set(_common_betas_for_base_url(NATIVE)) <= betas(kwargs)


def test_a_beta_the_request_already_carries_survives_the_advisor():
    kwargs = built()
    kwargs["extra_headers"] = {"anthropic-beta": "some-beta-2026-01-01"}
    advisor.apply_advisor(kwargs, model=EXECUTOR, base_url=NATIVE, is_oauth=False, tool=advisor.advisor_tool(CONFIG))
    assert {"some-beta-2026-01-01", advisor.ADVISOR_BETA} <= betas(kwargs)


@pytest.mark.parametrize("model,base_url,is_oauth", [
    (EXECUTOR, "https://openrouter.ai/api", False),
    ("muse-spark-1.3", NATIVE, False),
    (EXECUTOR, NATIVE, True),
])
def test_redteam_no_advisor_off_the_native_api_key_route_or_for_a_model_that_is_not_claude(model, base_url, is_oauth):
    kwargs = built(model=model, base_url=base_url)
    before = copy.deepcopy(kwargs)
    advisor.apply_advisor(kwargs, model=model, base_url=base_url, is_oauth=is_oauth, tool=advisor.advisor_tool(CONFIG))
    assert kwargs == before


def test_a_reply_that_consulted_the_advisor_keeps_every_block_in_order():
    normalized = AnthropicTransport().normalize_response(
        message([THOUGHT, CONSULT, ADVICE, {"type": "text", "text": "6 paid orders"}], "end_turn"))
    ordered = normalized.provider_data["anthropic_content_blocks"]
    assert [b["type"] for b in ordered] == ["thinking", "server_tool_use", "advisor_tool_result", "text"]
    assert next(b for b in ordered if b["type"] == "advisor_tool_result")["content"] == ADVICE["content"]


def test_the_advice_goes_back_on_the_wire_verbatim_in_its_place_without_output_only_fields():
    stored = [THOUGHT, {**CONSULT, "caller": {"type": "direct"}}, {**ADVICE, "citations": None},
              {"type": "text", "text": "6 paid orders"}]
    _, wire = convert_messages_to_anthropic([
        {"role": "user", "content": "how many new orders?"},
        {"role": "assistant", "content": "6 paid orders", "anthropic_content_blocks": stored},
        {"role": "user", "content": "and last year?"},
    ], base_url=NATIVE, model=EXECUTOR)
    replayed = wire[1]["content"]
    assert [b["type"] for b in replayed] == ["thinking", "server_tool_use", "advisor_tool_result", "text"]
    assert replayed[1] == CONSULT and replayed[2] == ADVICE


def test_a_turn_paused_on_the_advisor_is_resent_and_the_parts_join_into_one_reply():
    paused = message([THOUGHT, CONSULT], advisor.PAUSED)
    done = message([ADVICE, {"type": "text", "text": "6 paid orders"}], "end_turn")
    sent = []

    def send(kwargs):
        sent.append(kwargs)
        return done

    kwargs = built()
    advisor.apply_advisor(kwargs, model=EXECUTOR, base_url=NATIVE, is_oauth=False, tool=advisor.advisor_tool(CONFIG))
    joined = advisor.resume_paused(send, kwargs, paused)
    assert joined.stop_reason == "end_turn"
    assert [b.type for b in joined.content] == ["thinking", "server_tool_use", "advisor_tool_result", "text"]
    assert sent[0]["messages"][-1] == {"role": "assistant", "content": [THOUGHT, CONSULT]}
    assert sent[0]["messages"][:-1] == kwargs["messages"]


def test_a_paused_turn_is_resent_at_most_as_often_as_the_advisor_may_be_used():
    paused = message([CONSULT], advisor.PAUSED)
    send = MagicMock(return_value=paused)
    kwargs = built()
    advisor.apply_advisor(kwargs, model=EXECUTOR, base_url=NATIVE, is_oauth=False, tool=advisor.advisor_tool(CONFIG))
    advisor.resume_paused(send, kwargs, paused)
    assert send.call_count <= CONFIG["advisor"]["max_uses"]


def test_without_an_advisor_a_paused_turn_is_left_to_upstream():
    paused = message([CONSULT], advisor.PAUSED)
    send = MagicMock()
    assert advisor.resume_paused(send, built(), paused) is paused
    send.assert_not_called()


def test_what_the_advisor_spent_is_logged_on_its_own_line(caplog):
    reply = message([ADVICE], "end_turn", iterations=[
        {"type": "message", "input_tokens": 1224, "output_tokens": 499},
        {"type": "advisor_message", "model": "claude-opus-5-5", "input_tokens": 3060, "output_tokens": 751},
    ])
    with caplog.at_level(logging.INFO, logger=advisor.logger.name):
        advisor.log_advisor_usage(reply)
    assert "Advisor call: model=claude-opus-5-5 in=3060 out=751" in caplog.text


def _agent(model=EXECUTOR):
    from run_agent import AIAgent
    agent = AIAgent(api_key="test-key", base_url=NATIVE, model=model, quiet_mode=True, skip_context_files=True,
                    skip_memory=True)
    agent.api_mode = "anthropic_messages"
    return agent


@pytest.mark.parametrize("config,expected", [(CONFIG, True), ({}, False)])
def test_the_main_conversation_carries_the_advisor_only_when_one_is_configured(monkeypatch, config, expected):
    from agent import chat_completion_helpers
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    kwargs = chat_completion_helpers._build_anthropic_kwargs(
        _agent(), [{"role": "user", "content": "how many new orders?"}], [LOOKUP], None, {})
    assert (advisor.advisor_in(kwargs) is not None) is expected
