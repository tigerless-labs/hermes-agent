"""Anthropic's count of a request (``messages.count_tokens``) for ``compression.provider_token_count`` (fork, Lara).

The request is built the way the main call builds it (same converters, deferred tools off the wire until loaded)
without the one-shot settings that belong to the real request; only the fields the count endpoint reads are sent.
The count is free and rate-limited apart from message creation.
"""

from __future__ import annotations

from typing import Any, Optional

COUNTED_FIELDS = ("model", "system", "messages", "tools", "tool_choice", "thinking", "extra_headers")


def count_request_tokens(agent: Any, api_messages: list[dict[str, Any]], tools_for_api: Optional[list[dict[str, Any]]],
                         *, timeout: float) -> Optional[int]:
    client = getattr(agent, "_anthropic_client", None)
    if client is None:
        return None
    from agent.responses_tool_search import plain_tools

    compressor = getattr(agent, "context_compressor", None)
    request = agent._get_transport().build_kwargs(
        model=agent.model, messages=agent._prepare_anthropic_messages_for_api(api_messages),
        tools=plain_tools(tools_for_api, api_messages), max_tokens=agent.max_tokens,
        reasoning_config=agent.reasoning_config, is_oauth=agent._is_anthropic_oauth,
        preserve_dots=agent._anthropic_preserve_dots(),
        context_length=getattr(compressor, "context_length", None),
        base_url=getattr(agent, "_anthropic_base_url", None),
        drop_context_1m_beta=bool(getattr(agent, "_oauth_1m_beta_disabled", False)),
    )
    fields = {key: request[key] for key in COUNTED_FIELDS if request.get(key) is not None}
    return client.messages.count_tokens(**fields, timeout=timeout).input_tokens
