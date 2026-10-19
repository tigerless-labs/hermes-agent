"""A deployment can leave Hermes's own help guidance out of the system prompt (``agent.hermes_help_guidance``).

Every system prompt says the agent runs on Hermes and that helping the user configure or extend Hermes goes by
the official docs (or the hermes-agent skill). That holds for one owner on their own machine. On a shared
deployment the people in the chat do not run its configuration, and the guidance has the model teach them to edit
config files and wire MCP servers themselves. With the switch off, neither variant is in the prompt and every
other block stays as it was; unset or true keeps upstream's prompt.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.agent_init import _apply_agent_section
from agent.prompt_builder import HERMES_AGENT_HELP_GUIDANCE, HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS
from agent.system_prompt import build_system_prompt_parts
from tests.agent.test_system_prompt import _make_agent

_HELP_VARIANTS = (HERMES_AGENT_HELP_GUIDANCE, HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS)
_SKILLS_WITH_HERMES_AGENT = "## Skills\n- hermes-agent: Hermes's own commands and workflows"


def _stable(*, with_hermes_agent_skill: bool, **overrides) -> str:
    tools = ["skill_view", "skills_list"] if with_hermes_agent_skill else []
    agent = _make_agent(valid_tool_names=tools, **overrides)
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
        patch("model_tools.get_toolset_for_tool", return_value=None),
        patch("agent.prompt_builder.build_skills_system_prompt",
              return_value=_SKILLS_WITH_HERMES_AGENT if with_hermes_agent_skill else ""),
    ):
        return build_system_prompt_parts(agent)["stable"]


def _without(stable: str, block: str) -> str:
    return stable.replace(f"\n\n{block}", "", 1)


@pytest.mark.parametrize("with_hermes_agent_skill", [True, False])
def test_upstream_prompt_carries_the_help_guidance(with_hermes_agent_skill):
    stable = _stable(with_hermes_agent_skill=with_hermes_agent_skill)
    assert sum(variant in stable for variant in _HELP_VARIANTS) == 1


@pytest.mark.parametrize("with_hermes_agent_skill", [True, False])
def test_switch_off_leaves_out_either_variant(with_hermes_agent_skill):
    stable = _stable(with_hermes_agent_skill=with_hermes_agent_skill, _hermes_help_guidance=False)
    assert not any(variant in stable for variant in _HELP_VARIANTS)


@pytest.mark.parametrize("with_hermes_agent_skill", [True, False])
def test_switch_off_keeps_every_other_block(with_hermes_agent_skill):
    upstream = _stable(with_hermes_agent_skill=with_hermes_agent_skill)
    shown = next(variant for variant in _HELP_VARIANTS if variant in upstream)
    off = _stable(with_hermes_agent_skill=with_hermes_agent_skill, _hermes_help_guidance=False)
    assert off == _without(upstream, shown)


@pytest.mark.parametrize("with_hermes_agent_skill", [True, False])
def test_switch_on_is_upstream(with_hermes_agent_skill):
    assert _stable(with_hermes_agent_skill=with_hermes_agent_skill, _hermes_help_guidance=True) == _stable(
        with_hermes_agent_skill=with_hermes_agent_skill)


def _applied(agent_section: dict) -> bool:
    agent = SimpleNamespace(run_budget_seconds=None)
    _apply_agent_section(agent, {"agent": {"environment_probe": False, **agent_section}})
    return agent._hermes_help_guidance


@pytest.mark.parametrize("setting, expected", [
    ({}, True),
    ({"hermes_help_guidance": None}, True),
    ({"hermes_help_guidance": True}, True),
    ({"hermes_help_guidance": "on"}, True),
    ({"hermes_help_guidance": False}, False),
    ({"hermes_help_guidance": "false"}, False),
    ({"hermes_help_guidance": "off"}, False),
])
def test_config_sets_the_switch(setting, expected):
    assert _applied(setting) is expected
