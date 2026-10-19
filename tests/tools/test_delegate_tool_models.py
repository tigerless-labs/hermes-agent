"""Per-task model choice for delegate_task (fork addition, off unless ``delegation.models`` lists models)."""

import copy
import json
import threading
from unittest.mock import MagicMock

import pytest

from tools import delegate_tool
from tools import delegate_tool_models as models

DEFAULT = "claude-sonnet-5-5"
STRONGER = "claude-opus-5-5"
CHOOSING = {"provider": "anthropic", "model": DEFAULT, "models": [DEFAULT, STRONGER]}
PINNED = {"provider": "anthropic", "model": DEFAULT}
CREDS = {"model": DEFAULT, "provider": "anthropic", "base_url": "https://api.anthropic.com", "api_key": "test-key",
         "api_mode": "anthropic_messages"}
GOAL = "Research the main competitors' student health plans and compare their prices"


def _parent():
    parent = MagicMock()
    parent.model = "claude-haiku-5-5"
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    return parent


def _task_fields(monkeypatch, cfg):
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: cfg)
    return delegate_tool._build_dynamic_schema_overrides()["parameters"]["properties"]["tasks"]["items"]["properties"]


def test_without_a_model_list_the_schema_is_exactly_what_upstream_offers(monkeypatch):
    assert models.MODEL_FIELD not in _task_fields(monkeypatch, PINNED)


def test_with_a_model_list_each_task_may_name_one_of_those_models_and_only_those(monkeypatch):
    field = _task_fields(monkeypatch, CHOOSING)[models.MODEL_FIELD]
    assert field["enum"] == CHOOSING["models"]
    assert CHOOSING["model"] in field["description"]


def test_the_static_schema_is_never_changed_by_offering_the_choice(monkeypatch):
    before = copy.deepcopy(delegate_tool.DELEGATE_TASK_SCHEMA)
    _task_fields(monkeypatch, CHOOSING)
    assert delegate_tool.DELEGATE_TASK_SCHEMA == before


@pytest.mark.parametrize("task,expected", [
    ({"goal": GOAL, "model": STRONGER}, STRONGER),
    ({"goal": GOAL, "model": DEFAULT}, DEFAULT),
    ({"goal": GOAL}, DEFAULT),
])
def test_each_child_runs_on_the_model_its_task_chose_or_on_the_delegation_s_model(monkeypatch, task, expected):
    built = []
    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools",
                        lambda **kwargs: built.append(kwargs["model"]) or MagicMock())
    children, error = delegate_tool._build_children(
        [task], [None], CREDS, top_role="leaf", max_iterations=10, parent_agent=_parent(), routing_cfg=CHOOSING,
        live_deleg_id=None, live_writers=[])
    assert error is None and built == [expected]


def test_without_a_model_list_a_named_model_is_ignored_as_upstream_ignores_unknown_fields(monkeypatch):
    built = []
    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools",
                        lambda **kwargs: built.append(kwargs["model"]) or MagicMock())
    delegate_tool._build_children(
        [{"goal": GOAL, "model": STRONGER}], [None], CREDS, top_role="leaf", max_iterations=10,
        parent_agent=_parent(), routing_cfg=PINNED, live_deleg_id=None, live_writers=[])
    assert built == [CREDS["model"]]


@pytest.mark.parametrize("unlisted", ["claude-opus-5-5-contributor", "gpt-6.1-sol", "anthropic/claude-opus-5-5"])
def test_redteam_a_model_off_the_list_is_refused_before_any_child_is_built(monkeypatch, unlisted):
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: CHOOSING)
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", lambda cfg, parent: dict(CREDS))
    builder = MagicMock()
    monkeypatch.setattr(delegate_tool, "_build_children", builder)
    reply = json.loads(delegate_tool.delegate_task(tasks=[{"goal": GOAL, "model": unlisted}], parent_agent=_parent()))
    assert "error" in reply and unlisted in reply["error"]
    builder.assert_not_called()
