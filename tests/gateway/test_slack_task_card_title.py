"""Slack-native task-card header comes from ``extra.task_card_title``."""

from types import SimpleNamespace

from tests.gateway.test_slack_approval_buttons import _make_adapter
from gateway.run_turn_runner import TurnRunner, _task_card_title


def test_configured_title_reaches_card_and_fallback_text():
    adapter = _make_adapter()
    adapter.config.extra["task_card_title"] = "Lara is working"
    state = TurnRunner._TaskCardState(adapter)
    state.tasks = {"t": {"id": "t", "title": "terminal", "status": "in_progress"}}
    state.task_order = ["t"]
    assert _task_card_title(adapter) == adapter.config.extra["task_card_title"]
    assert state.fallback_text().splitlines()[0] == adapter.config.extra["task_card_title"]


def test_unset_title_keeps_upstream_default_for_every_adapter():
    slack_default = _task_card_title(_make_adapter())
    assert slack_default
    assert _task_card_title(SimpleNamespace()) == slack_default
