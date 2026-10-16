"""``cron.silent_marker_hint``: false drops the part of the cron hint that asks the model to answer
``[SILENT]`` when there is nothing to report. It is for a deployment whose runs say that another way. The
rest of the hint (delivery, failure, recursion) stays. Unset or true, the hint is upstream's. Driven
through the real ``_build_job_prompt`` with the setting read from ``config.yaml``.
"""

import pytest
import yaml

from cron import scheduler_prompt
from tests.cron.test_cron_script import cron_env  # noqa: F401 - fixture

JOB = {"id": "leads", "prompt": "Report new leads only."}


def _prompt(cron_env, cron_config):  # noqa: F811
    from cron.scheduler import _build_job_prompt

    (cron_env / "config.yaml").write_text(
        yaml.safe_dump({} if cron_config is None else {"cron": cron_config}), encoding="utf-8")
    return _build_job_prompt(JOB)


def test_switched_off_the_hint_never_mentions_the_silent_marker(cron_env):  # noqa: F811
    prompt = _prompt(cron_env, {"silent_marker_hint": False})
    assert "SILENT" not in prompt
    assert JOB["prompt"] in prompt


@pytest.mark.parametrize("cron_config", [None, {}, {"silent_marker_hint": True}], ids=["no-cron", "unset", "on"])
def test_left_alone_the_hint_is_upstreams(cron_env, cron_config):  # noqa: F811
    assert _prompt(cron_env, cron_config).startswith(scheduler_prompt._CRON_HINT)


def test_only_the_silent_part_goes(cron_env):  # noqa: F811
    upstream = _prompt(cron_env, None)
    switched_off = _prompt(cron_env, {"silent_marker_hint": False})
    assert scheduler_prompt._CRON_HINT_SILENT in scheduler_prompt._CRON_HINT
    assert upstream.replace(scheduler_prompt._CRON_HINT_SILENT, "", 1) == switched_off


@pytest.mark.parametrize("value", ["false", "off", 0], ids=["text-false", "text-off", "zero"])
def test_a_false_written_another_way_also_switches_it_off(cron_env, value):  # noqa: F811
    assert "SILENT" not in _prompt(cron_env, {"silent_marker_hint": value})
