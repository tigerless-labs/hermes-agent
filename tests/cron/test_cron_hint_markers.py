"""``cron.marker_hints``: false drops the parts of the cron hint that ask the model to answer with an in-text
marker: ``[SILENT]`` when there is nothing to report, ``[CRON_FAILURE]`` when a delegated child failed. It is for
a deployment that delivers its runs itself and reads neither marker. The rest of the hint (delivery, recursion)
stays. Unset or true, the hint is upstream's. Driven through the real ``_build_job_prompt`` with the setting read
from ``config.yaml``.
"""

import pytest
import yaml

from cron import scheduler_prompt
from tests.cron.test_cron_script import cron_env  # noqa: F401 - fixture

JOB = {"id": "leads", "prompt": "Report new leads only."}
MARKERS = ("SILENT", "CRON_FAILURE")


def _prompt(cron_env, cron_config):  # noqa: F811
    from cron.scheduler import _build_job_prompt

    (cron_env / "config.yaml").write_text(
        yaml.safe_dump({} if cron_config is None else {"cron": cron_config}), encoding="utf-8")
    return _build_job_prompt(JOB)


def test_switched_off_the_hint_never_mentions_a_marker(cron_env):  # noqa: F811
    prompt = _prompt(cron_env, {"marker_hints": False})
    assert not any(marker in prompt for marker in MARKERS)
    assert JOB["prompt"] in prompt


@pytest.mark.parametrize("cron_config", [None, {}, {"marker_hints": True}], ids=["no-cron", "unset", "on"])
def test_left_alone_the_hint_is_upstreams(cron_env, cron_config):  # noqa: F811
    prompt = _prompt(cron_env, cron_config)
    assert prompt.startswith(scheduler_prompt._CRON_HINT)
    assert all(marker in prompt for marker in MARKERS)


def test_only_the_marker_parts_go(cron_env):  # noqa: F811
    upstream = _prompt(cron_env, None)
    switched_off = _prompt(cron_env, {"marker_hints": False})
    parts = scheduler_prompt._CRON_HINT_MARKER_PARTS
    assert all(part in scheduler_prompt._CRON_HINT for part in parts)
    remaining = upstream
    for part in parts:
        remaining = remaining.replace(part, "", 1)
    assert remaining == switched_off


@pytest.mark.parametrize("value", ["false", "off", 0], ids=["text-false", "text-off", "zero"])
def test_a_false_written_another_way_also_switches_it_off(cron_env, value):  # noqa: F811
    prompt = _prompt(cron_env, {"marker_hints": value})
    assert not any(marker in prompt for marker in MARKERS)
