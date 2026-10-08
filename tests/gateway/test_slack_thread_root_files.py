"""``platforms.slack.extra.thread_root_files``: the thread root's files of every kind reach the first
turn in that thread, not only its images."""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import _THREAD_ROOT_IMAGE_MAX, SlackAdapter, _ThreadContextCache

CHANNEL, ROOT_TS, TEAM = "C0TEAM", "1791479438.925359", "T0TEAM"


def _file(name: str, mimetype: str) -> dict:
    return {"id": f"F-{name}", "name": name, "mimetype": mimetype, "size": 100,
            "url_private_download": f"https://files.slack.test/{name}"}


def _adapter(extra: dict, files: list, *, skipped: frozenset = frozenset()):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test", extra=extra))
    root = {"ts": ROOT_TS, "user": "U_BOT", "text": "result", "files": files}
    adapter._thread_context_cache[adapter._thread_cache_key(CHANNEL, ROOT_TS, TEAM)] = _ThreadContextCache(
        content="", messages=[root])
    cached = []

    async def cache_slack_file(kind, f, url, mimetype, team_id):
        cached.append((kind, f["name"]))
        if f["name"] in skipped:
            return None
        return f"/cache/{f['name']}", mimetype, ""

    adapter._cache_slack_file = cache_slack_file
    return adapter, cached


def _collect(adapter):
    return asyncio.run(adapter._collect_thread_root_images(CHANNEL, ROOT_TS, TEAM))


ROOT_FILES = [_file("stock.csv", "text/csv"), _file("chart.png", "image/png"),
              _file("table.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")]


def test_with_the_setting_every_file_on_the_root_reaches_the_first_turn():
    adapter, cached = _adapter({"thread_root_files": 10}, ROOT_FILES)
    paths, types = _collect(adapter)
    assert paths == [f"/cache/{f['name']}" for f in ROOT_FILES]
    assert types == [f["mimetype"] for f in ROOT_FILES]
    assert [kind for kind, _ in cached] == ["document", "image", "document"]


def test_without_the_setting_only_the_root_s_images_come_along():
    adapter, cached = _adapter({}, ROOT_FILES)
    paths, _ = _collect(adapter)
    assert paths == ["/cache/chart.png"] and cached == [("image", "chart.png")]


def test_no_more_files_come_along_than_the_setting_allows():
    files = [_file(f"part-{index}.csv", "text/csv") for index in range(5)]
    adapter, _ = _adapter({"thread_root_files": 2}, files)
    assert _collect(adapter)[0] == ["/cache/part-0.csv", "/cache/part-1.csv"]


def test_a_file_the_cache_turns_down_is_left_out_and_the_rest_still_come():
    adapter, _ = _adapter({"thread_root_files": 10}, ROOT_FILES, skipped=frozenset({"stock.csv"}))
    assert _collect(adapter)[0] == ["/cache/chart.png", "/cache/table.xlsx"]


@pytest.mark.parametrize("setting", [0, -3, "10", True, None, 2.5])
def test_a_setting_that_is_not_a_positive_count_keeps_images_only(setting):
    adapter, _ = _adapter({"thread_root_files": setting}, ROOT_FILES)
    assert _collect(adapter)[0] == ["/cache/chart.png"]


def test_without_the_setting_the_image_cap_still_holds():
    files = [_file(f"shot-{index}.png", "image/png") for index in range(_THREAD_ROOT_IMAGE_MAX + 2)]
    adapter, _ = _adapter({}, files)
    assert len(_collect(adapter)[0]) == _THREAD_ROOT_IMAGE_MAX
