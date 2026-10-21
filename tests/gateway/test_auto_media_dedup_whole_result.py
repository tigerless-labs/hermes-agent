"""An earlier turn's generated image stays delivered once (``gateway.auto_media_dedup_whole_result``).

A generated image's result names it under several path fields (host, model-visible). The history dedup
records one of them; when mid-run compression shrinks the message list the auto-append scans every
message, and upstream then appends the first field the dedup did not record: an earlier turn's image
rides a later text-only reply. With the switch on, a result whose image is already in history is
skipped as a whole.
"""

import json
from pathlib import Path

import pytest
import yaml

import gateway.run as gateway_run
from gateway.run import _collect_auto_append_media_tags, _collect_history_media_paths

HOST = "/var/lib/hermes/cache/generated/chats/slack-C1/images/cover.webp"
VISIBLE = "/root/.hermes/cache/generated/chats/slack-C1/images/cover.webp"


def _generated(call_id: str, host: str, visible: str) -> list[dict]:
    payload = {"success": True, "image": host, "host_image": host, "agent_visible_image": visible}
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "function": {"name": "image_generate"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload)},
    ]


EARLIER_TURN = [{"role": "user", "content": "write the article"}, *_generated("old", HOST, VISIBLE),
                {"role": "assistant", "content": "submitted for review"}]


def _compacted_turn(*produced: dict) -> list[dict]:
    """What run_conversation returns after compaction: a summary, the kept tail, this turn's messages."""
    return [{"role": "user", "content": "[summary of earlier turns]"}, *EARLIER_TURN[1:],
            {"role": "user", "content": "does it mention our project?"}, *produced,
            {"role": "assistant", "content": "yes, a dozen times"}]


def _appended(messages: list[dict]) -> list[str]:
    tags, _voice = _collect_auto_append_media_tags(
        messages, history_offset=len(EARLIER_TURN) + 40, history_media_paths=_collect_history_media_paths(EARLIER_TURN))
    return tags


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    return tmp_path


@pytest.fixture
def switched_on(home: Path) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({"gateway": {"auto_media_dedup_whole_result": True}}))


def test_the_switch_is_off_by_default_and_known_to_the_config() -> None:
    from hermes_cli.config import _validate_config_key
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["gateway"]["auto_media_dedup_whole_result"] is False
    assert _validate_config_key("gateway.auto_media_dedup_whole_result")[0]


def test_switched_off_upstream_appends_the_earlier_images_other_path(home: Path) -> None:
    assert _appended(_compacted_turn()) == [f"MEDIA:{VISIBLE}"]


def test_switched_on_an_earlier_turns_image_is_not_appended_after_compaction(switched_on: None) -> None:
    assert _appended(_compacted_turn()) == []


def test_switched_on_an_image_made_this_turn_is_still_appended_once(switched_on: None) -> None:
    fresh_host, fresh_visible = HOST.replace("cover", "figure"), VISIBLE.replace("cover", "figure")
    upstream_tags = _collect_auto_append_media_tags(_generated("new", fresh_host, fresh_visible), history_offset=0)[0]
    assert _appended(_compacted_turn(*_generated("new", fresh_host, fresh_visible))) == upstream_tags


def test_switched_on_any_recorded_path_marks_the_image_delivered(switched_on: None) -> None:
    for recorded in (HOST, VISIBLE):
        tags, _voice = _collect_auto_append_media_tags(
            _compacted_turn(), history_offset=len(EARLIER_TURN) + 40, history_media_paths={recorded})
        assert tags == []
