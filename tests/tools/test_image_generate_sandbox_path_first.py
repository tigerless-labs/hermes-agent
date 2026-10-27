"""A sandboxed model gets the generated image's sandbox path first (``image_gen.sandbox_path_first``).

The result names the file twice under a sandboxed terminal: ``image`` (the host path the gateway delivers) and
``agent_visible_image`` (where the sandbox sees it). Models took ``image`` first, found nothing at that path in
the sandbox and only then tried the other field: one wasted call on every generated image. With the switch on,
``image`` is the sandbox path; ``host_image`` still carries the host path, and the gateway reads that one first.
"""

import json
from types import SimpleNamespace

import pytest

import hermes_cli.config as hermes_config
from tools import image_generation_tool


@pytest.fixture
def generated(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    image_dir = hermes_home / "cache" / "images"
    image_dir.mkdir(parents=True)
    image = image_dir / "meta_cover.webp"
    image.write_bytes(b"webp")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def produce(*, switch, sandboxed=True):
        image_gen = {} if switch is None else {"sandbox_path_first": switch}
        monkeypatch.setattr(hermes_config, "load_config", lambda: {"image_gen": image_gen})
        env = SimpleNamespace(_remote_home="/root") if sandboxed else None
        monkeypatch.setattr(image_generation_tool, "_active_terminal_env", lambda task_id: env)
        if not sandboxed:
            monkeypatch.setattr(image_generation_tool, "_agent_cache_base_for_env", lambda env: None)
        raw = json.dumps({"success": True, "image": str(image)})
        return str(image), json.loads(image_generation_tool._postprocess_image_generate_result(raw, task_id="t1"))

    return produce


def test_switched_on_the_model_reads_the_sandbox_path_first(generated):
    host, result = generated(switch=True)
    assert result["image"] == result["agent_visible_image"] == "/root/.hermes/cache/images/meta_cover.webp"
    assert result["host_image"] == host


@pytest.mark.parametrize("switch", [False, None])
def test_switched_off_the_result_is_upstreams(generated, switch):
    host, result = generated(switch=switch)
    assert result["image"] == result["host_image"] == host
    assert result["agent_visible_image"] != host


def test_without_a_sandbox_the_result_keeps_the_host_path(generated):
    host, result = generated(switch=True, sandboxed=False)
    assert result["image"] == host
    assert "agent_visible_image" not in result


def test_the_gateway_still_attaches_the_host_copy(generated):
    from gateway.run import _collect_auto_append_media_tags

    host, result = generated(switch=True)
    messages = [{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "image_generate", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": json.dumps(result)}]
    tags, _ = _collect_auto_append_media_tags(messages)
    assert tags == [f"MEDIA:{host}"]
