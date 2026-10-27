"""The file tools keep out of the sandbox's engine folder (``terminal.refuse_engine_dir_writes``).

Inside a container sandbox the engine's own folder sits at ``/root/.hermes``: the chat's caches there are read-only
copies and media delivery never fetches from it. A model that wrote its deliverable there either hit the read-only
mount or wrote a file that could never be attached. With the switch on, write_file and patch refuse such a target
before touching the sandbox and say where to write instead.
"""

import json

import pytest

import hermes_cli.config as hermes_config
import tools.file_tools as file_tools
import tools.file_tools_paths as file_tools_paths

SANDBOX_CWD = "/root/work"


class ReachedFileOps(Exception):
    pass


@pytest.fixture
def sandbox(monkeypatch):
    calls: list[str] = []

    def file_ops(task_id="default"):
        calls.append(task_id)
        raise ReachedFileOps("reached the sandbox's file operations")

    def configure(*, switch=True, backend="docker"):
        terminal = {} if switch is None else {"refuse_engine_dir_writes": switch}
        monkeypatch.setattr(hermes_config, "load_config_readonly", lambda: {"terminal": terminal})
        monkeypatch.setattr(file_tools_paths, "_terminal_env_type_for_task", lambda task_id="default": backend)
        monkeypatch.setattr(file_tools_paths, "_resolve_base_dir", lambda *args, **kwargs: SANDBOX_CWD)
        monkeypatch.setattr(file_tools, "_get_file_ops", file_ops)
        return calls

    return configure


def _refusal(result: str) -> str | None:
    error = json.loads(result).get("error") or ""
    return None if "reached the sandbox's file operations" in error else error


ENGINE_DIR_TARGETS = [
    "/root/.hermes/cache/outbox/article.md",
    "/root/.hermes/cache/generated/chats/slack-C0C1TU6QBS7/files/article.md",
    "/root/.hermes/cache/documents/chats/slack-C0C1TU6QBS7/remote_956bc21919a3/article.md",
    "~/.hermes/cache/blog/article.md",
    "../.hermes/cache/exports/report.csv",
    "/tmp/../root/.hermes/notes.md",
]

ATTACHABLE_TARGETS = [
    "/tmp/article/article.md",
    "/root/work/report.csv",
    "draft.md",
    "/root/.hermes-notes/article.md",
]


def test_off_by_default():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["terminal"]["refuse_engine_dir_writes"] is False


@pytest.mark.parametrize("target", ENGINE_DIR_TARGETS)
def test_write_file_into_engine_dir_is_refused_before_the_sandbox_is_touched(sandbox, target):
    calls = sandbox()
    refusal = _refusal(file_tools.write_file_tool(target, "# Article\n", task_id="s1"))
    assert refusal is not None
    assert calls == []
    assert "/tmp" in refusal
    assert target in refusal


@pytest.mark.parametrize("target", ENGINE_DIR_TARGETS)
def test_patch_into_engine_dir_is_refused_before_the_sandbox_is_touched(sandbox, target):
    calls = sandbox()
    refusal = _refusal(file_tools.patch_tool(path=target, old_string="old", new_string="", task_id="s1"))
    assert refusal is not None
    assert calls == []
    assert "/tmp" in refusal


def test_a_multi_file_patch_with_one_engine_dir_target_is_refused_whole(sandbox):
    calls = sandbox()
    patch = ("*** Begin Patch\n"
             "*** Update File: /tmp/article/article.md\n@@\n-old\n+new\n"
             "*** Update File: /root/.hermes/cache/outbox/article-preview.html\n@@\n-old\n+new\n"
             "*** End Patch")
    refusal = _refusal(file_tools.patch_tool(mode="patch", patch=patch, task_id="s1"))
    assert refusal is not None
    assert calls == []
    assert "/root/.hermes/cache/outbox/article-preview.html" in refusal


@pytest.mark.parametrize("target", ATTACHABLE_TARGETS)
def test_writes_outside_the_engine_dir_reach_the_sandbox(sandbox, target):
    calls = sandbox()
    assert _refusal(file_tools.write_file_tool(target, "# Article\n", task_id="s1")) is None
    assert calls


@pytest.mark.parametrize("switch", [False, None])
def test_switch_off_keeps_upstream_writes(sandbox, switch):
    calls = sandbox(switch=switch)
    assert _refusal(file_tools.write_file_tool(ENGINE_DIR_TARGETS[0], "# Article\n", task_id="s1")) is None
    assert calls


def test_host_backends_keep_upstream_writes(sandbox):
    calls = sandbox(backend="local")
    assert _refusal(file_tools.write_file_tool(ENGINE_DIR_TARGETS[0], "# Article\n", task_id="s1")) is None
    assert calls


def test_unreadable_config_keeps_upstream_writes(sandbox, monkeypatch):
    calls = sandbox()

    def broken():
        raise OSError("config unreadable")

    monkeypatch.setattr(hermes_config, "load_config_readonly", broken)
    assert _refusal(file_tools.write_file_tool(ENGINE_DIR_TARGETS[0], "# Article\n", task_id="s1")) is None
    assert calls
