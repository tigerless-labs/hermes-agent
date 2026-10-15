"""A large result spilled before the session's container starts (``terminal.docker_cache_scope: chat``).

The docker backend creates a session's container on its first terminal or code call and mounts the chat's
caches then. A spill written earlier is referenced where that container will read it, so the model's first
read lands. A context outside any chat has no exchange area, so nothing of it lands on the host: with a
sandbox the result goes into the sandbox only, without one the model gets it truncated inline. Shared caches
and other backends keep the upstream host path.
"""

from __future__ import annotations

import posixpath
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_constants import borrowed_chat_cache_scope
from tools.credential_files import get_cache_directory_mounts
from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, extract_persisted_path, maybe_persist_tool_result

THRESHOLD = 1_000
CONTENT = "row," * THRESHOLD


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir()
    import tools.tool_result_storage as storage
    monkeypatch.setattr(storage, "_spillover_pruned_homes", set())
    return tmp_path / "hermes"


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERMINAL_ENV", "docker")


@pytest.fixture
def scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERMINAL_DOCKER_CACHE_SCOPE", "chat")


class _Chat:
    def __init__(self, chat_id: str) -> None:
        self.chat_id = chat_id

    def __enter__(self):
        self.tokens = set_session_vars(platform="slack", chat_id=self.chat_id)
        return self

    def __exit__(self, *_):
        clear_session_vars(self.tokens)


def _spill(tool_use_id: str) -> str:
    result = maybe_persist_tool_result(content=CONTENT, tool_name="airtable__list_records_for_table",
                                       tool_use_id=tool_use_id, env=None, threshold=THRESHOLD)
    path = extract_persisted_path(result)
    assert path, result
    return path


def _read_in_container(container_path: str) -> str | None:
    """What a container created in the current context reads at *container_path*, through its mounts."""
    for mount in get_cache_directory_mounts():
        root = mount["container_path"]
        if container_path == root or container_path.startswith(root + "/"):
            host = Path(mount["host_path"]) / posixpath.relpath(container_path, root)
            return host.read_text(encoding="utf-8") if host.is_file() else None
    return None


@pytest.mark.usefixtures("docker", "scoped")
def test_spill_before_the_container_names_the_path_the_coming_container_reads(home: Path) -> None:
    with _Chat("C1"):
        path = _spill("call_before")
        assert not Path(path).is_relative_to(home)
        assert _read_in_container(path) == CONTENT


@pytest.mark.usefixtures("docker", "scoped")
def test_background_run_borrowing_a_chat_names_that_chats_container_path(home: Path) -> None:
    with borrowed_chat_cache_scope("slack", "C1"):
        path = _spill("call_borrowed")
        assert not Path(path).is_relative_to(home)
        assert _read_in_container(path) == CONTENT


@pytest.mark.usefixtures("docker", "scoped")
def test_another_chats_container_cannot_read_the_spill(home: Path) -> None:
    with _Chat("C1"):
        path = _spill("call_private")
    with _Chat("C2"):
        assert _read_in_container(path) is None


def _host_files(home: Path) -> list[Path]:
    return [path for path in home.rglob("*") if path.is_file()]


@pytest.mark.usefixtures("docker", "scoped")
def test_redteam_outside_any_chat_without_a_sandbox_nothing_lands_on_the_host(home: Path) -> None:
    result = maybe_persist_tool_result(content=CONTENT, tool_name="airtable__list_records_for_table",
                                       tool_use_id="call_no_chat", env=None, threshold=THRESHOLD)
    assert PERSISTED_OUTPUT_TAG not in result and CONTENT.startswith(result.split("\n\n")[0])
    assert _host_files(home) == []


@pytest.mark.usefixtures("docker", "scoped")
def test_redteam_outside_any_chat_with_a_sandbox_the_result_goes_into_the_sandbox_only(home: Path) -> None:
    sandbox = MagicMock()
    sandbox.get_temp_dir.return_value = "/tmp"
    sandbox.execute.side_effect = [{"output": "", "returncode": 0},
                                   {"output": f"{len(CONTENT.encode())}\n", "returncode": 0}]
    result = maybe_persist_tool_result(content=CONTENT, tool_name="airtable__list_records_for_table",
                                       tool_use_id="call_no_chat_sandbox", env=sandbox, threshold=THRESHOLD)
    path = extract_persisted_path(result)
    assert path and not Path(path).is_relative_to(home)
    assert any(path in str(call.args[0]) for call in sandbox.execute.call_args_list)
    assert _host_files(home) == []


@pytest.mark.usefixtures("docker")
def test_shared_caches_outside_any_chat_keep_the_upstream_host_path(home: Path) -> None:
    path = _spill("call_shared_no_chat")
    assert Path(path).is_relative_to(home)
    assert Path(path).read_text(encoding="utf-8") == CONTENT


@pytest.mark.usefixtures("docker")
def test_shared_caches_keep_the_upstream_host_path(home: Path) -> None:
    with _Chat("C1"):
        path = _spill("call_shared")
    assert Path(path).is_relative_to(home)
    assert Path(path).read_text(encoding="utf-8") == CONTENT


@pytest.mark.usefixtures("scoped")
def test_local_backend_keeps_the_host_path(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERMINAL_ENV", "local")
    with _Chat("C1"):
        path = _spill("call_local")
    assert Path(path).is_relative_to(home)
    assert Path(path).read_text(encoding="utf-8") == CONTENT
