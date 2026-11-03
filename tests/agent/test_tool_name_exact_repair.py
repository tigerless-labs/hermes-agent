"""``agent.tool_name_fuzzy_repair: false``: an unknown tool name is never renamed to a different tool.

Upstream repairs a tool name the model wrote in two steps: deterministic spellings of the same name (case,
separators, CamelCase, a ``_tool`` suffix), then a fuzzy match onto the nearest registered name. With a large
on-demand catalogue whose write tools carry API-style names, the fuzzy step turned a guessed read call into a
write call: ``gdrive__drive_files_list`` ran as ``gdrive__drive_files_delete`` and
``gmail__gmail_users_messages_modify`` as ``gmail__gmail_users_messages_send``. Switched off, only the deterministic
spellings repair — plus ``<platform>.<tool>`` written for ``<platform>__<tool>`` — and anything else stays unknown,
so the model is told the tool does not exist. Unset keeps upstream.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

HUB = {
    "tool_search",
    "gdrive__search_files", "gdrive__drive_files_delete", "gdrive__drive_files_update",
    "gmail__search_threads", "gmail__label_message", "gmail__gmail_users_messages_send",
    "execute_code", "read_file",
}


def _configure(tmp_path, monkeypatch, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.fixture
def repair():
    from run_agent import AIAgent
    return AIAgent._repair_tool_call.__get__(SimpleNamespace(valid_tool_names=HUB), AIAgent)


@pytest.fixture
def exact(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, "agent:\n  tool_name_fuzzy_repair: false\n")


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, "agent:\n  max_turns: null\n")


GUESSES_NEAR_WRITE_TOOLS = ["gdrive__drive_files_list", "gdrive__drive_files_get", "gmail__gmail_users_messages_modify",
                            "gmail__gmail_users_labels_create", "gmail__gmail_users_messages_get"]


@pytest.mark.parametrize("guess", GUESSES_NEAR_WRITE_TOOLS)
def test_redteam_a_guess_near_a_write_tool_stays_unknown(exact, repair, guess):
    assert repair(guess) is None


@pytest.mark.parametrize("written, meant", [
    ("GDRIVE__SEARCH_FILES", "gdrive__search_files"),
    ("Execute-Code", "execute_code"),
    ("ExecuteCode", "execute_code"),
    ("read_file_tool", "read_file"),
    ("gdrive.search_files", "gdrive__search_files"),
    ("gmail.label_message", "gmail__label_message"),
])
def test_spellings_of_the_same_tool_still_repair(exact, repair, written, meant):
    assert repair(written) == meant


def test_a_namespace_dot_never_reaches_another_tool(exact, repair):
    assert repair("gdrive.drive_files_list") is None


def test_upstream_still_fuzzy_matches_when_unset(upstream, repair):
    assert repair("gdrive__drive_files_list") == "gdrive__drive_files_delete"


@pytest.mark.parametrize("value", ["true", "null", "0", "\"false\""])
def test_anything_but_false_keeps_upstream(tmp_path, monkeypatch, repair, value):
    _configure(tmp_path, monkeypatch, f"agent:\n  tool_name_fuzzy_repair: {value}\n")
    assert repair("gdrive__drive_files_list") == "gdrive__drive_files_delete"
