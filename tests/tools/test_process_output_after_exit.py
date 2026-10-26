"""A sandbox job's last output survives its exit (``terminal.read_output_after_exit``).

The sandbox poller reads the log's new bytes, then checks whether the process is alive. A line written between
the two steps — typically the last line of a command that prints and exits, like ``sleep 240; date`` — was lost:
the poller saw the exit and finished without reading the log again.
"""

import time
from unittest.mock import patch

import pytest

import hermes_cli.config as hermes_config
from tools.process_registry import ProcessRegistry, ProcessSession

LAST_LINE = "Mon Oct 26 15:08:15 EDT 2026\n"


class RacingEnv:
    """The log is empty at the poll; the process prints its last line and exits before the liveness check."""

    def __init__(self, after_exit: str):
        self.commands: list[str] = []
        self._responses = iter([
            {"output": "0 0\n"},
            {"output": "1\n"},
            {"output": "0\n"},
            {"output": after_exit},
        ])

    def execute(self, command, **kwargs):
        self.commands.append(command)
        return next(self._responses, {"output": ""})


def _poll_to_exit(monkeypatch, terminal_section, env) -> ProcessSession:
    monkeypatch.setattr(hermes_config, "load_config_readonly", lambda: {"terminal": terminal_section})
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_race", command="sleep 240; date", task_id="t1", started_at=time.time())
    with patch("tools.process_registry.time.sleep", return_value=None), \
         patch.object(registry, "_move_to_finished"):
        registry._env_poller_loop(session, env, "/tmp/hermes_bg.log", "/tmp/hermes_bg.pid", "/tmp/hermes_bg.exit")
    return session


def test_off_by_default():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["terminal"]["read_output_after_exit"] is False


@pytest.mark.parametrize("section", [{}, {"read_output_after_exit": False}])
def test_off_upstream_finishes_without_reading_the_log_again(monkeypatch, section):
    env = RacingEnv(f"{len(LAST_LINE)} 0\n{LAST_LINE}")
    session = _poll_to_exit(monkeypatch, section, env)
    assert session.exited and session.exit_code == 0
    assert session.output_buffer == "" and len(env.commands) == 3


def test_on_the_line_written_just_before_exit_is_kept(monkeypatch):
    env = RacingEnv(f"{len(LAST_LINE)} 0\n{LAST_LINE}")
    session = _poll_to_exit(monkeypatch, {"read_output_after_exit": True}, env)
    assert session.exited and session.exit_code == 0
    assert session.output_buffer == LAST_LINE
    assert "hermes_bg.log" in env.commands[3]


def test_on_nothing_new_after_exit_leaves_the_output_as_it_was(monkeypatch):
    session = _poll_to_exit(monkeypatch, {"read_output_after_exit": True}, RacingEnv("0 0\n"))
    assert session.exited and session.exit_code == 0 and session.output_buffer == ""


@pytest.mark.parametrize("after_exit", ["", "not a header\nnoise", "12 7\nshifted text"])
def test_redteam_a_final_read_that_does_not_line_up_adds_nothing(monkeypatch, after_exit):
    session = _poll_to_exit(monkeypatch, {"read_output_after_exit": True}, RacingEnv(after_exit))
    assert session.exited and session.exit_code == 0 and session.output_buffer == ""


def test_redteam_a_final_read_that_fails_still_finishes_the_job(monkeypatch):
    class FailingEnv(RacingEnv):
        def execute(self, command, **kwargs):
            if len(self.commands) == 3:
                self.commands.append(command)
                raise RuntimeError("sandbox gone")
            return super().execute(command, **kwargs)

    session = _poll_to_exit(monkeypatch, {"read_output_after_exit": True}, FailingEnv(""))
    assert session.exited and session.exit_code == 0 and session.output_buffer == ""


def test_redteam_an_unreadable_config_keeps_upstream_s_behaviour(monkeypatch):
    def unreadable():
        raise OSError("config.yaml unreadable")

    env = RacingEnv(f"{len(LAST_LINE)} 0\n{LAST_LINE}")
    monkeypatch.setattr(hermes_config, "load_config_readonly", unreadable)
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_race", command="sleep 240; date", task_id="t1", started_at=time.time())
    with patch("tools.process_registry.time.sleep", return_value=None), \
         patch.object(registry, "_move_to_finished"):
        registry._env_poller_loop(session, env, "/tmp/hermes_bg.log", "/tmp/hermes_bg.pid", "/tmp/hermes_bg.exit")
    assert session.exited and session.output_buffer == "" and len(env.commands) == 3
