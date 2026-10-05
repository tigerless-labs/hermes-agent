"""display.agent_name: what the gateway's own chat notices call the agent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.run import GatewayRunner, _gateway_provider_error_reply
from gateway.session import SessionSource, build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

UPSTREAM = {
    "shutdown": (
        "⚠️ Hermes is shutting down — your current task will be interrupted. "
        "When it is back online, send any message and I'll try to pick up where we left off."),
    "restart": (
        "⚠️ Hermes is restarting — your current task will be interrupted. "
        "Send any message after the restart and I'll try to resume where you left off."),
    "cron_cut_short": (
        "⚠️ Scheduled job 'daily-digest' was cut short because Hermes is shutting down; "
        "no result this run. It will run again on schedule, or run it now with "
        "`hermes cron run daily-digest` once Hermes is back."),
    "back_online": "♻️ Gateway online — Hermes is back and ready.",
    "model_unreachable": (
        "⚠️ Hermes could not reach the AI model service (no further detail from the "
        "SDK). Use /retry to try again; if it persists, run `hermes doctor` on the host."),
    "no_home_channel": (
        "📬 No home channel is set for Slack. A home channel is where Hermes delivers cron job "
        "results and cross-platform messages.\n\nType /hermes sethome to make this chat your home "
        "channel, or ignore to skip."),
}


@pytest.fixture
def configure(tmp_path, monkeypatch):
    def write(display):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.delenv("SLACK_HOME_CHANNEL", raising=False)
        (tmp_path / "config.yaml").write_text(json.dumps({"display": display}))
    return write


async def _session_notice(restarting: bool) -> str:
    runner, adapter = make_restart_runner()
    source = make_restart_source()
    session_key = build_session_key(source)
    runner._running_agents = {session_key: MagicMock()}
    runner._cache_session_source(session_key, source)
    runner._restart_requested = restarting
    await runner._notify_active_sessions_of_shutdown()
    (_chat, message, _meta), = adapter.sent_calls
    return message


async def _cron_notice() -> str:
    runner, adapter = make_restart_runner()
    runner._thread_metadata_for_target = GatewayRunner._thread_metadata_for_target.__get__(runner, GatewayRunner)
    job = {"id": "be62d36a9914", "name": "daily-digest", "deliver": "telegram:123456"}
    with patch("cron.jobs.get_job", return_value=job), \
         patch("cron.scheduler._resolve_delivery_targets",
               return_value=[{"platform": "telegram", "chat_id": "123456", "thread_id": None}]):
        await runner._notify_interrupted_cron_jobs([job["id"]])
    message, = adapter.sent
    return message


async def _back_online_notice() -> str:
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id="home-chat", name="Home")
    await runner._send_home_channel_startup_notifications()
    message, = adapter.sent
    return message


async def _no_home_channel_notice() -> str:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.SLACK: PlatformConfig(enabled=True, token="***")})
    runner.session_store = MagicMock(has_any_sessions=MagicMock(return_value=True))
    runner._deliver_platform_notice = AsyncMock()
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="channel", user_id="U1")
    await runner._hmwa_first_contact_notes(source, [], [])
    return runner._deliver_platform_notice.await_args.args[1]


async def _notices() -> dict[str, str]:
    return {
        "shutdown": await _session_notice(restarting=False),
        "restart": await _session_notice(restarting=True),
        "cron_cut_short": await _cron_notice(),
        "back_online": await _back_online_notice(),
        "model_unreachable": _gateway_provider_error_reply("openai.APIConnectionError: Connection error."),
        "no_home_channel": await _no_home_channel_notice(),
    }


@pytest.mark.asyncio
async def test_without_a_name_every_notice_keeps_the_upstream_wording(configure):
    configure({})
    assert await _notices() == UPSTREAM


@pytest.mark.asyncio
async def test_a_name_replaces_only_the_agent_in_every_notice_and_keeps_the_commands(configure):
    configure({"agent_name": "Lara"})
    notices = await _notices()
    assert notices == {kind: text.replace("Hermes", "Lara") for kind, text in UPSTREAM.items()}
    assert all("Hermes" not in text for text in notices.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["", "   ", None])
async def test_redteam_a_blank_name_falls_back_to_the_upstream_wording(configure, blank):
    configure({"agent_name": blank})
    assert await _notices() == UPSTREAM


@pytest.mark.asyncio
async def test_redteam_a_name_with_format_fields_is_shown_as_written(configure):
    name = "{agent} %s {0}"
    configure({"agent_name": f"  {name} "})
    assert await _notices() == {kind: text.replace("Hermes", name) for kind, text in UPSTREAM.items()}
