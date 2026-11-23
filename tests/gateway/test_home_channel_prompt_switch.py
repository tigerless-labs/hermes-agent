"""onboarding.home_channel_prompt: "off" keeps a deployment whose results never go to a home channel
from asking every new conversation to set one; anything else keeps the upstream prompt."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.fixture
def prompts(tmp_path, monkeypatch):
    async def asked(onboarding) -> int:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.delenv("SLACK_HOME_CHANNEL", raising=False)
        (tmp_path / "config.yaml").write_text(json.dumps({} if onboarding is None else {"onboarding": onboarding}))
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={Platform.SLACK: PlatformConfig(enabled=True, token="***")})
        runner.session_store = MagicMock(has_any_sessions=MagicMock(return_value=True))
        runner._deliver_platform_notice = AsyncMock()
        source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="channel", user_id="U1")
        await runner._hmwa_first_contact_notes(source, [], [])
        return runner._deliver_platform_notice.await_count
    return asked


@pytest.mark.asyncio
@pytest.mark.parametrize("off", ["off", " OFF ", False])
async def test_switched_off_a_new_conversation_is_not_asked_to_set_a_home_channel(prompts, off):
    assert await prompts({"home_channel_prompt": off}) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("onboarding", [None, {}, {"home_channel_prompt": "ask"}, {"home_channel_prompt": None},
                                        {"home_channel_prompt": ["off"]}, "off"])
async def test_without_the_switch_or_with_anything_else_the_upstream_prompt_stays(prompts, onboarding):
    assert await prompts(onboarding) == 1
