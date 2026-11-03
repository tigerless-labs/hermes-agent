"""``ctx.interrupt_session(origin=...)`` stops the running turn of the session an
``inject_message(origin=...)`` turn runs in, in the plugin's OWN profile, the way ``/stop`` does.

Real ``GatewayRunner`` + real ``BasePlatformAdapter`` ingress; only the model call
(``_run_agent_inner``) is replaced by a turn that publishes its agent and runs until it is stopped
or released.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

PLUGIN = "desk"
NO_SWITCH = "inject-only"
STRING_SWITCH = "says-true"


class _Adapter(BasePlatformAdapter):
    def __init__(self, label: str):
        super().__init__(PlatformConfig(enabled=True, token=f"tok-{label}"), Platform.TELEGRAM)
        self.sent: list[tuple[str, str]] = []
        self._running = True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((str(chat_id), content))
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


class _Agent:
    def __init__(self, message: str, session_key: str):
        self.message = message
        self.session_key = session_key
        self.interrupts: list = []
        self.done = asyncio.Event()

    def hard_interrupt(self, message=None):
        self.interrupts.append(message)
        self.done.set()

    def release(self):
        self.done.set()


def _write_home(home) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"entries": {
        PLUGIN: {"allow_gateway_injection": True, "allow_gateway_interrupt": True},
        NO_SWITCH: {"allow_gateway_injection": True},
        STRING_SWITCH: {"allow_gateway_injection": True, "allow_gateway_interrupt": "true"},
    }}}), encoding="utf-8")
    (home / ".env").write_text("TELEGRAM_ALLOWED_USERS=u1\n", encoding="utf-8")


def _origin(chat_id: str, **extra) -> dict:
    return {"platform": "telegram", "chat_id": chat_id, "chat_type": "group", "thread_id": "7",
            "user_id": "u1", "user_name": "Requester", **extra}


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """Multiplexed gateway serving ``default``, ``alpha`` and ``beta`` (each its own bot); a route
    sends chat 999 on alpha's bot to ``beta``."""
    home = tmp_path / "hh"
    homes = {"default": home, "alpha": home / "profiles" / "alpha", "beta": home / "profiles" / "beta"}
    for profile_home in homes.values():
        _write_home(profile_home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)

    config = GatewayConfig(multiplex_profiles=True, sessions_dir=home / "sessions")
    config.profile_routes = parse_profile_routes([
        {"name": "beta-on-a", "platform": "telegram", "profile": "beta", "chat_id": "999", "bot_profile": "alpha"},
    ])
    runner = GatewayRunner(config)
    adapters = {name: _Adapter(name) for name in homes}
    runner.adapters = {Platform.TELEGRAM: adapters["default"]}
    runner._profile_adapters = {name: {Platform.TELEGRAM: adapters[name]} for name in ("alpha", "beta")}
    for name, adapter in adapters.items():
        adapter.gateway_runner = runner
        if name != "default":
            adapter.set_owner_profile(name)
            adapter.set_message_handler(runner._make_profile_message_handler(name))
        else:
            adapter.set_message_handler(runner._handle_message)

    agents: list[_Agent] = []

    async def _turn(message, context_prompt, history, source, session_id, session_key=None, **_kwargs):
        agent = _Agent(message, session_key)
        runner._running_agents[session_key] = agent  # what the real turn does once its agent is built
        agents.append(agent)
        await agent.done.wait()
        if agent.interrupts:
            return {"final_response": "", "interrupted": True, "interrupt_message": agent.interrupts[0],
                    "messages": [], "tools": [], "history_offset": len(history), "last_prompt_tokens": 0}
        return {"final_response": f"done: {message}", "messages": [], "tools": [],
                "history_offset": len(history), "last_prompt_tokens": 0}

    runner._run_agent_inner = _turn
    contexts = {}
    for profile in ("alpha", "beta"):
        for plugin in (PLUGIN, NO_SWITCH, STRING_SWITCH):
            manager = PluginManager(scope_key=str(homes[profile]))
            contexts[profile, plugin] = PluginContext(PluginManifest(name=plugin, key=plugin, source="user"), manager)
    managers = [ctx._manager for ctx in contexts.values()]
    with patch("hermes_cli.profiles.profiles_to_serve", return_value=list(homes.items())), \
            patch("hermes_cli.profiles.get_profile_dir", side_effect=lambda n: homes[n]), \
            patch("hermes_cli.profiles.profile_exists", side_effect=lambda n: n in homes), \
            patch("hermes_cli.plugins._known_plugin_managers", return_value=managers):
        yield SimpleNamespace(runner=runner, adapters=adapters, agents=agents, ctx=contexts)


async def _publish(runner):
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    runner._install_plugin_message_injector()


async def _until(predicate, rounds: int = 300):
    for _ in range(rounds):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


async def _start(gateway, profile: str, chat_id: str, text: str) -> _Agent:
    assert gateway.ctx[profile, PLUGIN].inject_message(text, origin=_origin(chat_id)) is True
    await _until(lambda: any(a.message.endswith(text) for a in gateway.agents))
    return next(a for a in gateway.agents if a.message.endswith(text))


async def _settle(gateway):
    await _until(lambda: not gateway.runner._background_tasks and all(
        not a._session_tasks and not a._active_sessions for a in gateway.adapters.values()))


@pytest.mark.asyncio
async def test_stops_the_running_turn_at_origin_and_clears_its_queue_like_stop(gateway):
    """From a worker thread (as an action handler hands it off): the turn at the origin is
    interrupted with /stop's reason, a follow-up queued behind it never runs, and a turn in another
    chat of the same profile keeps running."""
    runner, alpha, desk = gateway.runner, gateway.adapters["alpha"], gateway.ctx["alpha", PLUGIN]
    await _publish(runner)
    stopped = await _start(gateway, "alpha", "100", "Do the review")
    other = await _start(gateway, "alpha", "101", "Other chat's work")
    alpha._pending_messages[stopped.session_key] = MessageEvent(
        text="stale follow-up", message_type=MessageType.TEXT,
        source=alpha.build_source(chat_id="100", chat_type="group", thread_id="7", user_id="u1"))

    assert await asyncio.to_thread(desk.interrupt_session, origin=_origin("100")) is True
    await _until(lambda: stopped.session_key not in runner._running_agents)

    assert stopped.interrupts == [gateway_run._INTERRUPT_REASON_STOP]
    assert stopped.session_key not in alpha._pending_messages
    assert other.interrupts == [] and other.session_key in runner._running_agents
    other.release()
    await _settle(gateway)
    assert not any("stale follow-up" in a.message for a in gateway.agents)
    assert alpha.sent == [("101", f"done: {other.message}")]


@pytest.mark.asyncio
async def test_nothing_running_at_origin_is_false_and_changes_nothing(gateway):
    runner, alpha, desk = gateway.runner, gateway.adapters["alpha"], gateway.ctx["alpha", PLUGIN]
    await _publish(runner)
    assert desk.interrupt_session(origin=_origin("100")) is False  # no session there at all

    finished = await _start(gateway, "alpha", "101", "Quick one")
    finished.release()
    await _settle(gateway)
    parked = MessageEvent(text="parked", message_type=MessageType.TEXT,
                          source=alpha.build_source(chat_id="101", chat_type="group", thread_id="7", user_id="u1"))
    alpha._pending_messages[finished.session_key] = parked
    generation = runner._current_session_run_generation(finished.session_key)

    assert desk.interrupt_session(origin=_origin("101")) is False
    await asyncio.sleep(0.05)
    assert runner._current_session_run_generation(finished.session_key) == generation
    assert alpha._pending_messages[finished.session_key] is parked
    assert finished.interrupts == []


@pytest.mark.asyncio
async def test_another_profiles_turn_is_unreachable(gateway):
    """A plugin in ``alpha`` cannot stop ``beta``'s turns: not by naming the profile, not through a
    chat the gateway routes to ``beta``, not by the same chat id in its own profile."""
    runner, desk = gateway.runner, gateway.ctx["alpha", PLUGIN]
    await _publish(runner)
    in_beta = await _start(gateway, "beta", "200", "Beta's work")
    routed = await _start(gateway, "beta", "999", "Beta's routed chat")

    assert desk.interrupt_session(origin=_origin("200", profile="beta")) is False
    assert desk.interrupt_session(origin=_origin("999")) is False
    assert desk.interrupt_session(origin=_origin("200")) is False
    assert desk.interrupt_session(origin={"platform": "telegram"}) is False
    await asyncio.sleep(0.05)
    assert in_beta.interrupts == [] and routed.interrupts == []
    assert {in_beta.session_key, routed.session_key} <= set(runner._running_agents)

    assert gateway.ctx["beta", PLUGIN].interrupt_session(origin=_origin("200")) is True
    await _until(lambda: in_beta.session_key not in runner._running_agents)
    assert routed.interrupts == []
    routed.release()
    await _settle(gateway)


@pytest.mark.asyncio
async def test_without_the_switch_the_turn_keeps_running(gateway):
    runner = gateway.runner
    await _publish(runner)
    running = await _start(gateway, "alpha", "100", "Do the review")

    assert gateway.ctx["alpha", NO_SWITCH].interrupt_session(origin=_origin("100")) is False
    assert gateway.ctx["alpha", STRING_SWITCH].interrupt_session(origin=_origin("100")) is False
    await asyncio.sleep(0.05)
    assert running.interrupts == [] and running.session_key in runner._running_agents

    assert gateway.ctx["alpha", PLUGIN].interrupt_session(origin=_origin("100")) is True
    await _until(lambda: running.session_key not in runner._running_agents)
    await _settle(gateway)


@pytest.mark.asyncio
async def test_a_gateway_that_is_not_running_is_false(gateway):
    runner, desk = gateway.runner, gateway.ctx["alpha", PLUGIN]
    await _publish(runner)
    running = await _start(gateway, "alpha", "100", "Do the review")
    runner._running = False
    assert desk.interrupt_session(origin=_origin("100")) is False
    runner._running = True
    running.release()
    await _settle(gateway)
