"""``SLACK_EVENTS_SOCKET``: Slack payloads come over a local socket instead of Socket Mode.

Something beside the gateway (a relay pulling a queue the app's HTTP endpoint fills) hands each payload over a unix
socket as one JSON line, ``{"id", "kind", "body"}``, ``body`` exactly what Socket Mode would carry. Each is dispatched
through the same Bolt app in socket-mode form, so the listeners run unchanged; content a listener puts in its ack (a
slash command's ephemeral notice) goes to the payload's Slack ``response_url`` afterwards; ``{"ack": id}`` goes back
once the payload is dispatched. With the switch on, connect needs no app token and starts neither Socket Mode nor its
watchdog. Unset is the upstream behavior.
"""

import asyncio
import json
import os
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

import plugins.platforms.slack.adapter as slack_adapter
from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.events_socket import SlackEventsSocket, is_slack_reply_url

WAIT_SECONDS = 5.0
TEAM, CHANNEL, USER, BOT = "T0TEAM", "C0CHAN", "U0ANN", "U0BOT"


class FakeSlack(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), FakeSlackHandler)
        self.replies: list[tuple[str, dict]] = []
        self.lock = threading.Lock()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class FakeSlackHandler(BaseHTTPRequestHandler):
    server: FakeSlack

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path.partition("?")[0] == "/api/auth.test":
            answer = {"ok": True, "team_id": TEAM, "team": "Team", "user_id": BOT, "user": "lara", "bot_id": "B0BOT"}
        else:
            with self.server.lock:
                self.server.replies.append((self.path, json.loads(raw)))
            answer = {"ok": True}
        body = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Relay:
    """The sender's side of the socket: hands lines to whoever connects and keeps the answers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.answers: list[dict] = []
        self.connections = 0
        self.connected = asyncio.Event()
        self.closed = asyncio.Event()
        self.writer: asyncio.StreamWriter | None = None
        self.server: asyncio.AbstractServer | None = None
        self.probe = lambda: None
        self.seen_at_answer: dict = {}

    async def start(self) -> None:
        self.server = await asyncio.start_unix_server(self._serve, path=str(self.path))

    async def _serve(self, reader, writer) -> None:
        self.connections += 1
        self.writer = writer
        self.closed.clear()
        self.connected.set()
        try:
            while line := await reader.readline():
                answer = json.loads(line)
                self.seen_at_answer[answer.get("ack")] = self.probe()
                self.answers.append(answer)
        finally:
            self.closed.set()

    async def hand(self, raw: bytes) -> None:
        await asyncio.wait_for(self.connected.wait(), WAIT_SECONDS)
        self.writer.write(raw)
        await self.writer.drain()

    async def hand_over(self, message_id, kind, body) -> None:
        await self.hand(json.dumps({"id": message_id, "kind": kind, "body": body}).encode() + b"\n")

    async def answered(self, message_id) -> bool:
        async def wait() -> None:
            while {"ack": message_id} not in self.answers:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait(), WAIT_SECONDS)
        return True

    async def drop_connection(self) -> None:
        self.connected.clear()
        self.writer.close()

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()


def message_event(text: str, event_id: str = "Ev1") -> dict:
    return {"token": "t", "team_id": TEAM, "api_app_id": "A0APP", "type": "event_callback", "event_id": event_id,
            "event_time": 1, "event": {"type": "message", "channel": CHANNEL, "user": USER, "text": text,
                                       "ts": "1.0", "channel_type": "channel"}}


def slash_command(response_url: str, text: str = "help") -> dict:
    return {"token": "t", "team_id": TEAM, "team_domain": "team", "channel_id": CHANNEL, "channel_name": "general",
            "user_id": USER, "user_name": "ann", "command": "/hermes", "text": text, "api_app_id": "A0APP",
            "response_url": response_url, "trigger_id": "1.2.3"}


def button_click(response_url: str) -> dict:
    return {"type": "block_actions", "team": {"id": TEAM}, "user": {"id": USER}, "api_app_id": "A0APP",
            "channel": {"id": CHANNEL}, "container": {"type": "message", "message_ts": "1.0"},
            "response_url": response_url, "trigger_id": "1.2.3",
            "actions": [{"action_id": "hermes_feedback", "block_id": "b", "type": "button", "value": "up",
                         "action_ts": "1.0"}]}


@pytest.fixture
def fake_slack():
    server = FakeSlack()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def socket_path():
    folder = Path(tempfile.mkdtemp(prefix="ses-"))
    yield folder / "events.sock"
    for leftover in folder.iterdir():
        leftover.unlink()
    folder.rmdir()


@pytest.fixture
def bolt(fake_slack):
    client = AsyncWebClient(token="xoxb-test", base_url=f"{fake_slack.base}/api/")
    app = AsyncApp(token="xoxb-test", client=client)
    seen = {"events": [], "commands": [], "actions": []}

    @app.event("message")
    async def on_message(event):
        seen["events"].append(event)

    @app.command("/hermes")
    async def on_command(ack, command):
        await ack(response_type="ephemeral", text=f"Running /{command['text']}")
        seen["commands"].append(command)

    @app.action("hermes_feedback")
    async def on_click(ack, body):
        await ack()
        seen["actions"].append(body)

    return app, seen


@pytest_asyncio.fixture
async def wired(bolt, fake_slack, socket_path):
    app, seen = bolt
    relay = Relay(socket_path)
    await relay.start()
    events = SlackEventsSocket(app, str(socket_path))
    events.reconnect_initial_s = 0.05
    events.reconnect_max_s = 0.2
    events.reply_url = re.compile(re.escape(fake_slack.base) + r"/\S*")
    events.start()
    yield relay, events, seen
    await events.stop()
    await relay.stop()


async def until(check) -> None:
    async def wait() -> None:
        while not check():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), WAIT_SECONDS)


@pytest.mark.asyncio
async def test_an_event_runs_its_listener_and_is_answered_with_its_id(wired) -> None:
    relay, _, seen = wired
    await relay.hand_over("m-1", "events_api", message_event("hello Lara"))
    assert await relay.answered("m-1")
    await until(lambda: seen["events"])
    assert seen["events"][0]["text"] == "hello Lara"


@pytest.mark.asyncio
async def test_a_slash_command_s_ephemeral_ack_goes_to_its_reply_address_before_the_answer(wired, fake_slack) -> None:
    relay, _, seen = wired
    relay.probe = lambda: list(fake_slack.replies)
    await relay.hand_over("m-2", "slash_commands", slash_command(f"{fake_slack.base}/commands/T0TEAM/1/abc"))
    assert await relay.answered("m-2")
    await until(lambda: seen["commands"])
    reply = ("/commands/T0TEAM/1/abc", {"response_type": "ephemeral", "text": "Running /help"})
    assert fake_slack.replies == [reply] and relay.seen_at_answer["m-2"] == [reply]


@pytest.mark.asyncio
async def test_a_click_with_a_bare_ack_runs_its_listener_and_replies_nowhere(wired, fake_slack) -> None:
    relay, _, seen = wired
    await relay.hand_over("m-3", "interactive", button_click(f"{fake_slack.base}/actions/T0TEAM/1/abc"))
    assert await relay.answered("m-3")
    await until(lambda: seen["actions"])
    assert fake_slack.replies == []


@pytest.mark.asyncio
async def test_lines_without_an_id_are_skipped_and_the_reader_carries_on(wired) -> None:
    relay, _, seen = wired
    await relay.hand(b"not json\n[1, 2]\n{\"kind\": \"events_api\", \"body\": {}}\n{\"id\": 7, \"kind\": \"events_api\"}\n")
    await relay.hand_over("m-4", "events_api", message_event("after the garbage"))
    assert await relay.answered("m-4")
    assert relay.answers == [{"ack": "m-4"}]
    await until(lambda: seen["events"])
    assert [event["text"] for event in seen["events"]] == ["after the garbage"]


@pytest.mark.parametrize("kind, body", [("workflow_steps", message_event("x")), ("events_api", "just text"),
                                         ("events_api", ["a", "list"])])
@pytest.mark.asyncio
async def test_a_payload_that_cannot_be_dispatched_is_answered_without_dispatch(wired, kind, body) -> None:
    relay, _, seen = wired
    await relay.hand_over("m-5", kind, body)
    assert await relay.answered("m-5")
    await asyncio.sleep(0.1)
    assert seen == {"events": [], "commands": [], "actions": []}


@pytest.mark.parametrize("url", ["https://evil.example/hook", "http://hooks.slack.com/commands/T/1/a",
                                 "https://hooks.slack.com.evil.example/commands/T/1/a", "https://evilslack.com/x",
                                 "https://user@hooks.slack.com/commands/T/1/a", "", None])
def test_redteam_only_slack_s_own_https_addresses_count_as_reply_addresses(url) -> None:
    assert not is_slack_reply_url(url)


def test_slack_s_reply_addresses_count() -> None:
    assert is_slack_reply_url("https://hooks.slack.com/commands/T0TEAM/1/abc")


@pytest.mark.asyncio
async def test_redteam_an_ack_s_content_never_goes_to_an_address_off_slack(bolt, fake_slack, socket_path) -> None:
    app, seen = bolt
    relay = Relay(socket_path)
    await relay.start()
    events = SlackEventsSocket(app, str(socket_path))
    events.start()
    try:
        await relay.hand_over("m-6", "slash_commands", slash_command(f"{fake_slack.base}/commands/T0TEAM/1/abc"))
        assert await relay.answered("m-6")
        await until(lambda: seen["commands"])
        assert fake_slack.replies == []
    finally:
        await events.stop()
        await relay.stop()


@pytest.mark.asyncio
async def test_a_dispatch_that_fails_is_answered_and_the_next_payload_still_runs(wired, bolt) -> None:
    relay, _, seen = wired
    app, _ = bolt
    real, calls = app.async_dispatch, []

    async def failing_once(request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("listener blew up")
        return await real(request)

    app.async_dispatch = failing_once
    await relay.hand_over("m-7", "events_api", message_event("first", "Ev7"))
    await relay.hand_over("m-8", "events_api", message_event("second", "Ev8"))
    assert await relay.answered("m-7") and await relay.answered("m-8")
    await until(lambda: seen["events"])
    assert [event["text"] for event in seen["events"]] == ["second"]


@pytest.mark.asyncio
async def test_a_dropped_connection_is_made_again_and_reading_goes_on(wired) -> None:
    relay, _, seen = wired
    await relay.hand_over("m-9", "events_api", message_event("before", "Ev9"))
    assert await relay.answered("m-9")
    await relay.drop_connection()
    await relay.hand_over("m-10", "events_api", message_event("after", "Ev10"))
    assert await relay.answered("m-10")
    assert relay.connections == 2


@pytest.mark.asyncio
async def test_a_socket_that_is_not_there_yet_is_waited_for(bolt, socket_path) -> None:
    app, seen = bolt
    events = SlackEventsSocket(app, str(socket_path))
    events.reconnect_initial_s = 0.05
    events.reconnect_max_s = 0.1
    events.start()
    relay = Relay(socket_path)
    try:
        await asyncio.sleep(0.3)
        await relay.start()
        await relay.hand_over("m-11", "events_api", message_event("late", "Ev11"))
        assert await relay.answered("m-11")
    finally:
        await events.stop()
        await relay.stop()


@pytest.mark.asyncio
async def test_stopping_closes_the_connection_and_tries_no_more(wired) -> None:
    relay, events, _ = wired
    await asyncio.wait_for(relay.connected.wait(), WAIT_SECONDS)
    await asyncio.wait_for(events.stop(), WAIT_SECONDS)
    await asyncio.wait_for(relay.closed.wait(), WAIT_SECONDS)
    await asyncio.sleep(0.3)
    assert relay.connections == 1


def _adapter_connected(env: dict) -> tuple[SlackAdapter, MagicMock, MagicMock, dict]:
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    app = MagicMock()
    web_client = AsyncMock()
    web_client.auth_test = AsyncMock(return_value={"user_id": BOT, "user": "lara", "team_id": TEAM, "team": "Team"})
    socket_mode = MagicMock()
    socket_mode.start_async = AsyncMock(return_value=None)
    socket_mode.close_async = AsyncMock(return_value=None)
    reader = MagicMock()
    reader.stop = AsyncMock(return_value=None)
    locks = {}

    def acquire(scope, identity, metadata=None):
        locks[scope] = identity
        return True, None

    with (
        patch.object(slack_adapter, "AsyncApp", return_value=app),
        patch.object(slack_adapter, "AsyncWebClient", return_value=web_client),
        patch.object(slack_adapter, "AsyncSocketModeHandler", return_value=socket_mode) as handler_class,
        patch.object(slack_adapter, "SlackEventsSocket", return_value=reader) as reader_class,
        patch.dict(os.environ, env, clear=True),
        patch("gateway.status.acquire_scoped_lock", side_effect=acquire),
        patch("gateway.status.release_scoped_lock", return_value=None),
    ):
        connected = asyncio.run(_connect_and_disconnect(adapter))
    return adapter, handler_class, reader_class, {"connected": connected, "locks": locks, "reader": reader}


async def _connect_and_disconnect(adapter: SlackAdapter) -> dict:
    connected = await adapter.connect()
    state = {"connected": connected, "watchdog": adapter._socket_watchdog_task}
    await adapter.disconnect()
    return state


def test_with_the_switch_connect_needs_no_app_token_and_reads_the_socket_instead_of_socket_mode() -> None:
    _, handler_class, reader_class, found = _adapter_connected(
        {"SLACK_BOT_TOKEN": "xoxb-fake", "SLACK_EVENTS_SOCKET": "/run/relay/events.sock"})
    assert found["connected"]["connected"] is True
    handler_class.assert_not_called()
    assert found["connected"]["watchdog"] is None
    assert reader_class.call_args.args[1] == "/run/relay/events.sock"
    found["reader"].start.assert_called_once()
    found["reader"].stop.assert_awaited()
    assert found["locks"] == {"slack-events-socket": "/run/relay/events.sock"}


def test_without_the_switch_socket_mode_runs_as_upstream_and_no_socket_is_read() -> None:
    _, handler_class, reader_class, found = _adapter_connected(
        {"SLACK_BOT_TOKEN": "xoxb-fake", "SLACK_APP_TOKEN": "xapp-fake"})
    assert found["connected"]["connected"] is True
    handler_class.assert_called_once()
    reader_class.assert_not_called()
    assert found["locks"] == {"slack-app-token": "xapp-fake"}


def test_without_the_switch_a_missing_app_token_still_fails_as_upstream() -> None:
    adapter, handler_class, reader_class, found = _adapter_connected({"SLACK_BOT_TOKEN": "xoxb-fake"})
    assert found["connected"]["connected"] is False
    handler_class.assert_not_called()
    reader_class.assert_not_called()
    assert adapter.fatal_error_code == "missing_slack_app_token"
