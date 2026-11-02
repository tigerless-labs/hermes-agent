"""``SLACK_EVENTS_SOCKET`` (fork, Lara): Slack payloads from a local socket instead of Socket Mode.

A deployment whose Slack app delivers over HTTP runs something beside the gateway that receives (or pulls) the
payloads and hands each over a unix socket as one JSON line, ``{"id", "kind", "body"}``, ``body`` exactly what Socket
Mode would carry. Each is dispatched through the same Bolt app in socket-mode form, so every listener runs unchanged.
Content a listener puts in its ack (a slash command's ephemeral "Running /x") cannot ride a response the endpoint
already sent, so it goes to the payload's Slack ``response_url`` afterwards. ``{"ack": id}`` goes back once the payload
is dispatched; an unanswered one is the sender's to hand over again. Unset keeps Socket Mode.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

KINDS = frozenset({"events_api", "interactive", "slash_commands"})
LINE_LIMIT_BYTES = 16 * 1024 * 1024
SLACK_REPLY_URL = re.compile(r"https://(?:[a-z0-9-]+\.)*slack\.com/\S*")


def is_slack_reply_url(url: Any) -> bool:
    return isinstance(url, str) and SLACK_REPLY_URL.fullmatch(url) is not None


def _handed(line: bytes) -> Optional[tuple[str, Optional[dict]]]:
    """``(id, body)`` of one line; body None when the payload cannot be dispatched; None when there is no id."""
    try:
        payload = json.loads(line)
    except ValueError:
        return None
    message_id = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(message_id, str) or not message_id:
        return None
    body = payload.get("body")
    return message_id, body if payload.get("kind") in KINDS and isinstance(body, dict) else None


def _ack_content(response: Any) -> Optional[dict]:
    """What the listener's ack carried, in the shape Socket Mode would have sent it; None for a bare ack."""
    if getattr(response, "status", None) != 200 or not getattr(response, "body", None):
        return None
    content_type = (response.headers.get("content-type") or [""])[0]
    if content_type.startswith("application/json"):
        with contextlib.suppress(ValueError):
            content = json.loads(response.body)
            return content if isinstance(content, dict) and content else None
        return None
    return {"text": response.body}


class SlackEventsSocket:
    """Reads one socket for as long as it runs, reconnecting with a growing pause."""

    reconnect_initial_s = 0.5
    reconnect_max_s = 30.0
    reply_timeout_s = 10.0
    reply_url = SLACK_REPLY_URL

    def __init__(self, app: Any, path: str, *, trust_env: bool = True) -> None:
        self._app = app
        self._path = path
        self._trust_env = trust_env
        self._task: Optional[asyncio.Task] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._writer is not None:
            self._writer.close()

    async def _run(self) -> None:
        pause = self.reconnect_initial_s
        while True:
            try:
                reader, self._writer = await asyncio.open_unix_connection(self._path, limit=LINE_LIMIT_BYTES)
            except OSError as exc:
                logger.debug("[Slack] Events socket %s not reachable (%s); trying again", self._path, exc)
                await asyncio.sleep(pause)
                pause = min(pause * 2, self.reconnect_max_s)
                continue
            pause = self.reconnect_initial_s
            logger.info("[Slack] Reading events from %s", self._path)
            try:
                await self._read(reader, self._writer)
            except (OSError, ValueError, asyncio.IncompleteReadError) as exc:
                logger.warning("[Slack] Events socket %s dropped: %s", self._path, exc)
            finally:
                self._writer.close()
            logger.warning("[Slack] Events socket %s closed; reconnecting", self._path)
            await asyncio.sleep(pause)

    async def _read(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while line := await reader.readline():
            handed = _handed(line)
            if handed is None:
                logger.warning("[Slack] Skipped a line without an id from the events socket")
                continue
            message_id, body = handed
            if body is None:
                logger.warning("[Slack] Answered without dispatch a payload that is not one: %s", message_id)
            else:
                await self._dispatch(body)
            writer.write(json.dumps({"ack": message_id}).encode() + b"\n")
            await writer.drain()

    async def _dispatch(self, body: dict) -> None:
        from slack_bolt.request.async_request import AsyncBoltRequest

        try:
            response = await self._app.async_dispatch(AsyncBoltRequest(body=body, mode="socket_mode"))
        except Exception:
            logger.warning("[Slack] Dispatching a payload from the events socket failed", exc_info=True)
            return
        content = _ack_content(response)
        if content is not None:
            await self._reply_later(body.get("response_url"), content)

    async def _reply_later(self, url: Any, content: dict) -> None:
        if not isinstance(url, str) or self.reply_url.fullmatch(url) is None:
            logger.warning("[Slack] Dropped an ack's content: the payload has no Slack response_url")
            return
        import aiohttp

        try:
            async with aiohttp.ClientSession(trust_env=self._trust_env) as session:
                async with session.post(url, json=content,
                                        timeout=aiohttp.ClientTimeout(total=self.reply_timeout_s)) as answer:
                    if answer.status != 200:
                        logger.warning("[Slack] response_url answered %s to an ack's content", answer.status)
        except Exception as exc:
            logger.warning("[Slack] Posting an ack's content to response_url failed: %s", exc)
