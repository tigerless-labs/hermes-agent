"""Deliver ``MEDIA:<path>`` files that live inside a remote terminal sandbox (#466).

``validate_media_delivery_path`` only accepts files on the gateway host. When the terminal backend
is ssh / daytona / vercel (any backend that reports its ``_remote_home``), the agent's artifact is
on another filesystem, so the tag was silently dropped. This module pulls the file through the
active environment's ``fetch_file`` into the document cache (already an allowlisted delivery root)
and hands back the host copy. Backends without a known remote home (docker outside its mounts,
modal, singularity) still get the conservative any-component denylist, so a ``/root/...`` artifact
there is not fetched — set ``_remote_home`` on the environment to opt in.

The remote path is screened against the same denylist as local deliveries BEFORE any bytes move,
and again after ``readlink -f`` (fail closed when it cannot resolve) — a remote fetch must never
become a bypass of the host denylist. Strict mode (``HERMES_MEDIA_DELIVERY_STRICT``) keeps its
pre-existing behaviour: nothing is fetched, since a fetched copy would land in an allowlisted root
and skip the recency gate strict mode exists for. With ``gateway.trust_session_sandbox`` it fetches
from one sandbox only — the delivering session's own per-session container, whose files are that
session's own work — and never from the engine's mounts inside it (``~/.hermes``: credential files,
skills, config), whatever their names.
"""

from __future__ import annotations

import logging
import os
import posixpath
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import Optional

from gateway.platforms.base import (
    _MEDIA_DELIVERY_DENIED_HOME_SUBPATHS, _MEDIA_DELIVERY_DENIED_PREFIXES, _ROOT_CREDENTIAL_PATHS)

logger = logging.getLogger(__name__)

# Telegram bot uploads cap at 50 MB; the other platforms are in the same range. Mirrors
# tools.image_source._MAX_INGEST_BYTES.
_FETCH_MAX_BYTES = 50 * 1024 * 1024

_DENIED_PREFIXES = tuple(PurePosixPath(p) for p in _MEDIA_DELIVERY_DENIED_PREFIXES)
# Credential dirs under the sandbox home plus the Hermes stores, which live at ``~/.hermes`` there.
_DENIED_HOME_RELATIVE = tuple(PurePosixPath(s) for s in _MEDIA_DELIVERY_DENIED_HOME_SUBPATHS) + tuple(
    PurePosixPath(".hermes", *PurePosixPath(rel.replace(os.sep, "/")).parts) for rel in _ROOT_CREDENTIAL_PATHS)


def remote_path_is_denied(path: str, remote_home: Optional[str]) -> bool:
    """Pure string check (the remote fs can't be stat'd from here) applying the host denylist to a
    sandbox path. Unknown home ⇒ home-relative entries match ANY path component (conservative)."""
    target = PurePosixPath(posixpath.normpath(path))
    if not target.is_absolute():
        return True
    home = PurePosixPath(posixpath.normpath(remote_home)) if remote_home else None

    def _under(root: PurePosixPath) -> bool:
        return target == root or root in target.parents

    # The sandbox's own home may be a denied system prefix (/root); its credential subpaths are
    # separate, more specific entries — same exception as _path_under_denied_prefix.
    if any(_under(p) for p in _DENIED_PREFIXES if p != home):
        return True
    if home is not None:
        return any(_under(home / rel) for rel in _DENIED_HOME_RELATIVE)
    parts = target.parts
    return any(parts[i:i + len(rel.parts)] == rel.parts
               for rel in _DENIED_HOME_RELATIVE for i in range(len(parts) - len(rel.parts) + 1))


def _active_remote_env():
    """The live remote BaseEnvironment for the current session, or None (local backend / no env yet).
    Keyed by the session id the turn registered its sandbox under — or, once the turn is over, the id of
    the session whose reply is being delivered (falls back to the session key)."""
    from agent.prompt_builder import _REMOTE_TERMINAL_BACKENDS, _plugin_backend_is_remote
    from gateway.platforms.base import _tenv
    from gateway.session_context import delivering_session_id, get_session_env
    from tools.terminal_tool_lifecycle import get_active_env
    backend = _tenv("TERMINAL_ENV", "local").strip().lower()
    if backend not in _REMOTE_TERMINAL_BACKENDS and not _plugin_backend_is_remote(backend):
        return None
    return get_active_env(get_session_env("HERMES_SESSION_ID") or delivering_session_id()
                          or get_session_env("HERMES_SESSION_KEY") or "default")


def _own_session_sandbox():
    """The delivering session's own per-session sandbox, or None. Strict delivery trusts no other: not
    a shared container, not another session's, and not one named by a session id from ``os.environ``."""
    from gateway.session_context import delivering_session_id
    from tools.terminal_tool_lifecycle import get_active_env
    session_id = delivering_session_id()
    env = get_active_env(session_id) if session_id else None
    return env if getattr(env, "_session_scoped", False) else None


def _under_engine_mounts(path: str, remote_home: Optional[str]) -> bool:
    """``~/.hermes`` inside the sandbox: the engine's credential files, skills and config are mounted
    there under names no denylist can enumerate. Unknown home ⇒ any ``.hermes`` component."""
    target = PurePosixPath(posixpath.normpath(path))
    if not remote_home:
        return ".hermes" in target.parts
    mounts = PurePosixPath(posixpath.normpath(remote_home)) / ".hermes"
    return target == mounts or mounts in target.parents


def _held_back(path: str, remote_home: Optional[str], own_sandbox_only: bool) -> bool:
    return remote_path_is_denied(path, remote_home) or (own_sandbox_only and _under_engine_mounts(path, remote_home))


def fetch_remote_media(path: str) -> Optional[str]:
    """Host path of a validated copy of sandbox file ``path``, or None (never raises). Only fires
    when a remote backend is active; the caller has already failed local validation."""
    from gateway.media_policy import media_delivery_strict, media_delivery_trust_session_sandbox
    own_sandbox_only = media_delivery_strict()
    if own_sandbox_only and not media_delivery_trust_session_sandbox():
        return None
    env = _own_session_sandbox() if own_sandbox_only else _active_remote_env()
    if env is None:
        return None
    from gateway.platforms.base import (
        _log_safe_path, _normalize_media_tag_path, get_document_cache_dir, validate_media_delivery_path)
    from tools.environments.base import FileFetchError

    remote_home = getattr(env, "_remote_home", None)
    candidate = posixpath.normpath(_normalize_media_tag_path(str(path)) or "")
    if candidate == "~" or candidate.startswith("~/"):
        if not remote_home:
            return None
        candidate = posixpath.normpath(posixpath.join(remote_home, candidate[2:]))
    if not candidate.startswith("/") or _held_back(candidate, remote_home, own_sandbox_only):
        return None
    try:
        # ``[ -f ]`` in fetch_file follows symlinks, so the link TARGET is what gets screened;
        # an unresolvable path fails closed rather than trusting the unresolved name.
        resolved = env.fetch_realpath(candidate)
        if resolved is None or _held_back(resolved, remote_home, own_sandbox_only):
            return None
        basename = re.sub(r"[^\w.\-]", "_", posixpath.basename(resolved)) or "file"
        # The chat's own document cache when caches are chat-scoped: the only one delivery accepts.
        dest = Path(get_document_cache_dir()) / f"remote_{uuid.uuid4().hex[:12]}_{basename}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        env.fetch_file(resolved, dest, max_bytes=_FETCH_MAX_BYTES)
    except FileFetchError as exc:
        logger.warning("Remote media fetch of %s skipped: %s", _log_safe_path(candidate), exc)
        return None
    except Exception:
        logger.warning("Remote media fetch of %s failed", _log_safe_path(candidate), exc_info=True)
        return None
    validated = validate_media_delivery_path(str(dest))
    if not validated:
        dest.unlink(missing_ok=True)
        return None
    logger.info("Fetched remote media %s from the %s sandbox", _log_safe_path(candidate), type(env).__name__)
    return validated
