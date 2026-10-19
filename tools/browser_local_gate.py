"""Off switch for local browser sessions (Lara fork addition).

``browser.local_sessions: false`` makes every path that would start a browser on the engine's own host fail
instead: no cloud provider loaded, a cloud provider that failed, the private-URL sidecar, a real-profile attach.
A deployment whose browsers must stay inside a sandbox uses it so that no failure quietly moves the browser out.
Default on, as upstream.
"""

from utils import is_truthy_value

LOCAL_SESSIONS_KEY = "local_sessions"


class LocalBrowserRefused(RuntimeError):
    pass


def local_sessions_allowed() -> bool:
    from hermes_cli.config import load_config_readonly
    browser = load_config_readonly().get("browser") or {}
    return is_truthy_value(browser.get(LOCAL_SESSIONS_KEY), default=True)


def refuse_local_session(task_id: str) -> None:
    if not local_sessions_allowed():
        raise LocalBrowserRefused(f"Local browser sessions are off (browser.{LOCAL_SESSIONS_KEY}); no browser starts "
                                  f"on this host for task {task_id}.")
