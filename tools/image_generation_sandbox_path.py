"""``image_gen.sandbox_path_first``: a sandboxed model reads the generated image's sandbox path first.

Under a sandboxed terminal the result names the file as ``image`` (the host path) and ``agent_visible_image``
(where the sandbox sees it). Models took ``image``, found nothing there in the sandbox and only then tried the
other field. With the switch on, ``image`` is the sandbox path; ``host_image`` keeps the host path, which the
gateway reads first when it attaches the image.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def sandbox_path_first() -> bool:
    try:
        from hermes_cli.config import load_config
        from utils import is_truthy_value

        section = load_config().get("image_gen")
        return isinstance(section, dict) and is_truthy_value(section.get("sandbox_path_first"), default=False)
    except Exception as error:
        logger.warning("image_gen.sandbox_path_first unreadable (%s); the result keeps the host path", error)
        return False


def put_sandbox_path_first(payload: dict[str, Any], sandbox_path: str) -> None:
    if sandbox_path_first():
        payload["image"] = sandbox_path
