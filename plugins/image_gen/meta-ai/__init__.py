"""Meta Model API (``muse-image``): OpenAI-compatible (https://api.meta.ai/v1), so the OpenAI SDK
is pointed at Meta's base URL with ``META_MODEL_API_KEY``. Output is base64 WebP → image cache.
Source images route to ``/v1/images/edits`` as Meta's JSON body (``images`` of data or public URLs):
keys with Zero Data Retention reject the SDK's multipart ``images.edit()``. An edit with no requested
aspect ratio sends no ``size``, so Meta keeps the source image's shape.
Selection: ``model`` kwarg → ``META_IMAGE_MODEL`` → ``image_gen.meta-ai.model`` → ``image_gen.model``
→ :data:`DEFAULT_MODEL`."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.secret_scope import get_secret, get_secret_str
from agent.image_gen_provider import (
    resolve_aspect_ratio, save_b64_image, save_url_image, success_response)
from plugins.image_gen._common import (
    StaticImageGenProvider, collect_source_images, error_factory, import_openai, openai_importable,
    prompt_required_error, resolve_static_model, size_for)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
# Auth env vars in priority order (mirrors the ``meta-ai`` chat provider); MODEL_API_KEY is Meta's
# documented var, the rest are aliases. ``API_KEY_ENV`` is the one shown in setup/errors.
API_KEY_ENVS = ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY")
API_KEY_ENV = "META_MODEL_API_KEY"
BASE_URL_ENV = "META_BASE_URL"  # optional override, same var the chat provider honors


def _resolve_api_key() -> Optional[str]:
    """First non-empty auth env var, in priority order."""
    return next((val for val in map(get_secret, API_KEY_ENVS) if val), None)


def _resolve_base_url() -> str:
    # Through the secret scope like the key: under multiplexing os.environ holds the launch profile's
    # endpoint, and a routed profile's key must never be sent to another profile's base URL.
    return get_secret_str(BASE_URL_ENV).strip() or DEFAULT_BASE_URL


# Model ids are sent verbatim to ``/v1/images/generations``.
_MODELS: Dict[str, Dict[str, Any]] = {
    "muse-image-1.0": {
        "display": "Muse Image 1.0",
        "speed": "~10s",
        "strengths": "Meta Model API image generation",
        "price": "$0.01/image",
    },
}
DEFAULT_MODEL = "muse-image-1.0"
# ``/v1/images/edits`` takes 1–10 input images.
MAX_SOURCE_IMAGES = 10
_PASSTHROUGH_PREFIXES = ("http://", "https://", "data:")


def _resolve_model(caller_model: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    return resolve_static_model(
        _MODELS, DEFAULT_MODEL, env_var="META_IMAGE_MODEL", config_key="meta-ai", explicit=caller_model,
    )


def _image_entry(source: str) -> Dict[str, str]:
    """One ``images`` entry: URLs and data URIs pass through; a local path is inlined as a data URI."""
    if source.lower().startswith(_PASSTHROUGH_PREFIXES):
        return {"image_url": source}
    from agent.file_safety import raise_if_read_blocked  # credential-read guard before local bytes

    path = os.path.expanduser(source)
    raise_if_read_blocked(path)
    with open(path, "rb") as fh:  # windows-footgun: ok
        raw = fh.read()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return {"image_url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"}


class MetaImageGenProvider(StaticImageGenProvider):
    """Meta Model API ``images.generate`` backend (muse-image)."""

    provider_id = "meta-ai"
    label = "Meta Model API"
    models = _MODELS
    default_model_id = DEFAULT_MODEL
    setup = dict(
        name="Meta Model API", badge="paid", tag="Muse Image via Meta Model API (api.meta.ai)",
        key=API_KEY_ENV, prompt="Meta Model API key (LLM|... token)", url="https://api.meta.ai")

    def is_available(self) -> bool:
        return bool(_resolve_api_key()) and openai_importable()

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": MAX_SOURCE_IMAGES - 1,
                "max_source_images": MAX_SOURCE_IMAGES}

    def generate(
        self, prompt: str, aspect_ratio: Optional[str] = None, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return prompt_required_error("meta-ai", aspect)
        api_key = _resolve_api_key()
        if not api_key:
            return error_factory("meta-ai", aspect)(
                f"{API_KEY_ENV} not set. Run `hermes tools` -> Image "
                "Generation -> Meta Model API to configure.",
                "auth_required")

        openai, err = import_openai("meta-ai", aspect)
        if err:
            return err
        model_id, _meta = _resolve_model(kwargs.get("model"))
        size = size_for(aspect)
        fail = error_factory("meta-ai", aspect, model=model_id, prompt=prompt)
        sources = collect_source_images(image_url, reference_image_urls)
        if len(sources) > MAX_SOURCE_IMAGES:
            return fail(f"Meta image editing takes at most {MAX_SOURCE_IMAGES} source images",
                        "too_many_references")
        try:
            images = [_image_entry(source) for source in sources]
        except Exception as exc:
            return fail(f"Could not load source image for editing: {exc}", "io_error")
        client = openai.OpenAI(api_key=api_key, base_url=_resolve_base_url())
        keep_source_shape = bool(images) and not aspect_ratio
        try:
            if images:
                edit_body = {"model": model_id, "prompt": prompt, "images": images, "n": 1}
                if not keep_source_shape:
                    edit_body["size"] = size
                first = _first_item(client.post("/images/edits", cast_to=object, body=edit_body))
            else:
                first = _first_item(client.images.generate(model=model_id, prompt=prompt, size=size, n=1))
        except Exception as exc:
            logger.debug("Meta image generation failed", exc_info=True)
            return fail(f"Meta image generation failed: {exc}", "api_error")
        if first is None:
            return fail("Meta response contained no image data", "empty_response")

        b64 = _field(first, "b64_json")
        url = _field(first, "url")
        try:
            if b64:
                image_ref = str(save_b64_image(b64, prefix="meta", extension="webp"))
            elif url:
                image_ref = str(save_url_image(url, prefix="meta"))
            else:
                return fail("Meta response contained neither b64_json nor URL", "empty_response")
        except Exception as exc:
            return fail(f"Failed to save Meta image: {exc}", "io_error")
        extra: Dict[str, Any] = {} if keep_source_shape else {"size": size}
        if _field(first, "revised_prompt"):
            extra["revised_prompt"] = _field(first, "revised_prompt")
        return success_response(
            image=image_ref, model=model_id, prompt=prompt, aspect_ratio=aspect, provider="meta-ai",
            modality="image" if images else "text", extra=extra)


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _first_item(response: Any) -> Any:
    """``data[0]`` of an SDK response object or of a raw JSON body; ``None`` when absent."""
    data = _field(response, "data")
    try:
        return data[0]
    except (IndexError, KeyError, TypeError):
        return None


def register(ctx) -> None:
    """Plugin entry point -- wire ``MetaImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MetaImageGenProvider())
