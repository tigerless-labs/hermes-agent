"""Tests for the bundled Meta Model API image_gen plugin (muse-image)."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# The plugin directory uses a hyphen, which is not a valid Python identifier
# for the dotted-import form. Load it via importlib so tests don't need to
# touch sys.path or rename the directory.
meta_plugin = importlib.import_module("plugins.image_gen.meta-ai")


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64

    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _fake_response(*, b64=None, url=None, revised_prompt=None):
    item = SimpleNamespace(b64_json=b64, url=url, revised_prompt=revised_prompt)
    return SimpleNamespace(data=[item])


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Clear every auth + override env var so tests start from a clean slate.
    for env in (
        "MODEL_API_KEY",
        "META_API_KEY",
        "META_MODEL_API_KEY",
        "META_BASE_URL",
        "META_IMAGE_MODEL",
    ):
        monkeypatch.delenv(env, raising=False)
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("META_MODEL_API_KEY", "test-key")
    return meta_plugin.MetaImageGenProvider()


def _patched_openai(fake_client: MagicMock):
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    return patch.dict("sys.modules", {"openai": fake_openai})


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:




    def test_catalog_entries_have_display_speed_strengths_price(self, provider):
        for entry in provider.list_models():
            assert entry["display"]
            assert entry["speed"]
            assert entry["strengths"]
            assert entry["price"]



# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_api_key_unavailable(self):
        assert meta_plugin.MetaImageGenProvider().is_available() is False

    @pytest.mark.parametrize(
        "env", ["MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY"]
    )
    def test_each_auth_alias_makes_available(self, monkeypatch, env):
        monkeypatch.setenv(env, "test")
        assert meta_plugin.MetaImageGenProvider().is_available() is True


# ── Auth / base-url resolution ────────────────────────────────────────────────


class TestResolution:
    def test_api_key_priority_order(self, monkeypatch):
        # MODEL_API_KEY wins over the aliases.
        monkeypatch.setenv("META_MODEL_API_KEY", "third")
        monkeypatch.setenv("META_API_KEY", "second")
        monkeypatch.setenv("MODEL_API_KEY", "first")
        assert meta_plugin._resolve_api_key() == "first"


    def test_base_url_override(self, monkeypatch):
        monkeypatch.setenv("META_BASE_URL", "https://proxy.internal/v1")
        assert meta_plugin._resolve_base_url() == "https://proxy.internal/v1"

    def test_base_url_follows_the_profile_secret_scope(self, monkeypatch):
        """Under multiplexing os.environ is the launch profile's: a routed profile's key must go to
        ITS base URL, and a profile without an override gets the default — never the launch URL."""
        from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

        monkeypatch.setenv("META_BASE_URL", "https://launch.example/v1")
        set_multiplex_active(True)
        try:
            for scope, expected in (({"META_BASE_URL": "https://profile-b.example/v1"}, "https://profile-b.example/v1"),
                                    ({}, "https://api.meta.ai/v1")):
                token = set_secret_scope(scope)
                try:
                    assert meta_plugin._resolve_base_url() == expected
                finally:
                    reset_secret_scope(token)
        finally:
            set_multiplex_active(False)


# ── Model resolution ──────────────────────────────────────────────────────────


class TestModelResolution:

    def test_env_var_override_ignores_unknown(self, monkeypatch):
        monkeypatch.setenv("META_IMAGE_MODEL", "not-a-real-model")
        model_id, _meta = meta_plugin._resolve_model()
        # Unknown id is ignored; falls through to the default.
        assert model_id == meta_plugin.DEFAULT_MODEL

    def test_caller_model_kwarg_wins(self, monkeypatch):
        # The dispatcher forwards top-level image_gen.model as the `model`
        # kwarg; it must beat the env override (#55893 bug class).
        monkeypatch.setitem(
            meta_plugin._MODELS,
            "muse-image-test",
            dict(meta_plugin._MODELS["muse-image-1.0"]),
        )
        monkeypatch.setenv("META_IMAGE_MODEL", "muse-image-1.0")
        model_id, _meta = meta_plugin._resolve_model("muse-image-test")
        assert model_id == "muse-image-test"

    def test_caller_model_unknown_falls_through(self):
        model_id, _meta = meta_plugin._resolve_model("not-a-real-model")
        assert model_id == meta_plugin.DEFAULT_MODEL


# ── Generate ──────────────────────────────────────────────────────────────────


class TestGenerate:
    def test_model_kwarg_reaches_payload(self, provider, monkeypatch):
        monkeypatch.setitem(
            meta_plugin._MODELS,
            "muse-image-test",
            dict(meta_plugin._MODELS["muse-image-1.0"]),
        )
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        with _patched_openai(fake_client):
            result = provider.generate("a cat", model="muse-image-test")
        assert result["success"] is True
        assert (
            fake_client.images.generate.call_args.kwargs["model"] == "muse-image-test"
        )


    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert result["provider"] == "meta-ai"

    def test_missing_api_key(self):
        result = meta_plugin.MetaImageGenProvider().generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_b64_saves_to_cache(self, provider, tmp_path):
        png_bytes = bytes.fromhex(_PNG_HEX)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "muse-image-1.0"
        assert result["aspect_ratio"] == "landscape"
        assert result["provider"] == "meta-ai"
        assert result["modality"] == "text"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        call_kwargs = fake_client.images.generate.call_args.kwargs
        assert call_kwargs["model"] == "muse-image-1.0"
        assert call_kwargs["size"] == "1536x1024"
        assert call_kwargs["n"] == 1

    def test_client_uses_meta_base_url(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider.generate("a cat")

        assert (
            fake_openai.OpenAI.call_args.kwargs["base_url"] == "https://api.meta.ai/v1"
        )

    def test_base_url_override_reaches_client(self, provider, monkeypatch):
        monkeypatch.setenv("META_BASE_URL", "https://proxy.internal/v1")
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider.generate("a cat")

        assert (
            fake_openai.OpenAI.call_args.kwargs["base_url"]
            == "https://proxy.internal/v1"
        )

    @pytest.mark.parametrize(
        "aspect,expected_size",
        [
            ("landscape", "1536x1024"),
            ("square", "1024x1024"),
            ("portrait", "1024x1536"),
        ],
    )
    def test_aspect_ratio_mapping(self, provider, aspect, expected_size):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            provider.generate("a cat", aspect_ratio=aspect)

        assert fake_client.images.generate.call_args.kwargs["size"] == expected_size

    def test_revised_prompt_passed_through(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=_b64_png(),
            revised_prompt="A photo of a cat",
        )

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["revised_prompt"] == "A photo of a cat"

    def test_url_response_is_cached_locally(self, provider):
        """A URL response is materialized locally (symmetric to the openai/xai
        providers) so ephemeral signed URLs can't expire mid-flight."""
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None,
            url="https://example.com/img.webp",
        )

        with (
            _patched_openai(fake_client),
            patch.object(
                meta_plugin,
                "save_url_image",
                return_value=Path("/tmp/meta_20260524_000000_deadbeef.webp"),
            ) as mock_save_url,
        ):
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"].startswith("/")
        assert "example.com" not in result["image"]
        mock_save_url.assert_called_once()

    def test_empty_response_errors(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=None, url=None)

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_api_error_surfaced(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.side_effect = RuntimeError("boom")

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "boom" in result["error"]


# ── Edit (image-to-image) ─────────────────────────────────────────────────────


def _data_url(raw: bytes = bytes.fromhex(_PNG_HEX), mime: str = "image/png") -> str:
    import base64

    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _edit_body(fake_client: MagicMock) -> dict:
    call = fake_client.post.call_args
    assert call.args[0] == "/images/edits"
    return call.kwargs["body"]


class TestEdit:
    def test_capabilities_advertise_editing_within_metas_input_limit(self, provider):
        caps = provider.capabilities()
        assert "image" in caps["modalities"]
        assert caps["max_reference_images"] + 1 == meta_plugin.MAX_SOURCE_IMAGES

    def test_a_source_image_goes_to_the_edits_endpoint_as_json(self, provider):
        fake_client = MagicMock()
        fake_client.post.return_value = {"data": [{"b64_json": _b64_png()}]}
        source = _data_url()

        with _patched_openai(fake_client):
            result = provider.generate("make it blue", aspect_ratio="square", image_url=source)

        assert result["success"] is True
        assert result["modality"] == "image"
        fake_client.images.generate.assert_not_called()
        fake_client.images.edit.assert_not_called()
        body = _edit_body(fake_client)
        assert body["model"] == meta_plugin.DEFAULT_MODEL
        assert body["prompt"] == "make it blue"
        assert body["images"] == [{"image_url": source}]
        assert body["size"] == meta_plugin.size_for("square")
        assert Path(result["image"]).read_bytes() == bytes.fromhex(_PNG_HEX)

    def test_the_primary_image_leads_the_references(self, provider):
        fake_client = MagicMock()
        fake_client.post.return_value = {"data": [{"b64_json": _b64_png()}]}
        primary, ref_a, ref_b = _data_url(b"a"), "https://example.com/b.png", _data_url(b"c")

        with _patched_openai(fake_client):
            provider.generate("compose", image_url=primary, reference_image_urls=[ref_a, ref_b])

        assert _edit_body(fake_client)["images"] == [
            {"image_url": primary}, {"image_url": ref_a}, {"image_url": ref_b}]

    def test_a_local_file_is_sent_inline(self, provider, tmp_path):
        local = tmp_path / "photo.jpg"
        local.write_bytes(b"jpeg-bytes")
        fake_client = MagicMock()
        fake_client.post.return_value = {"data": [{"b64_json": _b64_png()}]}

        with _patched_openai(fake_client):
            provider.generate("sketch it", image_url=str(local))

        assert _edit_body(fake_client)["images"] == [{"image_url": _data_url(b"jpeg-bytes", "image/jpeg")}]

    def test_a_blocked_local_path_is_refused_before_any_request(self, provider):
        fake_client = MagicMock()
        with (
            _patched_openai(fake_client),
            patch("agent.file_safety.raise_if_read_blocked", side_effect=PermissionError("credential file")),
        ):
            result = provider.generate("leak it", image_url="/root/.ssh/id_rsa")

        assert result["success"] is False
        assert result["error_type"] == "io_error"
        fake_client.post.assert_not_called()
        fake_client.images.generate.assert_not_called()

    def test_too_many_sources_are_refused_before_any_request(self, provider):
        fake_client = MagicMock()
        refs = [_data_url(bytes([index])) for index in range(meta_plugin.MAX_SOURCE_IMAGES)]

        with _patched_openai(fake_client):
            result = provider.generate("collage", image_url=_data_url(), reference_image_urls=refs)

        assert result["success"] is False
        assert result["error_type"] == "too_many_references"
        fake_client.post.assert_not_called()

    def test_an_edit_error_is_surfaced(self, provider):
        fake_client = MagicMock()
        fake_client.post.side_effect = RuntimeError("400 invalid image")

        with _patched_openai(fake_client):
            result = provider.generate("make it blue", image_url=_data_url())

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "invalid image" in result["error"]

    def test_an_edit_without_image_data_is_an_empty_response(self, provider):
        fake_client = MagicMock()
        fake_client.post.return_value = {"data": []}

        with _patched_openai(fake_client):
            result = provider.generate("make it blue", image_url=_data_url())

        assert result["success"] is False
        assert result["error_type"] == "empty_response"


class TestEditShape:
    def test_an_edit_without_a_requested_ratio_keeps_the_sources_shape(self, provider):
        fake_client = MagicMock()
        fake_client.post.return_value = {"data": [{"b64_json": _b64_png()}]}

        with _patched_openai(fake_client):
            result = provider.generate("make it blue", image_url=_data_url())

        assert result["success"] is True
        assert "size" not in _edit_body(fake_client)

    def test_an_edit_with_a_requested_ratio_sends_its_size(self, provider):
        fake_client = MagicMock()
        fake_client.post.return_value = {"data": [{"b64_json": _b64_png()}]}

        with _patched_openai(fake_client):
            provider.generate("side by side", aspect_ratio="landscape", image_url=_data_url())

        assert _edit_body(fake_client)["size"] == meta_plugin.size_for("landscape")

    def test_text_to_image_without_a_ratio_uses_the_default(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            provider.generate("a cat")

        from agent.image_gen_provider import DEFAULT_ASPECT_RATIO

        assert fake_client.images.generate.call_args.kwargs["size"] == meta_plugin.size_for(DEFAULT_ASPECT_RATIO)
