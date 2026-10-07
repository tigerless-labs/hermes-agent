from __future__ import annotations

import pytest

from agent import image_gen_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class TestPluginDispatch:


    def test_deepinfra_key_alone_does_not_select_image_backend(self, monkeypatch):
        """DeepInfra chat credentials do not imply consent to image billing."""
        from tools import image_generation_tool

        monkeypatch.setenv("DEEPINFRA_API_KEY", "«redacted:sk-…»")
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: None)
        assert image_generation_tool._dispatch_to_plugin_provider("a cat", "square") is None

    def test_requirements_ignore_unselected_paid_plugin(self, monkeypatch):
        from tools import image_generation_tool

        monkeypatch.setattr(image_generation_tool, "check_fal_api_key", lambda: False)
        monkeypatch.setattr(
            image_generation_tool, "_read_configured_image_provider", lambda: None
        )
        assert image_generation_tool.check_image_generation_requirements() is False


_UNASKED = object()


class _RecordingProvider:
    name = "recording"
    display_name = "Recording"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, aspect_ratio=_UNASKED, **kwargs):
        self.calls.append({"aspect_ratio": aspect_ratio, **kwargs})
        return {"success": True, "image": "https://example.com/x.png"}


class TestAspectRatioReachesPluginsOnlyWhenAsked:
    @pytest.fixture
    def provider(self, monkeypatch):
        from tools import image_generation_tool

        recording = _RecordingProvider()
        monkeypatch.setattr(image_generation_tool, "_plugin_provider_name", lambda: "recording")
        monkeypatch.setattr(image_generation_tool, "_get_plugin_provider", lambda name, force=False: recording)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_model", lambda: None)
        monkeypatch.setenv("TERMINAL_ENV", "local")
        return recording

    def test_an_unasked_ratio_is_left_to_the_provider(self, provider):
        from tools import image_generation_tool

        image_generation_tool._handle_image_generate({"prompt": "edit it", "image_url": "https://example.com/a.png"})
        [call] = provider.calls
        assert call["aspect_ratio"] is _UNASKED

    def test_an_asked_ratio_reaches_the_provider(self, provider):
        from tools import image_generation_tool

        image_generation_tool._handle_image_generate({"prompt": "a cat", "aspect_ratio": "square"})
        [call] = provider.calls
        assert call["aspect_ratio"] == "square"
