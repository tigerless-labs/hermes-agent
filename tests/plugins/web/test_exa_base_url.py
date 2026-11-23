"""``EXA_BASE_URL`` points the Exa client at another address (a key-holding forwarder beside the
agent); unset, the client is built exactly as before and talks to Exa itself."""

from __future__ import annotations

import sys
import types

import pytest

import tools.web_tools as web_tools
from plugins.web import _common
from plugins.web.exa import provider

EXA = "https://api.exa.ai"
FORWARDER = "http://127.0.0.1:15601"


class _Exa:
    def __init__(self, api_key: str, base_url: str = EXA) -> None:
        self.api_key, self.base_url, self.headers = api_key, base_url, {}


@pytest.fixture
def exa(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "exa_py", types.SimpleNamespace(Exa=_Exa))
    monkeypatch.setattr(_common, "lazy_ensure", lambda feature: None)
    monkeypatch.setattr(web_tools, "_exa_client", None)
    monkeypatch.setenv("EXA_API_KEY", "a-key")
    monkeypatch.delenv("EXA_BASE_URL", raising=False)
    return monkeypatch


def test_unset_the_client_talks_to_exa_itself(exa) -> None:
    assert provider._get_exa_client().base_url == EXA


def test_blank_counts_as_unset(exa) -> None:
    exa.setenv("EXA_BASE_URL", "  ")
    assert provider._get_exa_client().base_url == EXA


def test_set_the_client_talks_to_that_address_with_the_same_key_and_integration_header(exa) -> None:
    exa.setenv("EXA_BASE_URL", FORWARDER)
    client = provider._get_exa_client()
    assert (client.base_url, client.api_key) == (FORWARDER, "a-key")
    assert client.headers["x-exa-integration"] == "hermes-agent"


def test_the_address_does_not_make_the_tool_available_without_a_key(exa) -> None:
    exa.setenv("EXA_BASE_URL", FORWARDER)
    exa.delenv("EXA_API_KEY")
    with pytest.raises(ValueError):
        provider._get_exa_client()
