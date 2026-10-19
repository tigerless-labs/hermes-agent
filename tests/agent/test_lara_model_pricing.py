"""The models Lara runs on are priced from the bundled official-docs snapshot, with no network (fork addition).

On the engine VM the egress list does not reach models.dev or a provider's /models endpoint, so a model
without a bundled row is priced "unknown" and every downstream usage report shows $0.
"""

from decimal import Decimal

import pytest

from agent import usage_pricing
from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, get_pricing_entry

ANTHROPIC = ("anthropic", "https://api.anthropic.com")
META = ("meta-ai", "https://api.meta.ai/v1")
LARA_MODELS = [("claude-haiku-5-5", *ANTHROPIC), ("claude-sonnet-5-5", *ANTHROPIC), ("claude-opus-5-5", *ANTHROPIC),
               ("muse-spark-1.3", *META)]
HAIKU_TIER = 100_000


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def unreachable(*args, **kwargs):
        raise AssertionError("pricing reached the network")
    monkeypatch.setattr(usage_pricing, "fetch_endpoint_model_metadata", unreachable)
    monkeypatch.setattr(usage_pricing, "_models_dev_pricing_entry", unreachable)


@pytest.mark.parametrize("model,provider,base_url", LARA_MODELS)
def test_each_model_lara_runs_on_is_priced_from_the_bundled_official_snapshot(model, provider, base_url):
    entry = get_pricing_entry(model, provider=provider, base_url=base_url)
    assert entry is not None and entry.source == "official_docs_snapshot" and entry.source_url
    cost = estimate_usage_cost(model, CanonicalUsage(input_tokens=1000, output_tokens=1000), provider=provider,
                               base_url=base_url)
    assert cost.status == "estimated" and cost.amount_usd > 0


@pytest.mark.parametrize("model,provider,base_url", LARA_MODELS)
def test_cache_hits_cost_less_than_fresh_input_and_output_costs_more(model, provider, base_url):
    entry = get_pricing_entry(model, provider=provider, base_url=base_url)
    assert entry.cache_read_cost_per_million < entry.input_cost_per_million < entry.output_cost_per_million


def _haiku(prompt_tokens: int) -> Decimal:
    usage = CanonicalUsage(input_tokens=1000, cache_read_tokens=prompt_tokens - 1000, output_tokens=1000)
    return estimate_usage_cost("claude-haiku-5-5", usage, provider=ANTHROPIC[0], base_url=ANTHROPIC[1]).amount_usd


def test_a_haiku_request_whose_whole_prompt_passes_the_tier_pays_the_higher_rate_on_all_of_it():
    entry = get_pricing_entry("claude-haiku-5-5", provider=ANTHROPIC[0], base_url=ANTHROPIC[1])
    assert entry.tier_threshold_tokens == HAIKU_TIER
    for rate in ("input", "output", "cache_read", "cache_write"):
        assert getattr(entry, f"{rate}_cost_per_million_above") > getattr(entry, f"{rate}_cost_per_million")
    assert _haiku(HAIKU_TIER + 1) > _haiku(HAIKU_TIER) * 2
