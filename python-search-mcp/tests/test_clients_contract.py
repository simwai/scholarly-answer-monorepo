"""Interface-contract tests for the LLM clients.

These exist because an accidental duplicate class definition once shadowed the
real implementation without breaking the import, leaving the object with almost
no methods. An import succeeding is not evidence a class is intact, so the public
surface is asserted explicitly.
"""

from __future__ import annotations

import requests

from scholarly_answer_engine.synth.client import LLMClient, LLMError
from scholarly_answer_engine.synth.inline_client import InlineChainClient
from scholarly_answer_engine.synth.router_client import RouterClient


def test_both_clients_implement_the_interface():
    assert issubclass(RouterClient, LLMClient)
    assert issubclass(InlineChainClient, LLMClient)


def test_inline_client_exposes_its_full_surface():
    """Guard against a redefinition that silently drops methods."""
    required = {
        "complete",
        "describe",
        "_kilo_call",
        "_kilo_auto_free_call",
        "_supported_parameters",
        "_kilo_reasoning_params",
        "_nim_call",
        "_post_with_retry",
        "_poll_nim",
        "_read_completion",
    }
    missing = [name for name in required if not hasattr(InlineChainClient, name)]
    assert not missing, f"InlineChainClient is missing methods: {missing}"


def test_router_client_exposes_its_full_surface():
    required = {"complete", "describe"}
    missing = [name for name in required if not hasattr(RouterClient, name)]
    assert not missing, f"RouterClient is missing methods: {missing}"


def test_router_client_requires_a_pin():
    import pytest

    with pytest.raises(ValueError, match="pin list is empty"):
        RouterClient(base_url="http://127.0.0.1:1", pin=[])


def test_router_client_describes_itself():
    client = RouterClient(base_url="http://127.0.0.1:8787", pin=["a:free", "b:free"])
    assert "8787" in client.describe()


def test_inline_client_describes_the_chain_with_a_nim_key():
    """With a NIM key, Step 5 leads the chain."""
    client = InlineChainClient(nim_api_key="nvapi-test")
    description = client.describe()
    assert "stepfun-ai/step-5-preview" in description
    assert "nemotron" in description


def test_inline_client_omits_nim_without_a_key():
    """Without a NIM key the NIM tier is skipped, and that is stated."""
    client = InlineChainClient(nim_api_key="")
    description = client.describe()
    assert "no NIM key" in description
    assert "nemotron" in description


class _StubSession:
    """A session whose get/post always fail, recording the call.

    Raises ``requests.ConnectionError`` because the production code catches
    ``requests.RequestException`` specifically - a stub must raise what real
    transport code would raise, not a bare OSError.
    """

    def __init__(self, submitted: list) -> None:
        self.submitted = submitted

    def get(self, *args, **kwargs):
        self.submitted.append(("get", args, kwargs))
        raise requests.ConnectionError("network disabled in tests")

    def post(self, *args, **kwargs):
        self.submitted.append(("post", args, kwargs))
        raise requests.ConnectionError("network disabled in tests")


def _stub_session(submitted: list) -> _StubSession:
    return _StubSession(submitted)


def test_router_client_reports_unreachable_as_llm_error():
    submitted: list = []
    client = RouterClient(
        base_url="http://127.0.0.1:1", pin=["x:free"], timeout=2.0, session=_stub_session(submitted)
    )
    try:
        client.complete("sys", "user")
    except LLMError as exc:
        assert "unreachable" in str(exc)
        assert submitted, "the router was never called"
    else:
        raise AssertionError("expected LLMError for an unreachable router")


def test_inline_client_propagates_llm_error_when_all_tiers_fail():
    submitted: list = []
    client = InlineChainClient(timeout=2.0, nim_api_key="nvapi-test")
    client.session = _stub_session(submitted)
    try:
        client.complete("sys", "user")
    except LLMError as exc:
        assert "inline chain" in str(exc)
        # NIM (Step 5), then the two Kilo tiers, were all attempted.
        assert len(submitted) >= 3, f"expected all tiers attempted, saw {len(submitted)}"
    else:
        raise AssertionError("expected LLMError when every tier is unreachable")


def test_inline_chain_skips_nim_without_a_key():
    submitted: list = []
    client = InlineChainClient(timeout=2.0, nim_api_key="")
    client.session = _stub_session(submitted)
    try:
        client.complete("sys", "user")
    except LLMError:
        pass
    urls = [call[1][0] for call in submitted if call[0] == "post"]
    assert not any("nvidia.com" in u for u in urls), "NIM must be skipped without a key"
    assert any("ai-gateway.kilo.ai" in u for u in urls), "Kilo tiers should still run"


def test_inline_chain_attempts_tiers_in_order():
    submitted: list = []
    client = InlineChainClient(timeout=2.0, nim_api_key="nvapi-test")
    client.session = _stub_session(submitted)
    try:
        client.complete("sys", "user")
    except LLMError:
        pass
    urls = [call[1][0] for call in submitted if call[0] == "post"]
    nim = [u for u in urls if "integrate.api.nvidia.com" in u]
    kilo = [u for u in urls if "ai-gateway.kilo.ai" in u]
    assert nim and kilo, "both providers should be attempted"
    # NIM (Step 5 first) runs before Kilo.
    assert urls.index(nim[0]) < urls.index(kilo[0]), "NIM Step 5 should lead the chain"


def test_inline_client_sends_nim_stream_false_and_reasoning_kwargs():
    submitted: list = []
    client = InlineChainClient(timeout=2.0, nim_api_key="nvapi-test")
    client.session = _stub_session(submitted)
    try:
        client.complete("sys", "user")
    except LLMError:
        pass
    nim_posts = [c for c in submitted if c[0] == "post" and "nvidia.com" in c[1][0]]
    assert nim_posts, "NIM was never called"
    payload = nim_posts[0][2]["json"]
    # Step 5 leads the chain on NIM.
    assert payload["model"] == "stepfun-ai/step-5-preview"
    assert payload["stream"] is False, "NIM requires stream=false explicitly"
    assert payload["reasoning_budget"] == 16384
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["max_tokens"] >= payload["reasoning_budget"]


def test_inline_client_reads_kilo_catalog_for_reasoning_support():
    submitted: list = []
    client = InlineChainClient(timeout=2.0)
    client.session = _stub_session(submitted)
    # No catalog available, so reasoning support is unknown and degrades visibly.
    params, supported = client._kilo_reasoning_params()
    assert params == {}
    assert supported is False
    assert submitted, "the Kilo catalog should have been consulted"
