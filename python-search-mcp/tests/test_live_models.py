"""Live verification that the configured LLM tier IDs actually exist.

This is the single load-bearing assumption in the LLM configuration, and it has
been wrong twice: a StepFun free ID that does not exist on Kilo, and a NIM
Nemotron ID in the wrong long form. Free-tier IDs also disappear server-side
without notice.

Run with:  RUN_LIVE_TESTS=1 python -m pytest tests -k live
"""

from __future__ import annotations

import json
import os
import urllib.request

import pytest

from scholarly_answer_engine.config import DEFAULT_PIN, KILO_FREE_NEMOTRON, NIM_STEP5

KILO_CATALOG = "https://ai-gateway.kilo.ai/api/v1/models"
NIM_CATALOG = "https://integrate.api.nvidia.com/v1/models"

live_only = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"), reason="set RUN_LIVE_TESTS=1 to verify live model IDs"
)


def _fetch(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "scholarly-answer-engine/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _catalog(url: str) -> dict[str, dict]:
    return {m.get("id"): m for m in _fetch(url).get("data", []) if m.get("id")}


def _is_free(model: dict) -> bool:
    price = model.get("pricing") or {}
    return str(price.get("prompt")) == "0" and str(price.get("completion")) == "0"


@live_only
def test_stepfun_step5_exists_on_nim():
    """Step 5 Preview is a free endpoint on NIM under the stepfun-ai/ prefix."""
    catalog = _catalog(NIM_CATALOG)
    assert NIM_STEP5 in catalog, (
        f"{NIM_STEP5} not in NIM's catalog; all step* IDs: "
        f"{[k for k in catalog if 'step' in k]}"
    )


@live_only
def test_stepfun_step5_is_not_confused_with_the_paid_kilo_id():
    """Guards the mistake that made this default wrong twice.

    `stepfun-ai/step-5-preview` (NIM, free) and `stepfun/step-5-preview`
    (Kilo, ~$1/M) are different IDs for the same model.
    """
    kilo = _catalog(KILO_CATALOG)
    assert NIM_STEP5 not in kilo, "the NIM id must not be sent to Kilo"
    assert "stepfun/step-5-preview" in kilo, "Kilo does serve a paid Step 5"
    assert not _is_free(kilo["stepfun/step-5-preview"]), "Kilo's Step 5 is paid, not free"


@live_only
def test_every_configured_tier_exists_and_is_free():
    """Each tier must be a real, reachable, genuinely free model."""
    pin = [p.strip() for p in DEFAULT_PIN.split(",") if p.strip()]
    assert pin, "DEFAULT_PIN is empty"

    kilo = _catalog(KILO_CATALOG)
    nim = _catalog(NIM_CATALOG)

    unresolved: list[str] = []
    for model_id in pin:
        if model_id in kilo:
            assert _is_free(kilo[model_id]), f"Kilo tier {model_id} is not free"
        elif model_id in nim:
            # NIM's list carries no pricing; free status comes from build.nvidia.com.
            continue
        else:
            unresolved.append(model_id)

    assert not unresolved, f"configured tiers resolve on neither provider: {unresolved}"


@live_only
def test_kilo_free_tier_has_reasoning_support():
    """The Kilo fallback tier must be able to run 'thinking high'."""
    catalog = _catalog(KILO_CATALOG)
    model = catalog.get(KILO_FREE_NEMOTRON)
    assert model is not None, f"{KILO_FREE_NEMOTRON} missing from Kilo"
    supported = set(model.get("supported_parameters") or [])
    assert supported & {"reasoning", "reasoning_effort"}, (
        f"{KILO_FREE_NEMOTRON} supports no reasoning parameter; supported={sorted(supported)}"
    )
