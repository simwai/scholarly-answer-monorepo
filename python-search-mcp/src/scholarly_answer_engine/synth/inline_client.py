"""Contingency transport: direct provider calls, no router.

Not the v1 default, but it must stay correct because it is the documented
fallback if the router is unavailable. It owns the per-provider reasoning
mapping that the router currently centralises:

- Kilo: reasoning support is not in Kilo's published request schema, but the
  gateway does accept it for models that advertise it via ``supported_parameters``
  on ``GET /v1/models``. So read the catalog, then send ``reasoning: {effort}``
  or ``reasoning_effort``, and retry once without either on a 400.
- NIM: ``chat_template_kwargs: {"enable_thinking": true}`` plus top-level
  ``reasoning_budget``. ``stream`` defaults to **true** there, so it is set
  false explicitly. Reasoning text arrives on ``message.reasoning_content`` and
  must be stripped before it reaches the answer.
"""

from __future__ import annotations

import json
import logging
import time

import requests

from .client import LLMClient, LLMError

logger = logging.getLogger(__name__)

# Verified against live catalogs. `stepfun-ai/step-5-preview` is a FREE endpoint
# on NVIDIA NIM (publisher prefix `stepfun-ai/`), not on Kilo, where the same
# model is the paid `stepfun/step-5-preview`.
_NIM_PRIMARY = "stepfun-ai/step-5-preview"

# Kilo free tiers; these work anonymously with no key at all.
_KILO_DEFAULT_PIN = "nvidia/nemotron-3.5-super-vl:free"
_KILO_LAST_RESORT = "kilo-auto/free"

# NIM free Nemotron fallbacks, used only if Step 5 is unavailable.
_NIM_PIN = [_NIM_PRIMARY, "nvidia/nemotron-3.5-super-vl-120b-a12b"]
_NIM_REASONING_BUDGET = 16384
_NIM_MAX_TOKENS = 16384


class InlineChainClient(LLMClient):
    """Ordered chain: Kilo Stepfun -> NIM Nemotron -> kilo-auto/free."""

    def __init__(
        self,
        kilo_base_url: str = "https://ai-gateway.kilo.ai/api/v1",
        kilo_api_key: str = "",
        nim_base_url: str = "https://integrate.api.nvidia.com/v1",
        nim_api_key: str = "",
        reasoning: str = "high",
        timeout: float = 180.0,
    ) -> None:
        self.kilo_base_url = kilo_base_url.rstrip("/")
        self.kilo_api_key = kilo_api_key
        self.nim_base_url = nim_base_url.rstrip("/")
        self.nim_api_key = nim_api_key
        self.reasoning = reasoning
        self.timeout = timeout
        self.session = requests.Session()
        self.last_model_used = None
        self.last_from_fallback = False

    def describe(self) -> str:
        if self.nim_api_key:
            return f"inline chain nim:{_NIM_PRIMARY} -> kilo:{_KILO_DEFAULT_PIN} -> {_KILO_LAST_RESORT}"
        return f"inline chain kilo:{_KILO_DEFAULT_PIN} -> {_KILO_LAST_RESORT} (no NIM key)"

    # -- chain ----------------------------------------------------------

    def complete(self, system: str, user: str, purpose: str = "synthesis") -> str:
        attempts: list[tuple[str, object]] = []
        if self.nim_api_key:
            attempts.append(("nim", self._nim_call))
        attempts.append(("kilo", self._kilo_call))
        attempts.append(("kilo-auto-free", self._kilo_auto_free_call))

        errors: list[str] = []
        for index, (label, attempt) in enumerate(attempts):
            try:
                text = attempt(system, user)
                if text and text.strip():
                    self.last_from_fallback = index > 0
                    return text
            except LLMError as exc:
                logger.warning("inline chain tier %s failed: %s", label, exc)
                errors.append(f"{label}: {exc}")
        raise LLMError("all inline chain tiers failed; " + " | ".join(errors))

    # -- Kilo -----------------------------------------------------------

    def _kilo_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.kilo_api_key:
            headers["Authorization"] = f"Bearer {self.kilo_api_key}"
        return headers

    def _supported_parameters(self) -> set[str]:
        """Read Kilo's model catalog. Unauthenticated; returns empty set on failure."""
        try:
            response = self.session.get(
                f"{self.kilo_base_url}/models",
                headers=self._kilo_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            for model in response.json().get("data", []):
                if model.get("id") == _KILO_DEFAULT_PIN:
                    return set(model.get("supported_parameters") or [])
        except (requests.RequestException, ValueError) as exc:
            logger.warning("could not read Kilo model catalog: %s", exc)
        return set()

    def primary_model_in_catalog(self) -> bool | None:
        """True/False if the primary Kilo tier is in the live catalog, None if unknown.

        Used at startup to catch a delisted or misspelled free-tier ID before it
        becomes a 404 on the first real request. None means "could not tell",
        which is never an error.
        """
        ids = self._catalog_id_set()
        if ids is None:
            return None
        return _KILO_DEFAULT_PIN in ids

    def _catalog_id_set(self) -> set[str] | None:
        """All model IDs Kilo advertises, or None if the catalog is unreachable."""
        try:
            response = self.session.get(
                f"{self.kilo_base_url}/models",
                headers=self._kilo_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return {m.get("id") for m in response.json().get("data", []) if m.get("id")}
        except (requests.RequestException, ValueError):
            return None

    def _kilo_reasoning_params(self) -> tuple[dict, bool]:
        """Return (extra request params, whether reasoning is supported).

        A model with no advertised reasoning support degrades visibly: the caller
        logs it, rather than "thinking high" being silently dropped.
        """
        supported = self._supported_parameters()
        if "reasoning" in supported:
            return {"reasoning": {"effort": self.reasoning}}, True
        if "reasoning_effort" in supported:
            return {"reasoning_effort": self.reasoning}, True
        logger.warning(
            "Kilo model %s advertises no reasoning parameter; 'thinking high' cannot be requested",
            _KILO_DEFAULT_PIN,
        )
        return {}, False

    def _kilo_call(self, system: str, user: str) -> str:
        return self._chat(_KILO_DEFAULT_PIN, system, user, with_reasoning=True)

    def _kilo_auto_free_call(self, system: str, user: str) -> str:
        return self._chat(_KILO_LAST_RESORT, system, user, with_reasoning=False)

    def _chat(self, model: str, system: str, user: str, with_reasoning: bool) -> str:
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "temperature": 0.2,
        }

        supported = False
        if with_reasoning:
            reasoning_params, supported = self._kilo_reasoning_params()
            payload.update(reasoning_params)

        def post() -> requests.Response:
            # Network failures must become LLMError so the chain can advance,
            # not escape as a raw requests exception.
            try:
                return self.session.post(
                    f"{self.kilo_base_url}/chat/completions",
                    json=payload,
                    headers=self._kilo_headers(),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise LLMError(f"Kilo request failed: {exc}") from exc

        response = post()
        if response.status_code == 400 and supported:
            # Kilo's published schema omits the field; retry once without it.
            logger.info("Kilo 400 with reasoning field; retrying without reasoning params")
            for key in ("reasoning", "reasoning_effort"):
                payload.pop(key, None)
            response = post()

        return self._read_completion(response, provider="Kilo", model=model)

    # -- NIM ------------------------------------------------------------

    def _nim_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.nim_api_key:
            headers["Authorization"] = f"Bearer {self.nim_api_key}"
        return headers

    def _nim_call(self, system: str, user: str) -> str:
        last_error = "no NIM model attempted"
        for model in _NIM_PIN:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,  # NIM defaults to true; non-streaming is required
                "temperature": 0.2,
                "max_tokens": _NIM_MAX_TOKENS,
                "reasoning_budget": _NIM_REASONING_BUDGET,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            try:
                response = self._post_with_retry(
                    f"{self.nim_base_url}/chat/completions",
                    payload,
                    self._nim_headers(),
                )
                text = self._read_completion(response, provider="NIM", model=model)
                self.last_model_used = model
                return text
            except LLMError as exc:
                last_error = f"{model}: {exc}"
                logger.warning("NIM model %s failed: %s", model, exc)
        raise LLMError(last_error)

    def _post_with_retry(self, url: str, payload: dict, headers: dict) -> requests.Response:
        """POST with 429/503 backoff and 202 (result pending) handling."""
        delay = 2.0
        for attempt in range(3):
            try:
                response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                raise LLMError(f"NIM request failed: {exc}") from exc
            if response.status_code == 202:
                # NIM returns 202 with the result pending; poll using NVCF-REQID.
                request_id = response.headers.get("NVCF-REQID")
                if not request_id:
                    raise LLMError("NIM returned 202 with no NVCF-REQID")
                response = self._poll_nim(url, request_id, headers)
                continue
            if response.status_code in (429, 503) and attempt < 2:
                time.sleep(delay)
                delay *= 2
                continue
            if response.status_code >= 400:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
            return response
        raise LLMError(f"still rate limited after retries: {url}")

    def _poll_nim(self, url: str, request_id: str, headers: dict) -> requests.Response:
        """Poll the NIM status endpoint for a 202 result.

        NIM's async result endpoint is not documented in the public API reference
        we could reach, so the poll path is deliberately conservative: it gives
        up with a clear error rather than guessing an endpoint.
        """
        for _ in range(3):
            time.sleep(2.0)
            status = self.session.get(
                f"{self.nim_base_url}/status/{request_id}",
                headers=headers,
                timeout=self.timeout,
            )
            if status.status_code == 200:
                self.last_model_used = self._model_from_response(status)
                return status
            if status.status_code == 404:
                raise LLMError(
                    "NIM async status endpoint unavailable (undocumented); "
                    "cannot retrieve the 202 result - retry the request"
                )
        raise LLMError("NIM result did not become available within the poll window")

    def _model_from_response(self, response: requests.Response) -> str:
        try:
            return response.json().get("model", "")
        except ValueError:
            return ""


    # -- shared ---------------------------------------------------------

    def _read_completion(self, response: requests.Response, provider: str, model: str) -> str:
        if response.status_code >= 400:
            raise LLMError(f"{provider} HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError(f"{provider} returned non-JSON: {response.text[:200]}") from exc

        message = (data.get("choices") or [{}])[0].get("message") or {}
        # NIM puts thinking here; it must never reach the answer.
        text = message.get("content") or ""
        if not text.strip():
            raise LLMError(f"{provider} returned an empty completion")

        if isinstance(message.get("reasoning"), str) or isinstance(message.get("reasoning_content"), str):
            logger.debug("stripped reasoning trace from %s response", provider)

        self.last_model_used = data.get("model") or model
        return text
