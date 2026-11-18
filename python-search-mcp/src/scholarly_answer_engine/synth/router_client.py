"""The primary transport: a Free-Tier LLM Router.

The router owns the ordered tier list, keys, circuits and substitution, so this
client is deliberately thin - build an OpenAI-shaped request, send it, parse the
provenance out of the response body.

See plans/1794058305330-free-tier-llm-router.md for the server-side contract.
"""

from __future__ import annotations

import json
import logging

import requests

from .client import LLMClient, LLMError

logger = logging.getLogger(__name__)


class RouterClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        pin: list[str],
        allow_fallback: bool = True,
        deadline: float = 120.0,
        timeout: float = 180.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.pin = [p for p in pin if p]
        if not self.pin:
            raise ValueError("router pin list is empty; set ROUTER_PIN")
        self.allow_fallback = allow_fallback
        self.deadline = deadline
        self.timeout = timeout
        self.session = session or requests.Session()
        self.last_model_used = None
        self.last_from_fallback = False

    def describe(self) -> str:
        return f"router {self.base_url} pin={self.pin[0]} (+{max(0, len(self.pin) - 1)} tiers)"

    def complete(self, system: str, user: str, purpose: str = "synthesis") -> str:
        payload = {
            "model": self.pin[0],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "temperature": 0.2,
            "purpose": purpose,
        }
        headers = {
            "Content-Type": "application/json",
            "x-router-pin": ",".join(self.pin),
            "x-router-allow-fallback": "true" if self.allow_fallback else "false",
            "x-router-deadline": str(int(self.deadline)),
            "x-router-contract-version": "1",
        }
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"router unreachable at {self.base_url}: {exc}") from exc

        if response.status_code == 503:
            raise LLMError(f"router reports no usable tier (circuit open): {self._short(response)}")

        if response.status_code >= 400:
            raise LLMError(f"router HTTP {response.status_code}: {self._short(response)}")

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError(f"router returned non-JSON: {self._short(response)}") from exc

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"router response missing completion: {self._short(response)}") from exc

        provenance = data.get("router") or {}
        self.last_model_used = provenance.get("model_used") or data.get("model") or ""
        self.last_from_fallback = bool(provenance.get("from_fallback", False))
        if not isinstance(text, str) or not text.strip():
            raise LLMError("router returned an empty completion")
        return text

    @staticmethod
    def _short(response) -> str:
        try:
            return response.text[:300]
        except Exception:  # noqa: BLE001 - diagnostics only; never mask the real error
            return "<unreadable body>"
