"""Unpaywall resolves a legal open-access copy for a DOI.

Requires only a real email address (used as a key substitute); ~100K calls/day.
This is a *resolver*, not a search provider, so it deliberately does not inherit
from ``BaseProvider`` - it has no ``search``.
"""

from __future__ import annotations

from typing import Any

import requests

from ..throttle import ProviderThrottle

USER_AGENT = "scholarly-answer-engine/0.1 (OA full text; mailto in request)"


class UnpaywallResolver:
    name = "unpaywall"
    endpoint = "https://api.unpaywall.org/v2"

    def __init__(
        self,
        session: requests.Session | None = None,
        throttle: ProviderThrottle | None = None,
        timeout: float = 25.0,
        contact_email: str = "research@example.invalid",
    ) -> None:
        self.session = session or requests.Session()
        self.throttle = throttle or ProviderThrottle(0.0)
        self.timeout = timeout
        self.contact_email = contact_email

    def oa_pdf_url(self, doi: str) -> str | None:
        """Return the best lawful OA PDF URL for ``doi``, or None.

        A miss is normal (most DOIs have no open copy), so this degrades quietly
        rather than raising - the caller falls back to the abstract.
        """
        self.throttle.wait(self.name)
        try:
            response = self.session.get(
                f"{self.endpoint}/{doi}",
                params={"email": self.contact_email},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            from .base import ProviderError

            raise ProviderError(f"unpaywall request failed: {exc}") from exc
        return self._extract_pdf(data)

    @staticmethod
    def _extract_pdf(data: dict[str, Any]) -> str | None:
        location = data.get("best_oa_location") or {}
        for key in ("url_for_pdf", "url"):
            url = location.get(key)
            if isinstance(url, str) and url.startswith("http"):
                return url
        for loc in data.get("oa_locations") or []:
            url = loc.get("url_for_pdf") or loc.get("url")
            if isinstance(url, str) and url.startswith("http"):
                return url
        return None


# Retained name for callers importing the old symbol.
UnpaywallProvider = UnpaywallResolver
