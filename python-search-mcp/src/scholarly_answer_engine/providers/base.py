"""Provider interface and shared HTTP plumbing."""

from __future__ import annotations

import abc
from typing import Any

import requests

from ..models import Work
from ..throttle import ProviderThrottle

USER_AGENT = "scholarly-answer-engine/0.1 (literature search; mailto in request)"


class ProviderError(Exception):
    """A provider failed in a way the caller should see."""


class BaseProvider(abc.ABC):
    """Every provider is isolated: a failure here degrades corroboration only."""

    name: str = "base"

    def __init__(self, session: requests.Session, throttle: ProviderThrottle, timeout: float) -> None:
        self.session = session
        self.throttle = throttle
        self.timeout = timeout

    @abc.abstractmethod
    def search(self, query: str, limit: int) -> list[Work]:
        """Return up to ``limit`` normalised works, best-first."""

    def build_query(self, question: str) -> str:
        """Adapt a natural-language question to this provider's query language.

        The default passes the question through unchanged, which is correct for
        providers that do their own relevance ranking over free text. Providers
        with a stricter query language (arXiv's boolean fields) override this.
        """
        return question

    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        self.throttle.wait(self.name)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            code = getattr(exc.response, "status_code", None)
            # 429 is pacing; anything else is a provider-level failure.
            raise ProviderError(f"{self.name} HTTP {code}: {exc}") from exc
        except (requests.RequestException, ValueError) as exc:
            raise ProviderError(f"{self.name} request failed: {exc}") from exc

    def _get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        self.throttle.wait(self.name)
        try:
            response = self.session.get(url, params=params or {}, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.HTTPError as exc:
            code = getattr(exc.response, "status_code", None)
            raise ProviderError(f"{self.name} HTTP {code}: {exc}") from exc
        except requests.RequestException as exc:
            raise ProviderError(f"{self.name} request failed: {exc}") from exc

    def _record(self, work: Work, rank: int) -> Work:
        work.add_provider(self.name, rank)
        return work


def make_session(contact_email: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"{USER_AGENT} (mailto:{contact_email})",
            "Accept": "application/json, application/xml, text/xml, */*",
        }
    )
    return session


def clean_text(value: str | None, limit: int | None = None) -> str | None:
    """Collapse whitespace and optionally truncate."""
    if not value:
        return None
    text = " ".join(value.split())
    if limit and len(text) > limit:
        return text[:limit]
    return text or None
