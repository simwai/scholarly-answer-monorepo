"""LLM client interface shared by the router client and the inline contingency."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(Exception):
    """The synthesis call failed in a way the caller should surface."""


class LLMClient(ABC):
    """One method, so the engine never depends on the transport.

    ``RouterClient`` is the v1 default. ``InlineChainClient`` implements the same
    contract directly against providers, and exists as a contingency.
    """

    #: model that actually answered the last successful call
    last_model_used: str | None = None
    last_from_fallback: bool = False

    @abstractmethod
    def complete(self, system: str, user: str, purpose: str = "synthesis") -> str:
        """Return the model's text answer. Raises LLMError on failure."""

    @abstractmethod
    def describe(self) -> str:
        """Short human-readable description for logs and error messages."""
