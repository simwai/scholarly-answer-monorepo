"""MCP server surface: two tools over the same pipeline.

``perplexity-search`` answers a question with an inline-cited synthesis.
``papers`` returns the raw normalised retrieval, no synthesis - useful for
inspecting what the providers actually returned.
"""

from __future__ import annotations

import logging
import signal

from mcp.server.fastmcp import FastMCP

from .config import Config
from .engine import AnswerEngine
from .synth.client import LLMError
from .synth.inline_client import _KILO_DEFAULT_PIN, InlineChainClient

logger = logging.getLogger(__name__)

TOOL_DESCRIPTION = """Answer a question from the academic literature, Perplexity-style.

Returns a cited synthesis grounded in open-access full text where it legally
exists, with per-source provenance and cross-provider corroboration.

This tool covers ACADEMIC LITERATURE ONLY. It searches Crossref, arXiv and
Europe PMC. It cannot answer questions about news, products, current events,
people or anything outside scholarly publications - it will say so rather than
guess.

args:
    question (str): the research question.
    max_sources (int, optional): how many sources to ground the answer in.
    follow_up (str, optional): prior conversation turns for continuity.
"""

PAPERS_DESCRIPTION = """Return the raw literature retrieval for a question, with no synthesis.

Same providers as perplexity-search, plus RRF ranking, cross-provider
deduplication and citation counts. Useful for seeing what the search actually found.

args:
    question (str): the research question.
    limit (int, optional): number of works to return.
"""


def create_server(config: Config | None = None) -> tuple[FastMCP, AnswerEngine]:
    config = config or Config().sanitized()
    engine = AnswerEngine(config)

    mcp = FastMCP(
        "scholarly-answer-engine",
        instructions=(
            "Answers research questions from academic literature with inline [n] citations, "
            "grounded in open-access full text where available. Academic literature only."
        ),
    )

    @mcp.tool(name="perplexity-search", description=TOOL_DESCRIPTION)
    def perplexity_search(question: str, max_sources: int | None = None, follow_up: str | None = None) -> dict:
        """Answer a question with an inline-cited synthesis."""
        return engine.search(question, max_sources=max_sources, follow_up=follow_up).to_dict()

    @mcp.tool(name="papers", description=PAPERS_DESCRIPTION)
    def papers(question: str, limit: int | None = None) -> dict:
        """Return raw normalised retrieval with provenance, no synthesis."""
        return engine.papers(question, limit=limit)

    return mcp, engine


def _warn_on_unverified_pin(config: Config) -> None:
    """Check the configured Kilo tier against the live catalog; warn on mismatch.

    Free-tier model IDs disappear server-side without notice, and a wrong ID only
    surfaces as a 404 on the first real request, so this makes it visible at
    startup instead. Best-effort: a network failure must never block startup.

    Only meaningful for the inline transport - when the router serves the request,
    catalog verification is the router's job.
    """
    if config.llm_transport != "inline":
        return

    try:
        client = InlineChainClient(
            kilo_base_url=config.kilo_base_url,
            kilo_api_key=config.kilo_api_key,
            nim_base_url=config.nim_base_url,
            nim_api_key=config.nim_api_key,
            reasoning=config.reasoning,
        )
        found = client.primary_model_in_catalog()
    except Exception as exc:  # noqa: BLE001 - startup advisory only
        logger.warning("could not verify the inline LLM pin at startup: %s", exc)
        return

    if found is False:
        logger.warning(
            "inline tier %s is NOT in Kilo's catalog - the first request will 404; "
            "see README 'LLM tiers (verified)'",
            _KILO_DEFAULT_PIN,
        )
    elif found is True:
        logger.info("inline tier %s resolves in Kilo's catalog", _KILO_DEFAULT_PIN)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = Config().sanitized()
        _warn_on_unverified_pin(config)
        mcp, engine = create_server(config)
    except LLMError as exc:
        logger.error("startup failed: %s", exc)
        raise SystemExit(2) from exc
    except Exception as exc:  # pragma: no cover - defensive startup guard
        logger.error("startup failed unexpectedly: %s", exc)
        raise SystemExit(2) from exc

    def shutdown(*_args) -> None:
        engine.close()
        raise SystemExit(0)

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), shutdown)

    try:
        mcp.run()
    finally:
        # Stop a router this process spawned, but never one it merely connected to.
        if engine.router is not None and engine.router.process is not None:
            engine.router.stop()
