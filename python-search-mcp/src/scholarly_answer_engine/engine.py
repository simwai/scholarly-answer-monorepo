"""Pipeline orchestration: cache -> retrieve -> expand (conditional) -> ground -> synthesise.

Implements the plan's data flow, including the conditional-expansion trigger and
the requirement that the original query's list is always fused back in alongside
any sub-query lists.
"""

from __future__ import annotations

import logging

from .cache import ResponseCache
from .config import Config
from .fulltext import FullTextResolver
from .models import SearchOutcome, Work
from .normalize import merge_works
from .providers import ProviderRegistry
from .retrieval import enrichment, expansion
from .retrieval.fusion import apply_priors, corroboration_summary, needs_expansion, score_rrf
from .supervisor import RouterSupervisor
from .synth.client import LLMClient, LLMError
from .synth.inline_client import InlineChainClient
from .synth.prompts import build_synthesis_user
from .synth.router_client import RouterClient
from .synthesis import add_citation_numbers, post_check
from .synthesis.cross_validation import augment_with_corroboration

logger = logging.getLogger(__name__)


class AnswerEngine:
    def __init__(self, config: Config) -> None:
        self.config = config.sanitized()
        self.registry = ProviderRegistry(self.config)
        self.fulltext = FullTextResolver(self.config, self.registry.europepmc, self.registry.unpaywall)
        self.cache = ResponseCache(self.config)
        self.router: RouterSupervisor | None = None
        self.llm = self._build_llm()

    def _build_llm(self) -> LLMClient:
        if self.config.llm_transport == "inline":
            logger.info("using inline chain client (contingency transport)")
            return InlineChainClient(
                kilo_base_url=self.config.kilo_base_url,
                kilo_api_key=self.config.kilo_api_key,
                nim_base_url=self.config.nim_base_url,
                nim_api_key=self.config.nim_api_key,
                reasoning=self.config.reasoning,
            )
        logger.info("using router client: %s", self.config.router_url)
        self.router = self._ensure_router()
        try:
            return RouterClient(
                base_url=self.config.router_url,
                pin=self.config.router_pin,
                allow_fallback=self.config.router_allow_fallback,
                deadline=self.config.router_deadline,
            )
        except ValueError as exc:
            raise LLMError(str(exc)) from exc

    def _ensure_router(self) -> RouterSupervisor | None:
        """Start the router if needed, so one command is enough to run the tool.

        A failure is not fatal: the ``papers`` tool needs no LLM, so the server
        still starts and reports the problem loudly.
        """
        if not self.config.router_autostart:
            return None
        supervisor = RouterSupervisor(
            router_url=self.config.router_url,
            router_dir=self.config.router_dir or None,
            port=self.config.router_port or None,
        )
        if not supervisor.ensure():
            logger.warning(
                "router unavailable at %s - synthesis will fail, but retrieval-only "
                "tools still work. Start it manually or set ROUTER_AUTOSTART=false.",
                self.config.router_url,
            )
        return supervisor

    # -- public API -----------------------------------------------------

    def search(self, question: str, max_sources: int | None = None, follow_up: str | None = None) -> SearchOutcome:
        limit = max_sources or self.config.max_sources
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        cache_key = self.cache.key(question, limit, follow_up)
        hit = self.cache.get(cache_key)
        if hit is not None:
            logger.info("cache hit for question (age=%ss)", hit.get("cache_age"))
            return self._outcome_from_cache(hit)

        outcomes = self._retrieve(question, follow_up, max_sources=limit)
        works = outcomes[0]
        sub_queries = outcomes[1]
        failed_providers = outcomes[2]

        self.fulltext.resolve(works)
        selected = works[:limit]

        answer = self._synthesise(question, selected, follow_up)
        answer = add_citation_numbers(answer, selected)
        claims, warnings = post_check(answer, selected)

        # Build citation objects before cross-validation so we can map
        # claim citations back to source works.
        citation_list = [w.to_citation(i) for i, w in enumerate(selected, start=1)]

        # Claim-level cross-validation: run each answer claim against the
        # underlying full-text sources and count distinct engines that agree.
        # This replaces the old snippet-level count with actual textual
        # corroboration.
        answer, claims, summary = augment_with_corroboration(
            answer, citation_list, selected
        )

        for provider in failed_providers:
            warnings.append(f"provider returned no results: {provider}")

        # Merge corroboration summary into the output dict.
        corroboration_summary = {
            "total_claims": summary.total_claims,
            "corroborated_claims": summary.corroborated_claims,
            "single_source_claims": summary.single_source_claims,
            "corroborated_percentage": summary.corroboration_percentage,
            "contradicted_claims": summary.contradicted_claims,
            "engine_agreement": summary.engine_agreement,
            "providers_active": summary.providers_active,
        }

        outcome = SearchOutcome(
            answer=answer,
            citations=citation_list,
            sources=self._source_payload(selected),
            corroboration=corroboration_summary,
            model_used=self.llm.last_model_used or "",
            from_fallback=self.llm.last_from_fallback,
            cached=False,
            cache_age=None,
            expanded=bool(sub_queries),
            sub_queries=sub_queries,
            warnings=warnings,
        )
        self.cache.put(cache_key, outcome.to_dict())
        return outcome

    def papers(self, question: str, limit: int | None = None) -> dict:
        """Raw retrieval tool: normalised records with provenance, no synthesis."""
        limit = limit or self.config.per_provider_limit
        works, _, _ = self._retrieve(question.strip(), None)
        return {
            "works": [self._work_payload(w) for w in works[:limit]],
            "corroboration": corroboration_summary(works[:limit]),
        }

    def close(self) -> None:
        """Stop a router this process started, and release the provider session.

        Only a router the supervisor spawned is stopped; one the user launched by
        hand is left running.
        """
        if self.router is not None:
            self.router.stop()
        self.registry.close()

    # -- stages ---------------------------------------------------------

    def _retrieve(self, question: str, follow_up: str | None, max_sources: int | None = None) -> tuple[list[Work], list[str], list[str]]:
        """Run raw retrieval, expand if weak, fuse, enrich, rank."""
        raw, failed = self.registry.search_all(question, self.config.per_provider_limit)
        merged = merge_works(raw)
        scored = score_rrf(merged, k=self.config.rrf_k)

        sub_queries: list[str] = []
        if needs_expansion(scored, self.config, max_sources):
            logger.info("raw retrieval weak (works=%d) - expanding", len(scored))
            sub_queries = expansion.expand_question(
                self.llm, question, self.config.expansion_max_queries
            )
            rounds = [raw]
            for sub in sub_queries:
                sub_raw, sub_failed = self.registry.search_all(sub, self.config.per_provider_limit)
                failed.extend(sub_failed)
                if sub_raw:
                    rounds.append(sub_raw)
            if len(rounds) > 1:
                # Fuse original + expanded lists. Re-merging from every round
                # keeps providers/ranks from all of them, so a paper returned by
                # both the original and an expanded query keeps both terms.
                scored = score_rrf(merge_works([w for round_works in rounds for w in round_works]), k=self.config.rrf_k)
                scored = self._rerank_with_corroboration(scored)
        scored = apply_priors(scored)

        if self.config.openalex_enabled and self.registry.openalex:
            enrichment.enrich_citations(scored, self.registry.openalex, self.config.openalex_enrich_top)
            scored = apply_priors(scored)

        return scored, sub_queries, failed

    @staticmethod
    def _rerank_with_corroboration(works: list[Work]) -> list[Work]:
        return sorted(works, key=lambda w: (-w.corroboration, -w.rrf_score))

    def _synthesise(self, question: str, works: list[Work], follow_up: str | None) -> str:
        if not works:
            raise LLMError(
                "no scholarly sources were found for this question; this engine "
                "covers academic literature only and cannot answer non-academic questions"
            )
        user_prompt = build_synthesis_user(question, works)
        if follow_up:
            user_prompt = f"{user_prompt}\n\nConversation context (for continuity only):\n{follow_up}"
        try:
            return self.llm.complete(
                system="You are a precise literature-synthesis assistant.",
                user=user_prompt,
                purpose="synthesis",
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"synthesis failed: {exc}") from exc

    # -- payloads -------------------------------------------------------

    @staticmethod
    def _work_payload(work: Work) -> dict:
        return {
            "title": work.title,
            "authors": work.authors,
            "year": work.year,
            "venue": work.venue,
            "doi": work.doi,
            "arxiv_id": work.arxiv_id,
            "url": work.url,
            "cited_by": work.cited_by,
            "open_access": work.open_access,
            "grounding": work.grounding,
            "providers": work.providers,
            "corroboration": work.corroboration,
            "rrf_score": round(work.rrf_score, 6),
            "has_full_text": bool(work.full_text),
            "abstract": work.abstract,
        }

    def _source_payload(self, works: list[Work]) -> list[dict]:
        return [self._work_payload(w) for w in works]

    @staticmethod
    def _outcome_from_cache(hit: dict) -> SearchOutcome:
        from .models import Citation

        citations = [Citation(**c) for c in hit.get("citations", [])]
        return SearchOutcome(
            answer=hit.get("answer", ""),
            citations=citations,
            sources=hit.get("sources", []),
            corroboration=hit.get("corroboration", {}),
            model_used=hit.get("model_used", ""),
            from_fallback=hit.get("from_fallback", False),
            cached=True,
            cache_age=hit.get("cache_age"),
            expanded=hit.get("expanded", False),
            sub_queries=hit.get("sub_queries", []),
            warnings=hit.get("warnings", []),
        )
