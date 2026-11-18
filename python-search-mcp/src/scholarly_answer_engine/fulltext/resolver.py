"""Open-access full-text resolution and PDF text extraction.

Grounding an answer in what a paper actually *says* is the whole point of this
engine: an abstract never states the sample size, the ablation, or which
baseline was compared, which is exactly where an abstract-only agent invents
evidence.

Open-access sources, in preference order:
  1. Europe PMC fullTextXML (OA subset) - XML, cleanest to parse
  2. arXiv PDF - parsed locally
  3. Unpaywall-resolved OA PDF - parsed locally
Anything without a legal open copy stays at ``grounding: abstract``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import requests

from ..config import Config
from ..models import Work
from ..providers.europepmc import EuropePmcProvider
from ..providers.unpaywall import UnpaywallProvider
from .pdftext import extract_pdf_text

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF"


class FullTextResolver:
    def __init__(self, config: Config, europepmc: EuropePmcProvider, unpaywall: UnpaywallProvider) -> None:
        self.config = config
        self.europepmc = europepmc
        self.unpaywall = unpaywall
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "scholarly-answer-engine/0.1 (OA full text)"})

    def resolve(self, works: list[Work]) -> list[Work]:
        """Fill ``full_text`` for as many works as legally possible, in parallel."""
        if not self.config.fulltext_enabled or not works:
            return works
        targets = works[: self.config.max_sources]
        with ThreadPoolExecutor(max_workers=self.config.fetch_concurrency) as pool:
            list(pool.map(self._resolve_one, targets))
        return works

    def _resolve_one(self, work: Work) -> None:
        """Try each open-access source in order; never raises."""
        for attempt in (self._from_europe_pmc, self._from_arxiv, self._from_unpaywall):
            try:
                text = attempt(work)
            except Exception as exc:  # noqa: BLE001 - a failed OA fetch must not abort the answer
                logger.debug("full-text attempt failed for %r: %s", work.title[:60], exc)
                text = None
            if text and text.strip():
                work.full_text = text[: self.config.fulltext_chars]
                work.grounding = "fulltext"
                return
        work.full_text = None
        work.grounding = "abstract"

    def _from_europe_pmc(self, work: Work) -> str | None:
        if not work.pmcid or self.europepmc is None:
            return None
        return self.europepmc.full_text_xml(work.pmcid, self.config.fulltext_chars)

    def _from_arxiv(self, work: Work) -> str | None:
        if not work.arxiv_id or not work.open_access:
            return None
        url = f"https://arxiv.org/pdf/{work.arxiv_id}.pdf"
        return self._fetch_pdf_text(url)

    def _from_unpaywall(self, work: Work) -> str | None:
        if not work.doi:
            return None
        if work.oa_url and not work.pmcid and not work.arxiv_id:
            return self._fetch_pdf_text(work.oa_url)
        url = self.unpaywall.oa_pdf_url(work.doi)
        if not url:
            return None
        return self._fetch_pdf_text(url)

    def _fetch_pdf_text(self, url: str) -> str | None:
        """Download a PDF with bounded time and byte budget, then extract text."""
        try:
            response = self._session.get(
                url,
                timeout=self.config.fulltext_timeout,
                stream=True,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("pdf fetch failed for %s: %s", url[:80], exc)
            return None

        try:
            body = self._read_bounded(response)
        except Exception as exc:  # noqa: BLE001 - streaming reads can fail arbitrarily
            logger.debug("pdf read failed for %s: %s", url[:80], exc)
            return None
        if not body:
            return None
        if _PDF_MAGIC not in body[:2048]:
            logger.debug("response for %s is not a PDF", url[:80])
            return None
        return extract_pdf_text(body, self.config.fulltext_chars)

    def _read_bounded(self, response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= self.config.fulltext_max_bytes:
                break
        return b"".join(chunks)
