# Scholarly Answer Engine MCP

A Perplexity-like answer engine over **academic literature**. Ask a research
question; get a cited synthesis grounded in open-access full text where it
legally exists, with per-source provenance and cross-provider corroboration.

Built entirely on **free, keyless scholarly APIs**. No API keys are required for
retrieval, and LLM synthesis goes through a free-tier router.

Deliberately absent: any browser, any HTML scraping, any CAPTCHA bypass.

## Scope

This covers **academic literature only**. It searches Crossref, arXiv and
Europe PMC. It cannot answer questions about news, products, current events,
people, or anything outside scholarly publications - it will say so rather than
guess.

## Retrieval

| Provider | Key | Role |
|---|---|---|
| Crossref | none (mailto → polite pool) | authoritative DOI metadata + citation counts |
| arXiv | none | Atom XML; keyword-built boolean queries |
| Europe PMC | none | 40M+ works; **open-access full-text XML** |
| Unpaywall | real email only | resolve a lawful OA PDF per DOI |
| OpenAlex | none | **DOI singleton lookups only** (`search=` is metered since 2026-02) |
| Semantic Scholar | none | optional, off by default |

Providers are queried in parallel and fused with **RRF** (`k=60`), because their
relevance scales are opaque and incomparable. Corroboration falls out of the
fusion: the RRF sum has one term per provider list containing a document, so the
term count *is* the corroboration count. RefLoRA found by both Crossref and
arXiv ranks above a paper found once, and is flagged `corroborated`.

**Conditional query expansion.** arXiv's `all:"<question>"` matches a verbatim
phrase, so a natural-language question must be reduced to salient terms first -
this is done per-provider. Full sub-query expansion (via the LLM) fires only when
raw retrieval was weak: too few works, too few providers answering, or nothing
corroborated across providers. The trigger is computed from results already in
hand, so no LLM call is spent deciding.

**Open-access full text** is resolved from Europe PMC XML, arXiv PDFs parsed
locally, and Unpaywall-resolved OA PDFs. Papers without a legal open copy stay at
`grounding: abstract` and are marked as such, so the depth of evidence behind
each claim is visible.

## Tools

- **`perplexity-search`** — `question`, optional `max_sources`, `follow_up`.
  Returns `answer` with inline `[n]` citations, `citations[]`, `sources[]`,
  `corroboration`, `model_used`, `from_fallback`, `cached`, `cache_age`.
- **`papers`** — raw normalised retrieval with provenance and no synthesis.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env      # all defaults work without edits
python -m pytest tests   # 87 tests; 4 need RUN_LIVE_TESTS=1 for live model checks
```

Add to an MCP client:

```json
{
  "mcpServers": {
    "scholarly-answer-engine": {
      "command": "python",
      "args": ["-m", "scholarly_answer_engine"],
      "env": { "SEARCH_CONTACT_EMAIL": "you@example.com" }
    }
  }
}
```

To use the router from this machine, run it on a bindable port and point the
engine at it:

```bash
# in scholarly-answer-router/
npm install && npm run build
ROUTER_PORT=3000 node dist/index.js &

# in this project
export ROUTER_URL=http://127.0.0.1:3000
python -m scholarly_answer_engine
```

## Configuration

The search path is keyless out of the box. Synthesis uses the **router**; start
it with `ROUTER_PORT=3000 node dist/index.js` in `scholarly-answer-router`. The
engine defaults to `ROUTER_URL=http://127.0.0.1:8787` — set `ROUTER_URL` to match
the port the router bound. With no router running, set `SEARCH_LLM_TRANSPORT=inline`
to call providers directly from Python.

## Running

```bash
python -m scholarly_answer_engine            # stdio MCP server
python -m pytest tests                      # 84 tests (3 skip unless RUN_LIVE_TESTS=1)
npx @modelcontextprotocol/inspector python -m scholarly_answer_engine
```

## LLM tiers (verified)

Synthesis goes through the **Scholarly Answer Router**
(`C:\Users\Simon\source\repos\scholarly-answer-router`), which owns the ordered
tier walk, keys, circuits and substitution.

The default chain was corrected twice by live checks, and the corrections matter:

| Tier | Provider | Cost | Notes |
|---|---|---|---|
| `stepfun-ai/step-5-preview` | NVIDIA NIM | **free endpoint**, $0/$0 per 1M | 600B MoE, 1M ctx, reasoning + tools |
| `nvidia/nemotron-3.5-super-vl:free` | Kilo | **free** | reasoning, tools |
| `kilo-auto/free` | Kilo | **free** | anonymous, no key |

**Step 5 Preview is free on NIM, not on Kilo.** Kilo also lists a Step 5 as
`stepfun/step-5-preview` at ~$1/M in — a *different ID for the same model*. The
publisher prefix is the discriminator: `stepfun-ai/*` on NIM, `stepfun/*` on
Kilo. An earlier revision of this project wrongly concluded no free StepFun model
existed, because it only checked Kilo.

Tier 1 needs a free `NVIDIA_API_KEY` from <https://build.nvidia.com/settings> (no
credit card). Without it the router skips straight to Kilo's free tiers, which
work anonymously — the end-to-end answer is then produced by Nemotron.

Reasoning is per-provider inside the router: Kilo's schema has no reasoning field
but the gateway honours one for models advertising it, while NIM uses
`chat_template_kwargs.enable_thinking` + `reasoning_budget`. "reasoning high" is
therefore genuinely requested, not assumed. Kilo's free models intermittently
return `content: null` with text in `reasoning` — a reasoning-only failure that
is reported distinctly and advances the chain rather than being mistaken for an
empty answer.

| Var | Default | Meaning |
|---|---|---|
| `SEARCH_CONTACT_EMAIL` | `research@example.invalid` | contact for Crossref/Unpaywall polite pools |
| `SEARCH_ENGINES` | `crossref,arxiv,europepmc` | providers to query |
| `SEARCH_MAX_SOURCES` | `5` | sources grounded into the answer |
| `PER_PROVIDER_LIMIT` | `15` | results fetched per provider |
| `RRF_K` | `60` | RRF constant |
| `ENABLE_EXPANSION` | `true` | conditional query expansion |
| `EXPANSION_MIN_WORKS` | `3` | expand if fewer works than this |
| `EXPANSION_MIN_PROVIDERS` | `2` | expand if fewer providers answered |
| `ENABLE_FULLTEXT` | `true` | resolve open-access full text |
| `FULLTEXT_CHARS` | `12000` | max characters of full text per work |
| `ENABLE_CACHE` | `true` | client-side response cache |
| `CACHE_TTL_SECONDS` | `900` | 15 min default |
| `ENABLE_OPENALEX` | `true` | citation-count enrichment (DOI singletons only) |
| `ARXIV_INTERVAL` | `3.0` | arXiv pacing (1 req / 3s) |
| `ENABLE_SEMANTIC_SCHOLAR` | `false` | optional keyless provider |
| `SEARCH_LLM_TRANSPORT` | `router` | `router` or `inline` |
| `ROUTER_URL` | `http://127.0.0.1:8787` | router base URL |
| `ROUTER_PIN` | see `.env.example` | ordered LLM tier list |
| `ROUTER_ALLOW_FALLBACK` | `true` | allow substitution if a tier is unusable |
| `SEARCH_LLM_REASONING` | `high` | requested reasoning depth |
| `KILO_API_KEY` | unset | optional; Kilo anonymous tier works without it |
| `NVIDIA_API_KEY` | unset | required only for the NIM tier of the inline chain |

## Architecture

```
server.py            MCP tools (perplexity-search, papers)
engine.py            orchestration: cache → retrieve → expand → ground → synthesise
providers/           isolated per-provider clients + normalization
retrieval/keywords.py   salient-term extraction per provider query language
retrieval/fusion.py     RRF scoring, corroboration, expansion trigger
retrieval/enrichment.py openalex DOI-singleton citation counts
fulltext/resolver.py    OA full-text resolution (Europe PMC XML / arXiv / Unpaywall)
fulltext/pdftext.py     bounded, failure-tolerant PDF text extraction
synth/                router client (primary) + inline chain (contingency)
synthesis.py          citation parsing and post-check
cache.py              client-side TTL cache, checked before retrieval
throttle.py           per-provider interval limiting with jitter
normalize.py          DOI / arXiv ID / title-fingerprint identity and merge
```

## Design decisions worth knowing

- **No scraping, no browser.** Scholarly APIs return structured, citable records;
  scraping SERPs produces snippets unfit for citation and carries IP-blocking and
  ToS risk.
- **OpenAlex metering.** Its `search=` endpoint has been metered since Feb 2026.
  The module therefore exposes *no search method at all*, so the metered surface
  is unreachable by construction rather than by convention.
- **Citation post-check.** Every `[n]` in the answer must map to a real source,
  or the request reports a warning. This catches unsupported references but
  *cannot* catch a fabricated method, which is why the synthesis prompt forbids
  asserting methods the supplied text does not contain.
- **Cache placement.** Client-side, before retrieval, so a hit avoids re-running
  engines and fetch. Short TTL (15 min) with `cache_age` reported, because a
  cached answer may cite sources that have since changed.
- **Failures never silently degrade.** A provider returning nothing is recorded
  in `sources[]`; a paper cited only from its abstract is labelled.

## Tests

62 tests, no network: identity normalisation and cross-provider merging, RRF
fusion and the expansion trigger, citation post-check, cache TTL behaviour,
provider parsing against fixtures, and bounded PDF extraction.
