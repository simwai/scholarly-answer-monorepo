"""Perplexity-like answer engine over academic literature.

Retrieval uses only free, keyless scholarly APIs (Crossref, arXiv, Europe PMC),
resolves open-access full text where it legally exists, and synthesises a cited
answer through an LLM router.

Deliberately absent: any browser, any HTML scraping, any CAPTCHA bypass.
"""

__version__ = "0.1.0"
