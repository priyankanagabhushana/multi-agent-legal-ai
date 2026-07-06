"""OpenCaseLaw REST API client with caching, rate limiting, and error handling.

Base URL: https://mcp.opencaselaw.ch
Free, no API key, CORS enabled.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import httpx

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(method: str, url: str, params: dict) -> str:
    raw = f"{method}:{url}:{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()


class OpenCaseLawClient:
    """Client for the OpenCaseLaw REST API (29 endpoints).

    Rate limit: ≤ 5 requests/second. Caching enabled by default.
    """

    BASE = "https://mcp.opencaselaw.ch"

    def __init__(self, cache: bool = True, rate_limit: float = 0.25):
        self._client = httpx.Client(timeout=30.0)
        self._cache = cache
        self._rate_limit = rate_limit
        self._last_request = 0.0

    def _rate_limit_wait(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        params = params or {}
        url = f"{self.BASE}{path}"

        if self._cache:
            key = _cache_key("GET", url, params)
            cache_file = CACHE_DIR / f"{key}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text())

        self._rate_limit_wait()
        resp = self._client.get(url, params=params)
        self._last_request = time.monotonic()
        resp.raise_for_status()
        data = resp.json()

        if self._cache:
            cache_file = CACHE_DIR / f"{_cache_key('GET', url, params)}.json"
            cache_file.write_text(json.dumps(data, ensure_ascii=False))

        return data

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.BASE}{path}"
        self._rate_limit_wait()
        resp = self._client.post(url, json=body)
        self._last_request = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    # ── Decisions ──────────────────────────────────────────────────────────

    def search_decisions(
        self,
        q: str,
        limit: int = 10,
        offset: int = 0,
        language: str = "de",
        court: Optional[str] = None,
        canton: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> dict:
        """Search decisions by query. Returns paginated results."""
        params = {"q": q, "limit": limit, "offset": offset, "language": language}
        if court:
            params["court"] = court
        if canton:
            params["canton"] = canton
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        return self._get("/api/decisions", params)

    def get_decision(self, decision_id: str) -> dict:
        """Fetch a single decision with full text."""
        return self._get(f"/api/decisions/{decision_id}")

    def get_case_brief(self, decision_id: str) -> dict:
        """Structured brief: facts, reasoning, statutes, authority, related."""
        return self._get(f"/api/case-brief/{decision_id}")

    # ── Statutes ──────────────────────────────────────────────────────────

    def get_law_article(self, abbreviation: str, article: str) -> dict:
        """Look up a statute article (e.g., OR art. 271)."""
        return self._get(f"/api/laws/{abbreviation}", {"article": article})

    def search_laws(self, q: str, limit: int = 10, canton: Optional[str] = None) -> dict:
        """Full-text search over federal + cantonal articles."""
        params = {"q": q, "limit": limit}
        if canton:
            params["canton"] = canton
        return self._get("/api/laws/search", params)

    # ── Citation Graph ────────────────────────────────────────────────────

    def get_citations(self, decision_id: str) -> dict:
        """Both directions: what this cites + what cites it."""
        return self._get(f"/api/citations/{decision_id}")

    def get_leading_cases(
        self, statute: Optional[str] = None, topic: Optional[str] = None
    ) -> dict:
        """Authority-ranked decisions for a statute or topic."""
        params = {}
        if statute:
            params["statute"] = statute
        if topic:
            params["topic"] = topic
        return self._get("/api/leading-cases", params)

    # ── Doctrine & Commentary ─────────────────────────────────────────────

    def get_doctrine(
        self, abbreviation: str, article: Optional[str] = None
    ) -> dict:
        """Statute text + ranked BGEs + doctrine timeline + commentary excerpt."""
        params = {}
        if article:
            params["article"] = article
        return self._get(f"/api/doctrine/{abbreviation}", params)

    # ── Verification ──────────────────────────────────────────────────────

    def attest(self, text: str) -> dict:
        """Audit a draft response — verifies BGE/Art./quoted strings against corpus."""
        return self._post("/api/attest", {"text": text})

    # ── Metadata ──────────────────────────────────────────────────────────

    def get_courts(self) -> list[dict]:
        """List all 121 courts."""
        return self._get("/api/courts")

    def get_statistics(self) -> dict:
        """Aggregate stats — corpus, citation graph, statutes, languages."""
        return self._get("/api/statistics")

    def close(self):
        self._client.close()
