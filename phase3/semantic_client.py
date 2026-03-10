from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

import requests

from config import SEMANTIC_DEFAULT_LIMIT, SEMANTIC_SCHOLAR_API, SEMANTIC_SCHOLAR_TIMEOUT

# Semantic Scholar enforces 1 req/s regardless of authentication tier.
_MIN_INTERVAL = 1.0  # seconds between requests


class SemanticScholarClient:
    def __init__(self, logger: logging.Logger):
        self.log = logger
        self.disabled = str(os.getenv("DISABLE_SEMANTIC_API", "")).strip().lower() in {"1", "true", "yes"}
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self._headers = {"x-api-key": api_key} if api_key else {}
        self._last_request_at: float = 0.0  # monotonic timestamp of last request
        if api_key:
            logger.info("Semantic Scholar: authenticated (key ...%s)", api_key[-6:])
        else:
            logger.warning("Semantic Scholar: no API key — unauthenticated (1 req/s limit)")

    def _wait_for_rate_limit(self) -> None:
        """Block until at least _MIN_INTERVAL seconds have passed since the last request."""
        elapsed = time.monotonic() - self._last_request_at
        wait = _MIN_INTERVAL - elapsed
        if wait > 0:
            self.log.debug("Semantic Scholar rate-limit: sleeping %.2fs", wait)
            time.sleep(wait)

    def search(self, query: str, limit: int = SEMANTIC_DEFAULT_LIMIT) -> List[Dict[str, Any]]:
        if self.disabled:
            return []
        query = (query or "").strip()
        if not query:
            return []

        params = {
            "query": query,
            "limit": int(limit),
            "fields": "title,abstract,year,venue,authors,url,externalIds,citationCount",
        }
        try:
            self._wait_for_rate_limit()
            self._last_request_at = time.monotonic()
            resp = requests.get(
                SEMANTIC_SCHOLAR_API,
                params=params,
                headers=self._headers,
                timeout=SEMANTIC_SCHOLAR_TIMEOUT,
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                self.log.warning("Semantic Scholar 429 rate-limited — sleeping %ds", retry_after)
                time.sleep(retry_after)
                self._last_request_at = time.monotonic()
                resp = requests.get(
                    SEMANTIC_SCHOLAR_API,
                    params=params,
                    headers=self._headers,
                    timeout=SEMANTIC_SCHOLAR_TIMEOUT,
                )
            if resp.status_code != 200:
                self.log.warning("Semantic Scholar status=%s query=%s", resp.status_code, query[:80])
                return []
            data = resp.json()
            return data.get("data", []) if isinstance(data, dict) else []
        except Exception as exc:
            self.log.warning("Semantic Scholar error for query=%s err=%s", query[:80], exc)
            return []
