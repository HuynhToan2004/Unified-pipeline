from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import requests

from config import SEMANTIC_DEFAULT_LIMIT, SEMANTIC_SCHOLAR_API, SEMANTIC_SCHOLAR_TIMEOUT


class SemanticScholarClient:
    def __init__(self, logger: logging.Logger):
        self.log = logger
        self.disabled = str(os.getenv("DISABLE_SEMANTIC_API", "")).strip().lower() in {"1", "true", "yes"}

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
            resp = requests.get(
                SEMANTIC_SCHOLAR_API,
                params=params,
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
