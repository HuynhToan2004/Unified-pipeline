#!/usr/bin/env python3
"""Quick test script for the Semantic Scholar API.

Usage:
    python scripts/test_semantic_scholar.py
    python scripts/test_semantic_scholar.py --query "graph neural network" --limit 5
    python scripts/test_semantic_scholar.py --query "attention mechanism transformer" --limit 3 --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

# ── Load .env from project root ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with env_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    print(f"[INFO] Loaded {env_file}")

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS   = "title,abstract,year,venue,authors,url,externalIds,citationCount"

DEFAULT_QUERIES = [
    "attention is all you need transformer",
    "graph neural network node classification",
    "large language model instruction tuning",
]


def make_headers() -> dict[str, str]:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        print(f"[INFO] Using API key: ...{key[-6:]}")
        return {"x-api-key": key}
    print("[WARN] No SEMANTIC_SCHOLAR_API_KEY set — unauthenticated (rate-limited to ~1 req/s)")
    return {}


def search(query: str, limit: int, headers: dict[str, str], timeout: int = 20) -> list[dict]:
    params = {"query": query, "limit": limit, "fields": FIELDS}
    resp = requests.get(API_URL, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", []) if isinstance(data, dict) else []


def print_results(papers: list[dict], verbose: bool) -> None:
    if not papers:
        print("  (no results)")
        return
    for i, p in enumerate(papers, 1):
        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
        if len(p.get("authors") or []) > 3:
            authors += " et al."
        print(f"  [{i}] {p.get('title', '(no title)')}")
        print(f"       Year: {p.get('year', '?')}  |  Citations: {p.get('citationCount', '?')}  |  Authors: {authors}")
        if verbose and p.get("abstract"):
            abstract = (p["abstract"] or "")[:200].replace("\n", " ")
            print(f"       Abstract: {abstract}...")
        print()


def run_tests(queries: list[str], limit: int, verbose: bool) -> bool:
    headers = make_headers()
    all_ok = True
    last_request_at: float = 0.0

    for q in queries:
        print(f"\n{'─'*70}")
        print(f"Query: {q!r}  (limit={limit})")
        print(f"{'─'*70}")
        try:
            # Enforce 1 req/s
            elapsed = time.monotonic() - last_request_at
            if elapsed < 1.0:
                wait = 1.0 - elapsed
                print(f"  [rate-limit] waiting {wait:.2f}s...")
                time.sleep(wait)

            last_request_at = time.monotonic()
            results = search(q, limit=limit, headers=headers)
            print(f"  → {len(results)} result(s) returned")
            print_results(results, verbose)
        except requests.HTTPError as e:
            print(f"  [ERROR] HTTP {e.response.status_code}: {e.response.text[:200]}")
            all_ok = False
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            all_ok = False

    return all_ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Semantic Scholar API connectivity and key")
    p.add_argument("--query", "-q", action="append", default=None,
                   help="Query string (can pass multiple times). Defaults to 3 sample queries.")
    p.add_argument("--limit", "-n", type=int, default=3,
                   help="Max results per query (default: 3)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show abstract snippets")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    queries = args.query or DEFAULT_QUERIES

    print(f"\n{'='*70}")
    print("  Semantic Scholar API Test")
    print(f"{'='*70}")

    ok = run_tests(queries=queries, limit=args.limit, verbose=args.verbose)

    print(f"\n{'='*70}")
    if ok:
        print("  ✓ All queries completed successfully")
    else:
        print("  ✗ Some queries failed — check API key / network above")
    print(f"{'='*70}\n")

    sys.exit(0 if ok else 1)
