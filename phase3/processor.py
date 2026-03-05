from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from semantic_client import SemanticScholarClient
from utils import as_str, read_jsonl, safe_list, write_json, write_text


TOP_K_DEFAULT = 30


class Phase3NoveltyProcessor:
    def __init__(self, *, logger: logging.Logger):
        self.log = logger
        self.semantic_client = SemanticScholarClient(logger=self.log)

    def process_record(self, record: Dict[str, Any], max_candidates_per_query: int) -> Dict[str, Any]:
        paper = record.get("paper") if isinstance(record.get("paper"), dict) else {}
        review = record.get("review") if isinstance(record.get("review"), dict) else {}

        core_task = as_str(paper.get("core_task"))
        contributions = [as_str(x) for x in safe_list(paper.get("contributions")) if as_str(x)]

        query_specs = self._build_query_specs(core_task=core_task, contributions=contributions)
        raw_candidates: List[Dict[str, Any]] = []
        query_meta: List[Dict[str, Any]] = []

        for spec in query_specs:
            papers = self.semantic_client.search(spec["query"], limit=max_candidates_per_query)
            normalized = [
                self._normalize_candidate(paper_item=paper_item, source_query=spec["id"])
                for paper_item in papers
            ]
            raw_candidates.extend(normalized)
            query_meta.append(
                {
                    "id": spec["id"],
                    "query": spec["query"],
                    "status": "ok",
                    "count": len(normalized),
                }
            )

        total_candidates = len(raw_candidates)
        deduped = self._dedup_by_title(raw_candidates)
        after_dedup = len(deduped)

        paper_year = self._coerce_year(paper.get("paper_year"))
        year_filtered = self._filter_by_year(deduped, paper_year)
        after_year = len(year_filtered)

        final_candidates = year_filtered[:TOP_K_DEFAULT]

        return {
            "paper": paper,
            "review": review,
            "mode": "per_contribution",
            "paper_year": paper_year,
            "queries": query_meta,
            "candidate_pool_top30": [self._format_candidate(cand) for cand in final_candidates],
            "stats": {
                "total_candidates": total_candidates,
                "after_dedup": after_dedup,
                "after_nontechnical_filter": after_dedup,
                "after_year_filter": after_year,
                "final": len(final_candidates),
            },
        }

    def _build_query_specs(self, *, core_task: str, contributions: List[str]) -> List[Dict[str, str]]:
        query_specs: List[Dict[str, str]] = []
        for idx, contribution in enumerate(contributions, start=1):
            query = " ".join([part for part in [core_task, contribution] if part]).strip()
            if not query:
                continue
            query_specs.append({"id": f"C{idx}", "query": query})
        if not query_specs and core_task:
            query_specs.append({"id": "Q1", "query": core_task})
        return query_specs

    def _normalize_candidate(self, *, paper_item: Dict[str, Any], source_query: str) -> Dict[str, Any]:
        title = as_str(paper_item.get("title"))
        external_ids = paper_item.get("externalIds") if isinstance(paper_item.get("externalIds"), dict) else {}

        cand_id = as_str(paper_item.get("paperId"))
        if not cand_id:
            doi = as_str(external_ids.get("DOI") or external_ids.get("doi"))
            arxiv_id = as_str(
                external_ids.get("ArXiv")
                or external_ids.get("arXiv")
                or external_ids.get("arxiv")
            )
            if doi:
                cand_id = f"doi:{doi}"
            elif arxiv_id:
                cand_id = f"arxiv:{arxiv_id}"
            elif title:
                cand_id = f"title:{title.lower()}"

        return {
            "cand_id": cand_id,
            "title": title,
            "year": self._coerce_year(paper_item.get("year")),
            "venue": as_str(paper_item.get("venue")),
            "abstract": as_str(paper_item.get("abstract")),
            "url": as_str(paper_item.get("url")),
            "embedding": None,
            "source_query": source_query,
        }

    def _format_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "cand_id": candidate.get("cand_id"),
            "title": candidate.get("title"),
            "year": candidate.get("year"),
            "venue": candidate.get("venue"),
            "abstract": candidate.get("abstract"),
            "url": candidate.get("url"),
            "embedding": candidate.get("embedding"),
        }

    def _dedup_by_title(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_titles = set()
        deduped: List[Dict[str, Any]] = []
        for candidate in candidates:
            title_key = as_str(candidate.get("title")).lower()
            if not title_key or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            deduped.append(candidate)
        return deduped

    def _coerce_year(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _filter_by_year(self, candidates: List[Dict[str, Any]], paper_year: Optional[int]) -> List[Dict[str, Any]]:
        if paper_year is None:
            return candidates
        filtered: List[Dict[str, Any]] = []
        for candidate in candidates:
            candidate_year = self._coerce_year(candidate.get("year"))
            if candidate_year is None or candidate_year <= paper_year:
                filtered.append(candidate)
        return filtered


def run_phase3_novelty(
    *,
    phase2_novelty_input: Path,
    output_json: Path,
    output_md: Path,
    max_candidates_per_query: int = 8,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger or logging.getLogger(__name__)
    records = read_jsonl(phase2_novelty_input)

    processor = Phase3NoveltyProcessor(logger=log)
    outputs: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records, start=1):
        outputs.append(processor.process_record(rec, max_candidates_per_query=max_candidates_per_query))
        if idx % 10 == 0:
            log.info("Phase3 novelty progress: %s/%s", idx, len(records))

    report = {
        "meta": {
            "records": len(outputs),
            "phase3_mode": "retrieval_only",
            "input": str(phase2_novelty_input),
        },
        "items": outputs,
    }

    write_json(output_json, report)
    write_text(output_md, _render_markdown_report(report))
    log.info("Phase3 novelty completed. items=%s output=%s", len(outputs), output_json)
    return report


def _render_markdown_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    meta = report.get("meta", {})
    lines.append("# Phase3 Novelty Assessment Report")
    lines.append("")
    lines.append(f"- Records: {meta.get('records', 0)}")
    lines.append(f"- Mode: {meta.get('phase3_mode', 'unknown')}")
    lines.append("")

    for idx, item in enumerate(report.get("items", []), start=1):
        paper = item.get("paper", {})
        review = item.get("review", {})
        stats = item.get("stats", {})
        lines.append(f"## Item {idx}")
        lines.append(f"- Core task: {as_str(paper.get('core_task'))}")
        lines.append(f"- Queries used: {len(safe_list(item.get('queries')))}")
        lines.append(f"- Retrieved candidates (final): {as_str(stats.get('final'))}")
        lines.append(f"- Novelty claims: {len(safe_list(review.get('novelty_claims')))}")
        lines.append("")

    return "\n".join(lines) + "\n"
