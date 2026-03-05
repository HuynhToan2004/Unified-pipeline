from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from metrics import (
    calculate_cps,
    calculate_flaw_weights,
    calculate_macro_topic_stats,
    calculate_novelty_metrics,
    calculate_nsr,
    calculate_reviewer_performance,
)
from utils import as_str, read_json, read_jsonl, safe_list, write_json, write_text


def run_phase4_metrics(
    *,
    phase2_master_input: Path,
    phase3_retrieval_input: Path,
    output_json: Path,
    output_md: Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger or logging.getLogger(__name__)

    master_rows = read_jsonl(phase2_master_input)
    phase3_report = read_json(phase3_retrieval_input)
    phase3_items = safe_list(phase3_report.get("items"))

    flaw_dict = _build_flaw_dict(master_rows)
    flaw_weights = calculate_flaw_weights(flaw_dict)
    reviewer_cfi_scores = calculate_reviewer_performance(flaw_dict, flaw_weights)
    macro_topic_stats = calculate_macro_topic_stats(flaw_dict, flaw_weights)

    reviewer_type_map: Dict[str, str] = {}
    reviewer_csp_details: Dict[str, Dict[str, Any]] = {}

    for row in master_rows:
        reviewer_id = as_str(row.get("reviewer_id"))
        reviewer_type = as_str(row.get("reviewer_type"))
        reviewer_type_map[reviewer_id] = reviewer_type

        atomic_arguments = _collect_atomic_arguments(row)
        decision_text = _extract_decision_text(row)
        reviewer_csp_details[reviewer_id] = {
            "reviewer_type": reviewer_type,
            "cps": calculate_cps(atomic_arguments),
            "nsr": calculate_nsr(atomic_arguments, decision_text),
            "argument_count": len(atomic_arguments),
            "decision_text": decision_text,
        }

    novelty_per_reviewer: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(phase3_items):
        if idx >= len(master_rows):
            break
        reviewer_id = as_str(master_rows[idx].get("reviewer_id"))
        reviewer_type = as_str(master_rows[idx].get("reviewer_type"))
        novelty = calculate_novelty_metrics(item if isinstance(item, dict) else {})
        novelty["reviewer_type"] = reviewer_type
        novelty_per_reviewer[reviewer_id] = novelty

    aggregated_by_type = _aggregate_by_reviewer_type(
        reviewer_type_map=reviewer_type_map,
        reviewer_cfi_scores=reviewer_cfi_scores,
        reviewer_csp_details=reviewer_csp_details,
        novelty_per_reviewer=novelty_per_reviewer,
    )

    report = {
        "meta": {
            "phase": "phase4",
            "inputs": {
                "phase2_master": str(phase2_master_input),
                "phase3_retrieval": str(phase3_retrieval_input),
            },
            "records": len(master_rows),
        },
        "cfi": {
            "flaw_weights": flaw_weights,
            "macro_topic_stats": macro_topic_stats,
            "reviewer_scores": reviewer_cfi_scores,
        },
        "csp": {
            "reviewer_scores": reviewer_csp_details,
        },
        "novelty": {
            "reviewer_scores": novelty_per_reviewer,
        },
        "summary_by_reviewer_type": aggregated_by_type,
    }

    write_json(output_json, report)
    write_text(output_md, _render_markdown_report(report))
    log.info("Phase4 completed. records=%s output=%s", len(master_rows), output_json)
    return report


def _build_flaw_dict(master_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    flaw_dict: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for row in master_rows:
        reviewer_id = as_str(row.get("reviewer_id"))
        parsed_sections = row.get("parsed_sections") if isinstance(row.get("parsed_sections"), dict) else {}

        for section_payload in parsed_sections.values():
            payload = section_payload if isinstance(section_payload, dict) else {}
            arguments = safe_list(payload.get("llm_atomic_arguments_classification"))
            for argument in arguments:
                if not isinstance(argument, dict):
                    continue
                macro = as_str(argument.get("aspect_macro_topic")) or "Unknown"
                micro = as_str(argument.get("micro_flaw_label")) or "Unknown"
                quote = as_str(argument.get("text"))
                if quote:
                    flaw_dict[macro][micro][reviewer_id].append(quote)

    return {k: dict(v) for k, v in flaw_dict.items()}


def _collect_atomic_arguments(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    parsed_sections = row.get("parsed_sections") if isinstance(row.get("parsed_sections"), dict) else {}
    collected: List[Dict[str, Any]] = []

    for section_name, section_payload in parsed_sections.items():
        payload = section_payload if isinstance(section_payload, dict) else {}
        arguments = safe_list(payload.get("llm_atomic_arguments_classification"))
        for argument in arguments:
            if not isinstance(argument, dict):
                continue
            collected.append(
                {
                    "section": section_name,
                    "severity": as_str(argument.get("severity")) or "None",
                    "content": as_str(argument.get("text")),
                }
            )

    return collected


def _extract_decision_text(row: Dict[str, Any]) -> str:
    parsed_sections = row.get("parsed_sections") if isinstance(row.get("parsed_sections"), dict) else {}
    all_text = []
    for section_payload in parsed_sections.values():
        payload = section_payload if isinstance(section_payload, dict) else {}
        all_text.append(as_str(payload.get("raw_text")))
    joined = "\n".join([value for value in all_text if value])

    match = re.search(r"decision\s*:\s*([A-Za-z ]+)", joined, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    low = joined.lower()
    if "reject" in low:
        return "Reject"
    if "accept" in low:
        return "Accept"
    return "Unknown"


def _aggregate_by_reviewer_type(
    *,
    reviewer_type_map: Dict[str, str],
    reviewer_cfi_scores: Dict[str, float],
    reviewer_csp_details: Dict[str, Dict[str, Any]],
    novelty_per_reviewer: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for reviewer_id, reviewer_type in reviewer_type_map.items():
        current_type = reviewer_type or "Unknown"

        cfi_score = float(reviewer_cfi_scores.get(reviewer_id, 0.0))
        groups[current_type]["cfi"].append(cfi_score)

        csp_detail = reviewer_csp_details.get(reviewer_id, {})
        groups[current_type]["cps"].append(float(csp_detail.get("cps", 0.0)))
        groups[current_type]["nsr"].append(float(csp_detail.get("nsr", 0.0)))

        novelty_detail = novelty_per_reviewer.get(reviewer_id, {})
        groups[current_type]["novelty_risk_score"].append(float(novelty_detail.get("novelty_risk_score", 0.0)))

    summary: Dict[str, Any] = {}
    for reviewer_type, metrics_map in groups.items():
        summary[reviewer_type] = {
            "reviewer_count": len(metrics_map.get("cfi", [])),
            "avg_cfi": _avg(metrics_map.get("cfi", [])),
            "avg_cps": _avg(metrics_map.get("cps", [])),
            "avg_nsr": _avg(metrics_map.get("nsr", [])),
            "avg_novelty_risk_score": _avg(metrics_map.get("novelty_risk_score", [])),
        }

    return summary


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / float(len(values)), 4)


def _render_markdown_report(report: Dict[str, Any]) -> str:
    meta = report.get("meta", {})
    summary = report.get("summary_by_reviewer_type", {})
    lines: List[str] = []

    lines.append("# Phase4 Metrics Report")
    lines.append("")
    lines.append(f"- Records: {meta.get('records', 0)}")
    lines.append(f"- Phase2 input: {as_str((meta.get('inputs') or {}).get('phase2_master'))}")
    lines.append(f"- Phase3 input: {as_str((meta.get('inputs') or {}).get('phase3_retrieval'))}")
    lines.append("")

    lines.append("## Summary by Reviewer Type")
    if not isinstance(summary, dict) or not summary:
        lines.append("- No summary available.")
        lines.append("")
        return "\n".join(lines) + "\n"

    for reviewer_type, row in summary.items():
        lines.append(f"### {reviewer_type}")
        lines.append(f"- Reviewer count: {row.get('reviewer_count', 0)}")
        lines.append(f"- Avg CFI: {row.get('avg_cfi', 0.0)}")
        lines.append(f"- Avg CPS: {row.get('avg_cps', 0.0)}")
        lines.append(f"- Avg NSR: {row.get('avg_nsr', 0.0)}")
        lines.append(f"- Avg Novelty Risk: {row.get('avg_novelty_risk_score', 0.0)}")
        lines.append("")

    return "\n".join(lines) + "\n"
