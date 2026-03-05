from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

from config import SEVERITY_MAP


def calculate_nsr(arguments: List[Dict[str, Any]], paper_decision: str) -> float:
    sum_tokens_noise = 0
    sum_tokens_signal = 0

    for argument in arguments:
        severity = argument.get("severity")
        token_count = len(str(argument.get("content", "")).strip().split())
        if severity == "Minor":
            sum_tokens_noise += token_count
        elif severity in {"Fatal", "Major"}:
            sum_tokens_signal += token_count

    if sum_tokens_signal == 0:
        decision = (paper_decision or "").lower()
        if "accept" in decision:
            return 0.0
        return 100.0

    nsr = float(sum_tokens_noise) / float(sum_tokens_signal)
    return round(min(nsr, 100.0), 4)


def calculate_cps(arguments: List[Dict[str, Any]]) -> float:
    cps = 0.0
    section_rank_counter = defaultdict(int)

    for argument in arguments:
        section = str(argument.get("section") or "Unknown")
        severity = str(argument.get("severity") or "None")
        weight = float(SEVERITY_MAP.get(severity, 0.0))

        if section in {"Paper Decision", "Decision", "Conclusion"}:
            continue
        if weight == 0.0:
            continue

        section_rank_counter[section] += 1
        current_rank = section_rank_counter[section]
        denominator = math.log2(current_rank + 1)
        cps += weight / denominator

    return round(cps, 4)


def calculate_flaw_weights(
    flaw_dict: Dict[str, Dict[str, Dict[str, List[str]]]],
    meta_weight_bonus: float = 1.0,
) -> Dict[str, float]:
    flaw_weights: Dict[str, float] = {}

    for _, micro_flaws in flaw_dict.items():
        for micro_flaw, reviewers_map in micro_flaws.items():
            score = 0.0
            for reviewer_id, quotes in reviewers_map.items():
                if quotes:
                    score += 1.0
                    if "Meta" in reviewer_id:
                        score += float(meta_weight_bonus)
            if score > 0.0:
                flaw_weights[micro_flaw] = score

    return flaw_weights


def calculate_reviewer_performance(
    flaw_dict: Dict[str, Dict[str, Dict[str, List[str]]]],
    flaw_weights: Dict[str, float],
) -> Dict[str, float]:
    reviewer_scores = defaultdict(float)

    for _, micro_flaws in flaw_dict.items():
        for micro_flaw, reviewers_map in micro_flaws.items():
            weight = flaw_weights.get(micro_flaw, 0.0)
            if weight == 0.0:
                continue
            for reviewer_id, quotes in reviewers_map.items():
                if quotes:
                    reviewer_scores[reviewer_id] += weight

    return {key: round(value, 4) for key, value in reviewer_scores.items()}


def calculate_macro_topic_stats(
    flaw_dict: Dict[str, Dict[str, Dict[str, List[str]]]],
    flaw_weights: Dict[str, float],
) -> Dict[str, Any]:
    macro_stats: Dict[str, Any] = {}

    for macro_topic, micro_flaws in flaw_dict.items():
        total_score = 0.0
        micro_stats: Dict[str, float] = {}
        count = 0

        for micro_flaw in micro_flaws.keys():
            score = float(flaw_weights.get(micro_flaw, 0.0))
            if score <= 0.0:
                continue
            micro_stats[micro_flaw] = round(score, 4)
            total_score += score
            count += 1

        if count > 0:
            macro_stats[macro_topic] = {
                "count": count,
                "total_score": round(total_score, 4),
                "micro_flaws": micro_stats,
                "avg_score": round(total_score / float(count), 4),
            }

    return macro_stats


def calculate_novelty_metrics(item: Dict[str, Any]) -> Dict[str, Any]:
    review = item.get("review") if isinstance(item.get("review"), dict) else {}
    claims = review.get("novelty_claims") if isinstance(review.get("novelty_claims"), list) else []
    citations = review.get("all_citations_raw") if isinstance(review.get("all_citations_raw"), list) else []
    stats = item.get("stats") if isinstance(item.get("stats"), dict) else {}

    claim_count = len(claims)
    not_novel_count = 0
    prior_mentions = 0
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        if str(claim.get("stance", "")).strip().lower() == "not_novel":
            not_novel_count += 1
        if bool(claim.get("mentions_prior_work")):
            prior_mentions += 1

    retrieved_final = int(stats.get("final", 0) or 0)
    retrieval_signal = min(float(retrieved_final) / 10.0, 1.0)
    stance_signal = float(not_novel_count) / float(claim_count) if claim_count > 0 else 0.0
    prior_signal = min(float(prior_mentions + len(citations)) / 5.0, 1.0)

    novelty_risk_score = round(100.0 * (0.6 * stance_signal + 0.3 * retrieval_signal + 0.1 * prior_signal), 2)

    return {
        "claim_count": claim_count,
        "not_novel_claim_count": not_novel_count,
        "prior_work_mention_count": prior_mentions,
        "citation_count": len(citations),
        "retrieved_candidate_count": retrieved_final,
        "novelty_risk_score": novelty_risk_score,
    }
