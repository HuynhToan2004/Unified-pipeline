from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import ALL_MICRO_FLAWS, MACRO_TOPICS, NOVELTY_CUE_WORDS, SEVERITY_LEVELS, SEVERITY_MAP
from .io_utils import extract_citation_like_strings, flatten_sections_for_context, write_jsonl
from .llm_client import UnifiedLLMClient


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class Phase2Processor:
    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None, logger: Optional[logging.Logger] = None):
        self.llm = llm_client or UnifiedLLMClient()
        self.log = logger or logging.getLogger(__name__)

    def classify_spacy_sentence(self, text: str, section: str) -> Dict[str, str]:
        default = {
            "argument_role": _heuristic_argument_role(text, section),
            "aspect_macro_topic": _heuristic_macro_topic(text),
        }
        prompt = f"""
Classify this review sentence into JSON with exactly 2 keys:
{{"argument_role": "Claim|Premise", "aspect_macro_topic": "one allowed macro topic"}}

Allowed macro topics: {MACRO_TOPICS}
Sentence: {text}
Section: {section}
""".strip()
        out = self.llm.generate_json(prompt, default)
        role = str(out.get("argument_role", default["argument_role"])).strip()
        topic = str(out.get("aspect_macro_topic", default["aspect_macro_topic"])).strip()
        if role not in {"Claim", "Premise"}:
            role = default["argument_role"]
        if topic not in MACRO_TOPICS:
            topic = default["aspect_macro_topic"]
        return {"argument_role": role, "aspect_macro_topic": topic}

    def classify_atomic_argument(self, text: str, section: str) -> Dict[str, Any]:
        default = _heuristic_micro_and_severity(text, section)
        prompt = f"""
Classify this atomic review argument. Return strict JSON with keys:
{{"micro_flaw_label": "...", "severity": "Fatal|Major|Minor|None"}}

Allowed micro_flaw_label values: {ALL_MICRO_FLAWS}
Text: {text}
Section: {section}
""".strip()
        out = self.llm.generate_json(prompt, default)

        micro = str(out.get("micro_flaw_label", default["micro_flaw_label"])).strip()
        severity = str(out.get("severity", default["severity"])).strip()
        if micro not in ALL_MICRO_FLAWS:
            micro = default["micro_flaw_label"]
        if severity not in SEVERITY_LEVELS:
            severity = default["severity"]
        return {
            "micro_flaw_label": micro,
            "severity": severity,
            "severity_weight": SEVERITY_MAP[severity],
        }

    def extract_novelty_targets(self, paper_id: str, review_context: str) -> Dict[str, Any]:
        fallback = _heuristic_novelty_targets(paper_id, review_context)
        prompt = f"""
Extract novelty verification targets from peer-review text.
Return strict JSON with keys:
- paper.core_task
- paper.contributions (1-3 items with name/author_claim_text/description/source_hint)
- paper.key_terms
- paper.must_have_entities
- review_novelty_extracted.novelty_claims
- review_novelty_extracted.all_citations_raw

Review text:
{review_context}
""".strip()
        out = self.llm.generate_json(prompt, fallback)
        return _normalize_novelty_output(out, fallback)


def run_phase2(
    *,
    phase1_jsonl: Path,
    master_output_jsonl: Path,
    novelty_output_jsonl: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    log = logger or logging.getLogger(__name__)
    processor = Phase2Processor(logger=log)
    rows = read_jsonl(phase1_jsonl)

    master_rows: List[Dict[str, Any]] = []
    novelty_rows: List[Dict[str, Any]] = []

    for row in rows:
        parsed_sections = row.get("parsed_sections") or {}
        classified_sections: Dict[str, Any] = {}

        for section_name, section_payload in parsed_sections.items():
            sec_raw = str((section_payload or {}).get("raw_text", ""))
            spacy_sents = (section_payload or {}).get("spacy_sentences") or []
            llm_args = (section_payload or {}).get("llm_atomic_arguments") or []

            spacy_out: List[Dict[str, Any]] = []
            for s in spacy_sents:
                sent_id = str(s.get("sent_id", "")).strip()
                sent_text = str(s.get("text", "")).strip()
                if not sent_text:
                    continue
                cls = processor.classify_spacy_sentence(sent_text, section_name)
                spacy_out.append({"sent_id": sent_id, "text": sent_text, **cls})

            arg_out: List[Dict[str, Any]] = []
            for a in llm_args:
                arg_id = str(a.get("arg_id", "")).strip()
                arg_text = str(a.get("text", "")).strip()
                if not arg_text:
                    continue
                cls = processor.classify_atomic_argument(arg_text, section_name)
                arg_out.append({"arg_id": arg_id, "text": arg_text, **cls})

            classified_sections[section_name] = {
                "raw_text": sec_raw,
                "spacy_sentences_classification": spacy_out,
                "llm_atomic_arguments_classification": arg_out,
            }

        master_row = {
            "paper_id": row.get("paper_id"),
            "reviewer_id": row.get("reviewer_id"),
            "reviewer_type": row.get("reviewer_type"),
            "parsed_sections": classified_sections,
        }
        master_rows.append(master_row)

        review_context = flatten_sections_for_context(row.get("parsed_sections") or {})
        novelty_payload = processor.extract_novelty_targets(str(row.get("paper_id")), review_context)
        novelty_rows.append(
            {
                "paper_id": row.get("paper_id"),
                "reviewer_id": row.get("reviewer_id"),
                "reviewer_type": row.get("reviewer_type"),
                **novelty_payload,
            }
        )

    write_jsonl(master_output_jsonl, master_rows)
    write_jsonl(novelty_output_jsonl, novelty_rows)

    log.info(
        "Phase 2 done. master=%s novelty=%s llm_mode=%s",
        len(master_rows),
        len(novelty_rows),
        processor.llm.mode,
    )
    return master_rows, novelty_rows


def _heuristic_argument_role(text: str, section: str) -> str:
    low = text.lower()
    if section.lower() == "questions":
        return "Premise"
    if any(k in low for k in ["because", "since", "due to", "for example", "e.g."]):
        return "Premise"
    return "Claim"


def _heuristic_macro_topic(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ["baseline", "benchmark", "dataset", "metric", "evaluation", "experiment"]):
        return "Experimental Design & Evaluation"
    if any(k in low for k in ["novel", "novelty", "incremental", "contribution", "significance"]):
        return "Novelty & Contribution"
    if any(k in low for k in ["related", "citation", "prior work", "concurrent"]):
        return "Related work & Citations"
    if any(k in low for k in ["theory", "proof", "assumption", "method", "architecture", "intuition"]):
        return "Methodology & Theoretical Soundness"
    if any(k in low for k in ["complexity", "scalability", "limitation", "ethical", "broader impact"]):
        return "Applicability, Scalability & Limitations"
    if any(k in low for k in ["code", "reproduce", "implementation", "open-source", "repository"]):
        return "Reproducibility & Open Science"
    return "Clarity & Presentation"


def _heuristic_micro_and_severity(text: str, section: str) -> Dict[str, str]:
    low = text.lower()

    if any(k in low for k in ["fatal", "invalid", "incorrect proof", "data leakage"]):
        return {"micro_flaw_label": "Methodological Flaws", "severity": "Fatal"}
    if any(k in low for k in ["baseline", "ablation"]):
        return {"micro_flaw_label": "Missing/ weak Baselines", "severity": "Major"}
    if any(k in low for k in ["novelty", "incremental"]):
        return {"micro_flaw_label": "Limited Novelty", "severity": "Minor"}
    if any(k in low for k in ["citation", "related work", "prior work"]):
        return {"micro_flaw_label": "Missing Relevant Citations", "severity": "Major"}
    if any(k in low for k in ["clarity", "unclear", "typo", "grammar"]):
        return {"micro_flaw_label": "General writing & Clarity issues", "severity": "Minor"}
    if section.lower() == "questions":
        return {"micro_flaw_label": "Insufficient Implementation Details", "severity": "Minor"}
    return {"micro_flaw_label": "Insufficient Experimental Validation", "severity": "Minor"}


def _heuristic_novelty_targets(paper_id: str, review_context: str) -> Dict[str, Any]:
    lines = [ln.strip("-* \t") for ln in review_context.splitlines() if ln.strip()]
    claims: List[Dict[str, Any]] = []
    claim_idx = 1

    for line in lines:
        low = line.lower()
        if any(cue in low for cue in NOVELTY_CUE_WORDS):
            claims.append(
                {
                    "claim_id": f"s{claim_idx}",
                    "text": _limit_words(line, 40),
                    "stance": "not_novel" if "not" in low or "limited" in low or "incremental" in low else "novel",
                    "confidence_lang": "high" if any(k in low for k in ["clearly", "strongly"]) else "medium",
                    "mentions_prior_work": any(k in low for k in ["prior", "previous", "existing", "baseline"]),
                    "prior_work_strings": _extract_prior_terms(line),
                    "evidence_expected": "method_similarity",
                }
            )
            claim_idx += 1
        if len(claims) >= 5:
            break

    if not claims:
        claims.append(
            {
                "claim_id": "s1",
                "text": "No explicit novelty claim detected in this review context.",
                "stance": "unclear",
                "confidence_lang": "low",
                "mentions_prior_work": False,
                "prior_work_strings": [],
                "evidence_expected": "task_similarity",
            }
        )

    key_terms = _top_terms(review_context)
    entities = _extract_entities(review_context)

    return {
        "paper": {
            "core_task": _infer_core_task_from_context(review_context),
            "contributions": [
                {
                    "name": "Multimodal review-derived contribution",
                    "author_claim_text": _limit_words(_first_sentence(review_context), 35),
                    "description": "Auto-extracted from available review context (paper abstract/intro not provided).",
                    "source_hint": "Review-derived proxy",
                }
            ],
            "key_terms": key_terms,
            "must_have_entities": entities,
        },
        "review_novelty_extracted": {
            "novelty_claims": claims,
            "all_citations_raw": extract_citation_like_strings(review_context),
        },
    }


def _normalize_novelty_output(out: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(out, dict):
        return fallback

    paper = out.get("paper") if isinstance(out.get("paper"), dict) else fallback["paper"]
    review = out.get("review_novelty_extracted")
    if not isinstance(review, dict):
        review = fallback["review_novelty_extracted"]

    novelty_claims = review.get("novelty_claims")
    if not isinstance(novelty_claims, list) or not novelty_claims:
        review["novelty_claims"] = fallback["review_novelty_extracted"]["novelty_claims"]

    citations = review.get("all_citations_raw")
    if not isinstance(citations, list):
        review["all_citations_raw"] = fallback["review_novelty_extracted"]["all_citations_raw"]

    return {"paper": paper, "review_novelty_extracted": review}


def _infer_core_task_from_context(text: str) -> str:
    lower = text.lower()
    if "multimodal" in lower and "generation" in lower:
        return "Enable any-to-any multimodal understanding and generation."
    if "review" in lower and "paper" in lower:
        return "Assess scientific paper quality and novelty from peer reviews."
    return "Infer the main research task from available review context."


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text.strip()


def _limit_words(text: str, n: int) -> str:
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[:n]).strip() + "..."


def _extract_prior_terms(text: str) -> List[str]:
    patt = re.compile(r"\b(?:prior|previous|existing)\s+\w+(?:\s+\w+){0,4}\b", re.IGNORECASE)
    hits = patt.findall(text)
    uniq: List[str] = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    return uniq


def _top_terms(text: str, max_terms: int = 8) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
    stop = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "not",
        "paper",
        "model",
        "method",
        "review",
    }
    freq: Dict[str, int] = {}
    for tok in tokens:
        key = tok.lower()
        if key in stop:
            continue
        freq[key] = freq.get(key, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_terms]]


def _extract_entities(text: str, max_items: int = 8) -> List[str]:
    candidates = re.findall(r"\b(?:[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*)\b", text)
    uniq: List[str] = []
    for c in candidates:
        cc = c.strip()
        if cc not in uniq and len(cc) > 2:
            uniq.append(cc)
        if len(uniq) >= max_items:
            break
    return uniq
