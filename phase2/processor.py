from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import ALL_MICRO_FLAWS, MACRO_TOPICS, MICRO_TO_MACRO, SEVERITY_MAP
from llm_backend import UnifiedLLMBackend
from prompts import build_record_level_prompt
from utils import extract_citation_like_strings, flatten_review_context, load_paper_abstract_intro, read_jsonl, write_jsonl


class Phase2Processor:
    def __init__(self, llm_provider: str = "gemini", llm_model: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.llm = UnifiedLLMBackend(provider=llm_provider, model=llm_model, temperature=0.0)
        self.log = logger or logging.getLogger(__name__)

    def classify_record_one_call(
        self,
        *,
        paper_id: str,
        reviewer_id: str,
        reviewer_type: str,
        parsed_sections: Dict[str, Any],
        paper_abstract_intro: str,
        review_context: str,
    ) -> Dict[str, Any]:
        fallback = self._heuristic_record_output(
            paper_id=paper_id,
            reviewer_id=reviewer_id,
            reviewer_type=reviewer_type,
            parsed_sections=parsed_sections,
            paper_abstract_intro=paper_abstract_intro,
            review_context=review_context,
        )

        prompt = build_record_level_prompt(
            paper_id=paper_id,
            reviewer_id=reviewer_id,
            reviewer_type=reviewer_type,
            parsed_sections=parsed_sections,
            paper_abstract_intro=paper_abstract_intro,
            review_context=review_context,
        )

        llm_out = self.llm.generate_json(prompt, fallback)
        return self._normalize_record_output(llm_out, fallback)

    def _heuristic_record_output(
        self,
        *,
        paper_id: str,
        reviewer_id: str,
        reviewer_type: str,
        parsed_sections: Dict[str, Any],
        paper_abstract_intro: str,
        review_context: str,
    ) -> Dict[str, Any]:
        out_sections: Dict[str, Any] = {}
        for section_name, section_payload in parsed_sections.items():
            spacy_sentences = (section_payload or {}).get("spacy_sentences") or []
            atomic_arguments = (section_payload or {}).get("llm_atomic_arguments") or []
            section_result = self._heuristic_section_classification(section_name, spacy_sentences, atomic_arguments)
            out_sections[section_name] = {
                "spacy_sentences_classification": section_result["spacy_sentences_classification"],
                "llm_atomic_arguments_classification": section_result["llm_atomic_arguments_classification"],
            }

        novelty = self._heuristic_novelty_targets(paper_abstract_intro, review_context)
        return {
            "parsed_sections": out_sections,
            "paper": novelty["paper"],
            "review_novelty_extracted": novelty["review_novelty_extracted"],
        }

    def _normalize_record_output(self, output: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(output, dict):
            return fallback

        out_sections = output.get("parsed_sections") if isinstance(output.get("parsed_sections"), dict) else {}
        fallback_sections = fallback["parsed_sections"]

        normalized_sections: Dict[str, Any] = {}
        for section_name, fallback_section in fallback_sections.items():
            section_out = out_sections.get(section_name)
            if not isinstance(section_out, dict):
                section_out = fallback_section
            normalized_sections[section_name] = self._normalize_section_output(section_out, fallback_section)

        novelty_fallback = {
            "paper": fallback["paper"],
            "review_novelty_extracted": fallback["review_novelty_extracted"],
        }
        novelty_out = self._normalize_novelty_output(output, novelty_fallback)

        return {
            "parsed_sections": normalized_sections,
            "paper": novelty_out["paper"],
            "review_novelty_extracted": novelty_out["review_novelty_extracted"],
        }

    def _normalize_section_output(self, output: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(output, dict):
            return fallback

        spacy_rows = output.get("spacy_sentences_classification")
        if not isinstance(spacy_rows, list):
            spacy_rows = fallback["spacy_sentences_classification"]

        atomic_rows = output.get("llm_atomic_arguments_classification")
        if not isinstance(atomic_rows, list):
            atomic_rows = fallback["llm_atomic_arguments_classification"]

        norm_spacy: List[Dict[str, Any]] = []
        for row in spacy_rows:
            if not isinstance(row, dict):
                continue
            sent_id = str(row.get("sent_id", "")).strip()
            text = str(row.get("text", "")).strip()
            role = str(row.get("argument_role", "Claim")).strip()
            topic = str(row.get("aspect_macro_topic", "Clarity & Presentation")).strip()
            if role not in {"Claim", "Premise"}:
                role = "Claim"
            if topic not in MACRO_TOPICS:
                topic = _heuristic_macro_topic(text)
            if sent_id and text:
                norm_spacy.append(
                    {
                        "sent_id": sent_id,
                        "text": text,
                        "argument_role": role,
                        "aspect_macro_topic": topic,
                    }
                )

        norm_atomic: List[Dict[str, Any]] = []
        for row in atomic_rows:
            if not isinstance(row, dict):
                continue
            arg_id = str(row.get("arg_id", "")).strip()
            text = str(row.get("text", "")).strip()
            micro = str(row.get("micro_flaw_label", "Insufficient Experimental Validation")).strip()
            severity = str(row.get("severity", "Minor")).strip()
            if micro not in ALL_MICRO_FLAWS:
                micro = _heuristic_micro_flaw(text)
            if severity not in SEVERITY_MAP:
                severity = _heuristic_severity(text)
            if arg_id and text:
                norm_atomic.append(
                    {
                        "arg_id": arg_id,
                        "text": text,
                        "micro_flaw_label": micro,
                        "aspect_macro_topic": MICRO_TO_MACRO.get(micro, _heuristic_macro_topic(text)),
                        "severity": severity,
                        "severity_weight": SEVERITY_MAP[severity],
                    }
                )

        return {
            "spacy_sentences_classification": norm_spacy,
            "llm_atomic_arguments_classification": norm_atomic,
        }

    def _normalize_novelty_output(self, output: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(output, dict):
            return fallback

        paper = output.get("paper") if isinstance(output.get("paper"), dict) else fallback["paper"]
        review = (
            output.get("review_novelty_extracted")
            if isinstance(output.get("review_novelty_extracted"), dict)
            else fallback["review_novelty_extracted"]
        )

        claims = review.get("novelty_claims")
        if not isinstance(claims, list):
            review["novelty_claims"] = fallback["review_novelty_extracted"]["novelty_claims"]

        citations = review.get("all_citations_raw")
        if not isinstance(citations, list):
            review["all_citations_raw"] = fallback["review_novelty_extracted"]["all_citations_raw"]

        return {"paper": paper, "review_novelty_extracted": review}

    def _heuristic_section_classification(
        self,
        section_name: str,
        spacy_sentences: List[Dict[str, Any]],
        atomic_arguments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        spacy_out = []
        for row in spacy_sentences:
            text = str(row.get("text", "")).strip()
            sent_id = str(row.get("sent_id", "")).strip()
            if not text or not sent_id:
                continue
            spacy_out.append(
                {
                    "sent_id": sent_id,
                    "text": text,
                    "argument_role": _heuristic_argument_role(text, section_name),
                    "aspect_macro_topic": _heuristic_macro_topic(text),
                }
            )

        atomic_out = []
        for row in atomic_arguments:
            text = str(row.get("text", "")).strip()
            arg_id = str(row.get("arg_id", "")).strip()
            if not text or not arg_id:
                continue
            micro = _heuristic_micro_flaw(text)
            sev = _heuristic_severity(text)
            atomic_out.append(
                {
                    "arg_id": arg_id,
                    "text": text,
                    "micro_flaw_label": micro,
                    "severity": sev,
                    "severity_weight": SEVERITY_MAP[sev],
                }
            )

        return {
            "spacy_sentences_classification": spacy_out,
            "llm_atomic_arguments_classification": atomic_out,
        }

    def _heuristic_novelty_targets(self, paper_abstract_intro: str, review_context: str) -> Dict[str, Any]:
        source = (paper_abstract_intro or "").strip()
        if not source:
            source = review_context

        claims: List[Dict[str, Any]] = []
        lines = [ln.strip("-*• \t") for ln in review_context.splitlines() if ln.strip()]
        idx = 1
        for line in lines:
            low = line.lower()
            if any(k in low for k in ["novel", "novelty", "incremental", "prior work", "previous", "similar"]):
                claims.append(
                    {
                        "claim_id": f"s{idx}",
                        "text": _limit_words(line, 40),
                        "stance": "not_novel" if any(k in low for k in ["not", "limited", "incremental", "similar"]) else "novel",
                        "confidence_lang": "high" if any(k in low for k in ["clearly", "strongly"]) else "medium",
                        "mentions_prior_work": any(k in low for k in ["prior", "previous", "baseline", "related"]),
                        "prior_work_strings": _extract_prior_strings(line),
                        "evidence_expected": "method_similarity",
                    }
                )
                idx += 1
            if len(claims) >= 5:
                break

        return {
            "paper": {
                "core_task": _heuristic_core_task(source),
                "contributions": [
                    {
                        "name": "A multimodal generation framework",
                        "author_claim_text": _limit_words(_first_sentence(source), 30),
                        "description": "Auto-extracted from available paper/review context.",
                        "source_hint": "Abstract/Introduction proxy",
                    }
                ],
                "key_terms": _extract_key_terms(source),
                "must_have_entities": _extract_entities(source),
            },
            "review_novelty_extracted": {
                "novelty_claims": claims,
                "all_citations_raw": extract_citation_like_strings(review_context),
            },
        }


def run_phase2(
    *,
    phase1_jsonl: Path,
    master_output_jsonl: Path,
    novelty_output_jsonl: Path,
    paper_context_dir: Optional[Path] = None,
    llm_provider: str = "gemini",
    llm_model: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    log = logger or logging.getLogger(__name__)
    rows = read_jsonl(phase1_jsonl)
    processor = Phase2Processor(llm_provider=llm_provider, llm_model=llm_model, logger=log)

    master_rows: List[Dict[str, Any]] = []
    novelty_rows: List[Dict[str, Any]] = []

    for row in rows:
        paper_id = str(row.get("paper_id", ""))
        reviewer_id = str(row.get("reviewer_id", ""))
        reviewer_type = str(row.get("reviewer_type", ""))
        parsed_sections = row.get("parsed_sections") or {}

        review_context = flatten_review_context(parsed_sections)
        paper_context = load_paper_abstract_intro(paper_id, paper_context_dir=paper_context_dir)

        one_call_result = processor.classify_record_one_call(
            paper_id=paper_id,
            reviewer_id=reviewer_id,
            reviewer_type=reviewer_type,
            parsed_sections=parsed_sections,
            paper_abstract_intro=paper_context,
            review_context=review_context,
        )

        out_sections: Dict[str, Any] = {}
        normalized_sections = one_call_result.get("parsed_sections") or {}
        for section_name, section_payload in parsed_sections.items():
            section_result = normalized_sections.get(section_name, {})
            out_sections[section_name] = {
                "raw_text": str((section_payload or {}).get("raw_text", "")),
                "spacy_sentences_classification": (section_result or {}).get("spacy_sentences_classification", []),
                "llm_atomic_arguments_classification": (section_result or {}).get("llm_atomic_arguments_classification", []),
            }

        master_rows.append(
            {
                "paper_id": paper_id,
                "reviewer_id": reviewer_id,
                "reviewer_type": reviewer_type,
                "parsed_sections": out_sections,
            }
        )

        novelty_rows.append(
            _to_task1_like_output(
                paper=one_call_result.get("paper", {}),
                review_like=one_call_result.get("review_novelty_extracted", {}),
            )
        )

    write_jsonl(master_output_jsonl, master_rows)
    write_jsonl(novelty_output_jsonl, novelty_rows)
    log.info(
        "Phase2 completed. master=%s novelty=%s llm_mode=%s",
        len(master_rows),
        len(novelty_rows),
        processor.llm.mode,
    )

    return master_rows, novelty_rows


def _heuristic_argument_role(text: str, section_name: str) -> str:
    low = text.lower()
    if section_name.lower() == "questions":
        return "Premise"
    if any(k in low for k in ["because", "since", "due to", "for example", "e.g.", "therefore"]):
        return "Premise"
    return "Claim"


def _heuristic_macro_topic(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ["baseline", "benchmark", "dataset", "evaluation", "ablation", "metric"]):
        return "Experimental Design & Evaluation"
    if any(k in low for k in ["novelty", "novel", "incremental", "significance", "contribution"]):
        return "Novelty & Contribution"
    if any(k in low for k in ["citation", "related work", "prior", "concurrent"]):
        return "Related work & Citations"
    if any(k in low for k in ["proof", "theory", "assumption", "method", "intuition", "objective"]):
        return "Methodology & Theoretical Soundness"
    if any(k in low for k in ["limitation", "scalability", "complexity", "ethical", "broader impact"]):
        return "Applicability, Scalability & Limitations"
    if any(k in low for k in ["code", "reproduc", "implementation", "repository", "open-source"]):
        return "Reproducibility & Open Science"
    return "Clarity & Presentation"


def _heuristic_micro_flaw(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ["baseline", "ablation"]):
        return "Missing/ weak Baselines"
    if any(k in low for k in ["novelty", "incremental"]):
        return "Limited Novelty"
    if any(k in low for k in ["citation", "prior work", "related work"]):
        return "Missing Relevant Citations"
    if any(k in low for k in ["proof", "theoretical", "assumption"]):
        return "Weak Theoretical Justification/Proofs"
    if any(k in low for k in ["typo", "grammar", "unclear", "writing"]):
        return "General writing & Clarity issues"
    if any(k in low for k in ["reproduce", "implementation", "code", "repository"]):
        return "Insufficient Implementation Details"
    if any(k in low for k in ["limitation", "failure case", "robustness"]):
        return "Lack of Discussion on Limitations"
    return "Insufficient Experimental Validation"


def _heuristic_severity(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ["fatal", "invalid", "incorrect", "data leakage", "fundamental"]):
        return "Fatal"
    if any(k in low for k in ["missing comparison", "no baseline", "questionable evaluation", "major concern"]):
        return "Major"
    if any(k in low for k in ["typo", "grammar", "unclear", "question"]):
        return "Minor"
    return "Minor"


def _heuristic_core_task(text: str) -> str:
    low = text.lower()
    if "multimodal" in low and "generation" in low:
        return "Enable any-to-any multimodal understanding and generation."
    if "review" in low and "paper" in low:
        return "Assess scientific papers with structured peer-review analysis."
    return "Infer the research core task from abstract and introduction."


def _extract_key_terms(text: str, limit: int = 10) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "using",
        "paper",
        "model",
        "method",
    }
    freq: Dict[str, int] = {}
    for t in tokens:
        k = t.lower()
        if k in stop:
            continue
        freq[k] = freq.get(k, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]]


def _extract_entities(text: str, limit: int = 12) -> List[str]:
    cands = re.findall(r"\b(?:[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*)\b", text)
    banned = {
        "Summary",
        "Strengths",
        "Weaknesses",
        "Questions",
        "The",
        "This",
        "That",
        "We",
        "As",
        "In",
        "However",
        "Finally",
    }
    out: List[str] = []
    for c in cands:
        cc = c.strip()
        if cc in banned:
            continue
        if cc.isupper() and len(cc) <= 2:
            continue
        if cc not in out:
            out.append(cc)
        if len(out) >= limit:
            break
    return out


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0].strip() if parts else text.strip()


def _limit_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + "..."


def _extract_prior_strings(text: str) -> List[str]:
    hits = re.findall(r"\b(?:prior|previous|existing)\s+\w+(?:\s+\w+){0,5}\b", text, flags=re.IGNORECASE)
    uniq: List[str] = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    return uniq


def _to_task1_like_output(*, paper: Any, review_like: Any) -> Dict[str, Any]:
    paper_obj = paper if isinstance(paper, dict) else {}
    review_obj = review_like if isinstance(review_like, dict) else {}

    contributions_raw = paper_obj.get("contributions")
    contributions_out: List[str] = []
    if isinstance(contributions_raw, list):
        for item in contributions_raw:
            if isinstance(item, str):
                value = item.strip()
            elif isinstance(item, dict):
                value = str(item.get("description") or item.get("name") or "").strip()
            else:
                value = str(item).strip()
            if value:
                contributions_out.append(value)

    key_terms = paper_obj.get("key_terms")
    if not isinstance(key_terms, list):
        key_terms = []
    else:
        key_terms = [str(x).strip() for x in key_terms if str(x).strip()]

    must_have_entities = paper_obj.get("must_have_entities")
    if not isinstance(must_have_entities, list):
        must_have_entities = []
    else:
        must_have_entities = [str(x).strip() for x in must_have_entities if str(x).strip()]

    novelty_claims = review_obj.get("novelty_claims")
    if not isinstance(novelty_claims, list):
        novelty_claims = []

    citations = review_obj.get("all_citations_raw")
    if not isinstance(citations, list):
        citations = []

    return {
        "paper": {
            "core_task": str(paper_obj.get("core_task", "")).strip(),
            "contributions": contributions_out,
            "key_terms": key_terms,
            "must_have_entities": must_have_entities,
        },
        "review": {
            "novelty_claims": novelty_claims,
            "all_citations_raw": citations,
        },
    }
