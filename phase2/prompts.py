from __future__ import annotations

import json
from typing import Any, Dict, List

from config import ALL_MICRO_FLAWS, MACRO_TOPICS, TAXONOMY


def build_record_level_prompt(
    *,
    paper_id: str,
    reviewer_id: str,
    reviewer_type: str,
    parsed_sections: Dict[str, Any],
    paper_abstract_intro: str,
    review_context: str,
) -> str:
    taxonomy_json = json.dumps(TAXONOMY, ensure_ascii=False, indent=2)
    payload = {
        "paper_id": paper_id,
        "reviewer_id": reviewer_id,
        "reviewer_type": reviewer_type,
        "parsed_sections": parsed_sections,
        "paper_abstract_intro": paper_abstract_intro,
        "review_context": review_context,
    }

    return f"""
You are an expert Area Chair and scientific-review analyst.

Task: In ONE PASS, classify all review units and extract novelty verification targets.

You must output STRICT JSON with EXACT top-level keys:
{{
  "parsed_sections": {{
    "<SectionName>": {{
      "spacy_sentences_classification": [
        {{"sent_id": "s1", "text": "...", "argument_role": "Claim|Premise", "aspect_macro_topic": "..."}}
      ],
      "llm_atomic_arguments_classification": [
        {{"arg_id": "a1", "text": "...", "micro_flaw_label": "...", "severity": "Fatal|Major|Minor|None"}}
      ]
    }}
  }},
  "paper": {{
    "core_task": "...",
    "contributions": [
      {{"name": "...", "author_claim_text": "...", "description": "...", "source_hint": "..."}}
    ],
    "key_terms": ["..."],
    "must_have_entities": ["..."]
  }},
  "review_novelty_extracted": {{
    "novelty_claims": [
      {{
        "claim_id": "s1",
        "text": "...",
        "stance": "not_novel|somewhat_novel|novel|unclear",
        "confidence_lang": "high|medium|low",
        "mentions_prior_work": true,
        "prior_work_strings": ["..."],
        "evidence_expected": "method_similarity|task_similarity|results_similarity|theory_overlap|dataset_overlap"
      }}
    ],
    "all_citations_raw": ["..."]
  }}
}}

Allowed `aspect_macro_topic` values only:
{json.dumps(MACRO_TOPICS, ensure_ascii=False)}

Allowed micro-flaw taxonomy labels only:
{taxonomy_json}

Severity protocol:
- Fatal: critical flaw invalidating core claims.
- Major: specific serious weakness with concrete evidence.
- Minor: generic/vague critique, clarity issues, or ordinary question.
- None: non-critique summary/praise/neutral.

Hard constraints:
- Classify ALL input spacy sentences and ALL input atomic arguments.
- Keep original `sent_id`, `arg_id`, and `text` unchanged.
- `novelty_claims.text` should be verbatim or near-verbatim from review_context.
- `paper.contributions` must contain 1-3 items.
- Return ONE valid JSON object only, no markdown.

Input JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def build_section_classification_prompt(
    *,
    paper_id: str,
    reviewer_id: str,
    reviewer_type: str,
    section_name: str,
    section_text: str,
    spacy_sentences: List[Dict[str, Any]],
    atomic_arguments: List[Dict[str, Any]],
    review_context: str,
) -> str:
    taxonomy_json = json.dumps(TAXONOMY, ensure_ascii=False, indent=2)
    payload = {
        "paper_id": paper_id,
        "reviewer_id": reviewer_id,
        "reviewer_type": reviewer_type,
        "section": section_name,
        "section_text": section_text,
        "spacy_sentences": spacy_sentences,
        "atomic_arguments": atomic_arguments,
        "review_context": review_context,
    }

    return f"""
You are an expert meta-review analyst for AI conference reviews.

Task: Perform MULTI-DIMENSIONAL classification in one pass.
For the provided section, classify:
1) For each spacy sentence: argument_role + aspect_macro_topic.
2) For each atomic argument: micro_flaw_label + severity.

Allowed argument_role values:
- Claim
- Premise

Allowed aspect_macro_topic values (EXACT STRING):
{json.dumps(MACRO_TOPICS, ensure_ascii=False)}

Micro-flaw taxonomy (EXACT STRING labels only):
{taxonomy_json}

Severity definition (EXACT STRING):
- Fatal: Critical flaw invalidating core claims.
- Major: Specific significant weakness with concrete evidence/actionable gap.
- Minor: Generic, vague, presentation/clarification, or weakly evidenced critique.
- None: Non-critique (summary/praise/neutral).

STRICT OUTPUT JSON format:
{{
  "spacy_sentences_classification": [
    {{"sent_id": "s1", "text": "...", "argument_role": "Claim", "aspect_macro_topic": "Experimental Design & Evaluation"}}
  ],
  "llm_atomic_arguments_classification": [
    {{"arg_id": "a1", "text": "...", "micro_flaw_label": "Missing/ weak Baselines", "severity": "Major"}}
  ]
}}

Rules:
- Output must classify ALL input sentences and ALL input atomic arguments.
- Keep `sent_id` and `arg_id` unchanged.
- Use ONLY the allowed labels.
- No markdown, no explanation, output ONE valid JSON object.

Input JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def build_novelty_targets_prompt(
    *,
    paper_id: str,
    reviewer_id: str,
    reviewer_type: str,
    paper_abstract_intro: str,
    review_context: str,
) -> str:
    payload = {
        "paper_id": paper_id,
        "reviewer_id": reviewer_id,
        "reviewer_type": reviewer_type,
        "paper_abstract_intro": paper_abstract_intro,
        "review_context": review_context,
    }

    return f"""
You are an expert for novelty verification target extraction.

Goal: Build structured novelty targets from PAPER ABSTRACT+INTRO and REVIEW CONTEXT.

Output STRICT JSON with EXACT top-level keys:
{{
  "paper": {{
    "core_task": "...",
    "contributions": [
      {{"name": "...", "author_claim_text": "...", "description": "...", "source_hint": "..."}}
    ],
    "key_terms": ["..."],
    "must_have_entities": ["..."]
  }},
  "review_novelty_extracted": {{
    "novelty_claims": [
      {{
        "claim_id": "s1",
        "text": "verbatim claim from review_context",
        "stance": "not_novel|somewhat_novel|novel|unclear",
        "confidence_lang": "high|medium|low",
        "mentions_prior_work": true,
        "prior_work_strings": ["..."],
        "evidence_expected": "method_similarity|task_similarity|results_similarity|theory_overlap|dataset_overlap"
      }}
    ],
    "all_citations_raw": ["..."]
  }}
}}

Requirements:
- `core_task`: 8-20 words, specific and concrete.
- `contributions`: 1-3 items.
- `novelty_claims.text` must be verbatim or near-verbatim from review_context.
- Extract prior-work and citation-like strings only if present in review_context.
- If no novelty claim exists, return empty novelty_claims list.
- No markdown, no explanation.

Input JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
