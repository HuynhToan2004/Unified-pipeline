from __future__ import annotations

import json
from typing import Any, Dict, List


def build_novelty_judgement_prompt(
    *,
    paper_core_task: str,
    contributions: List[str],
    novelty_claims: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> str:
    payload = {
        "paper_core_task": paper_core_task,
        "contributions": contributions,
        "novelty_claims": novelty_claims,
        "candidate_papers": candidates,
    }

    return f"""
You are a strict novelty assessor for AI research papers.

Task:
Assess novelty risk by comparing target paper claims against candidate related works.

Output STRICT JSON:
{{
  "overall_novelty_risk": "low|medium|high",
  "summary": "2-4 concise sentences",
  "supporting_findings": [
    {{
      "claim_id": "s1",
      "risk": "low|medium|high",
      "matched_candidate_title": "...",
      "reason": "..."
    }}
  ],
  "top_similar_papers": [
    {{"title": "...", "year": 2023, "reason": "..."}}
  ]
}}

Rules:
- Use only given candidate abstracts/titles and novelty claims.
- Do not hallucinate missing bibliographic fields.
- If evidence is weak, prefer medium risk with explicit uncertainty.
- No markdown, no commentary.

Input JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
