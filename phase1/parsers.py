from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .config import SECTION_ALIASES, STANDARD_SECTIONS
except Exception:
    from config import SECTION_ALIASES, STANDARD_SECTIONS


_SECTION_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(Summary|Strengths|Weaknesses|Questions)\s*:?\s*(?:\*\*)?\s*$",
    re.IGNORECASE,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return data


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def parse_sections_from_json_review(review_obj: Dict[str, Any]) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    for sec in STANDARD_SECTIONS:
        value = review_obj.get(sec)
        if isinstance(value, str) and value.strip():
            sections[sec] = value.strip()
    return sections


def parse_meta_sections(human_meta_json: Dict[str, Any]) -> Dict[str, str]:
    meta = human_meta_json.get("Meta review")
    if not isinstance(meta, dict):
        return {}

    sections = {
        "Summary": str(meta.get("Metareview", "")).strip(),
        "Weaknesses": str(meta.get("Justification For Why Not Higher Score", "")).strip(),
        "Questions": str(meta.get("Justification For Why Not Lower Score", "")).strip(),
    }

    return {
        key: value
        for key, value in sections.items()
        if value and value.lower() not in {"n/a", "na", "none"}
    }


def parse_sections_from_markdown_review(raw_review: str) -> Dict[str, str]:
    blocks: Dict[str, List[str]] = {}
    current_section = None

    for line in raw_review.splitlines():
        m = _SECTION_RE.match(line.strip())
        if m:
            section_key = SECTION_ALIASES[m.group(1).lower()]
            current_section = section_key
            blocks.setdefault(current_section, [])
            continue

        if current_section:
            blocks[current_section].append(line)

    parsed = {key: "\n".join(lines).strip() for key, lines in blocks.items()}
    return {k: v for k, v in parsed.items() if v}


def collect_reviewer_sections(human_meta_json: Dict[str, Any], llm_raw_text: str) -> List[Tuple[str, str, Dict[str, str]]]:
    results: List[Tuple[str, str, Dict[str, str]]] = []

    meta_sections = parse_meta_sections(human_meta_json)
    if meta_sections:
        results.append(("Meta", "Meta", meta_sections))

    reviews = human_meta_json.get("reviews")
    if isinstance(reviews, list):
        for idx, review_obj in enumerate(reviews, start=1):
            if not isinstance(review_obj, dict):
                continue
            sections = parse_sections_from_json_review(review_obj)
            results.append((f"Human_{idx}", "Human", sections))

    llm_sections = parse_sections_from_markdown_review(llm_raw_text)
    results.append(("LLM_SEA", "LLM", llm_sections))

    return results
