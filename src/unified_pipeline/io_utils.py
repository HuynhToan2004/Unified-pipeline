from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SECTION_HEADER_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(Summary|Strengths|Weaknesses|Questions)\s*:?\s*(?:\*\*)?\s*$",
    re.IGNORECASE,
)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_sections_from_markdown_like(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = SECTION_HEADER_RE.match(line)
        if match:
            current = match.group(1).title()
            sections.setdefault(current, [])
            continue

        if current is not None:
            sections[current].append(raw_line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def parse_sections_from_review_object(review_obj: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in ["Summary", "Strengths", "Weaknesses", "Questions"]:
        value = review_obj.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    return out


def flatten_sections_for_context(parsed_sections: Dict[str, Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for section_name, section_payload in parsed_sections.items():
        text = (section_payload or {}).get("raw_text", "")
        if text:
            chunks.append(f"[{section_name}]\n{text}")
    return "\n\n".join(chunks)


def list_paper_ids(human_meta_dir: Path, sea_dir: Path) -> List[str]:
    human = {p.stem for p in human_meta_dir.glob("*.json")}
    sea = {p.stem for p in sea_dir.glob("*.txt")}
    return sorted(human.intersection(sea))


def reviewer_type_from_id(reviewer_id: str) -> str:
    low = reviewer_id.lower()
    if low.startswith("human"):
        return "Human"
    if low.startswith("meta"):
        return "Meta"
    return "LLM"


def extract_citation_like_strings(text: str) -> List[str]:
    patterns = [
        r"\b\d{4}\.\d{4,5}\b",
        r"https?://\S+",
        r"\b[A-Z][A-Za-z]+\s+et\s+al\.?\s*\(?20\d{2}\)?",
    ]
    found: List[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, text))
    seen = set()
    unique: List[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def get_human_meta_records(human_meta_json: Dict[str, Any]) -> List[Tuple[str, Dict[str, str]]]:
    rows: List[Tuple[str, Dict[str, str]]] = []

    meta_obj = (human_meta_json.get("Meta review") or {})
    if isinstance(meta_obj, dict):
        meta_sections = {
            "Summary": str(meta_obj.get("Metareview", "")).strip(),
            "Weaknesses": str(meta_obj.get("Justification For Why Not Higher Score", "")).strip(),
            "Questions": str(meta_obj.get("Justification For Why Not Lower Score", "")).strip(),
        }
        meta_sections = {k: v for k, v in meta_sections.items() if v and v != "N/A"}
        rows.append(("Meta", meta_sections))

    reviews = human_meta_json.get("reviews") or []
    if isinstance(reviews, list):
        for idx, review_obj in enumerate(reviews, start=1):
            if not isinstance(review_obj, dict):
                continue
            rid = f"Human_{idx}"
            rows.append((rid, parse_sections_from_review_object(review_obj)))

    return rows
