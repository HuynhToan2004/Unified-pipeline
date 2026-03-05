from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_review_context(parsed_sections: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for section_name, payload in (parsed_sections or {}).items():
        raw_text = str((payload or {}).get("raw_text", "")).strip()
        if raw_text:
            chunks.append(f"[{section_name}]\n{raw_text}")
    return "\n\n".join(chunks)


def load_paper_abstract_intro(paper_id: str, paper_context_dir: Optional[Path] = None) -> str:
    if paper_context_dir is None:
        return ""

    txt_path = paper_context_dir / f"{paper_id}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")

    json_path = paper_context_dir / f"{paper_id}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ["paper_abstract_intro", "abstract_intro", "abstract", "paper_text"]:
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return ""


def extract_citation_like_strings(text: str) -> List[str]:
    patterns = [
        r"\b\d{4}\.\d{4,5}\b",
        r"https?://\S+",
        r"\b[A-Z][A-Za-z]+\s+et\s+al\.?\s*\(?20\d{2}\)?",
        r"\[[0-9]+\]",
    ]
    found: List[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, text))

    uniq: List[str] = []
    seen = set()
    for x in found:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq
