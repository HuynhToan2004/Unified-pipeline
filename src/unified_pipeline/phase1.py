from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import spacy  # type: ignore
except Exception:
    spacy = None

from .constants import STANDARD_SECTIONS
from .io_utils import (
    get_human_meta_records,
    list_paper_ids,
    parse_sections_from_markdown_like,
    read_json,
    read_text,
    reviewer_type_from_id,
    write_jsonl,
)
from .llm_client import UnifiedLLMClient, basic_atomize


class Phase1Processor:
    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None, logger: Optional[logging.Logger] = None):
        self.llm = llm_client or UnifiedLLMClient()
        self.log = logger or logging.getLogger(__name__)
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.blank("en")
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
            except Exception:
                self.nlp = None

    def parse_spacy_sentences(self, text: str) -> List[Dict[str, str]]:
        out = []
        if self.nlp is not None:
            doc = self.nlp(text)
            for idx, sent in enumerate(doc.sents, start=1):
                s = sent.text.strip()
                if s:
                    out.append({"sent_id": f"s{idx}", "text": s})
            return out

        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        for idx, part in enumerate(parts, start=1):
            p = part.strip()
            if p:
                out.append({"sent_id": f"s{idx}", "text": p})
        return out

    def atomize_with_llm(self, section_name: str, section_text: str) -> List[Dict[str, str]]:
        fallback = {
            "atomic_arguments": basic_atomize(section_text),
        }
        prompt = f"""
You are extracting atomic arguments from a peer-review section.
Return STRICT JSON only:
{{"atomic_arguments": ["...", "..."]}}

Rules:
- Keep each argument independent and minimal.
- Preserve meaning, remove redundancy.
- Do not add facts.

Section: {section_name}
Text:
{section_text}
""".strip()
        response = self.llm.generate_json(prompt=prompt, fallback=fallback)
        args = response.get("atomic_arguments") if isinstance(response, dict) else None
        if not isinstance(args, list):
            args = fallback["atomic_arguments"]

        out = []
        for idx, value in enumerate(args, start=1):
            text = str(value).strip()
            if text:
                out.append({"arg_id": f"a{idx}", "text": text})
        return out

    def build_record(self, paper_id: str, reviewer_id: str, sections: Dict[str, str]) -> Dict[str, Any]:
        parsed_sections: Dict[str, Any] = {}

        for sec in STANDARD_SECTIONS:
            raw = (sections.get(sec) or "").strip()
            if not raw:
                continue
            parsed_sections[sec] = {
                "raw_text": raw,
                "spacy_sentences": self.parse_spacy_sentences(raw),
                "llm_atomic_arguments": self.atomize_with_llm(sec, raw),
            }

        return {
            "paper_id": paper_id,
            "reviewer_id": reviewer_id,
            "reviewer_type": reviewer_type_from_id(reviewer_id),
            "parsed_sections": parsed_sections,
        }


def run_phase1(
    *,
    human_meta_dir: Path,
    sea_dir: Path,
    output_jsonl: Path,
    paper_ids: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    log = logger or logging.getLogger(__name__)
    processor = Phase1Processor(logger=log)

    if paper_ids is None:
        paper_ids = list_paper_ids(human_meta_dir, sea_dir)

    rows: List[Dict[str, Any]] = []

    for paper_id in paper_ids:
        human_path = human_meta_dir / f"{paper_id}.json"
        sea_path = sea_dir / f"{paper_id}.txt"
        if not human_path.exists() or not sea_path.exists():
            log.warning("Skip %s due to missing input file(s)", paper_id)
            continue

        human_meta_json = read_json(human_path)
        sea_raw = read_text(sea_path)

        for reviewer_id, sections in get_human_meta_records(human_meta_json):
            rows.append(processor.build_record(paper_id, reviewer_id, sections))

        llm_sections = parse_sections_from_markdown_like(sea_raw)
        rows.append(processor.build_record(paper_id, "LLM_SEA", llm_sections))

    write_jsonl(output_jsonl, rows)
    log.info("Phase 1 done. records=%s output=%s llm_mode=%s", len(rows), output_jsonl, processor.llm.mode)
    return rows
