from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .atomizer import LLMAtomizer
    from .config import HUMAN_META_DIR, OUTPUT_JSONL, SEA_REVIEWS_DIR, STANDARD_SECTIONS
    from .parsers import collect_reviewer_sections, load_json, load_text
    from .sentence_extractor import SpacySentenceExtractor
except Exception:
    from atomizer import LLMAtomizer
    from config import HUMAN_META_DIR, OUTPUT_JSONL, SEA_REVIEWS_DIR, STANDARD_SECTIONS
    from parsers import collect_reviewer_sections, load_json, load_text
    from sentence_extractor import SpacySentenceExtractor


def setup_logger(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def list_paper_ids(human_meta_dir: Path, sea_dir: Path) -> List[str]:
    human_ids = {p.stem for p in human_meta_dir.glob("*.json")}
    sea_ids = {p.stem for p in sea_dir.glob("*.txt")}
    return sorted(human_ids.intersection(sea_ids))


def build_record(
    *,
    paper_id: str,
    reviewer_id: str,
    reviewer_type: str,
    sections: Dict[str, str],
    sentence_extractor: SpacySentenceExtractor,
    atomizer: LLMAtomizer,
) -> Dict[str, object]:
    parsed_sections: Dict[str, object] = {}

    for section_name in STANDARD_SECTIONS:
        raw_text = (sections.get(section_name) or "").strip()
        if not raw_text:
            continue

        parsed_sections[section_name] = {
            "raw_text": raw_text,
            "spacy_sentences": sentence_extractor.extract(raw_text),
            "llm_atomic_arguments": atomizer.atomize_section(section_name, raw_text),
        }

    return {
        "paper_id": paper_id,
        "reviewer_id": reviewer_id,
        "reviewer_type": reviewer_type,
        "parsed_sections": parsed_sections,
    }


def run_phase1(
    *,
    human_meta_dir: Path,
    sea_dir: Path,
    output_jsonl: Path,
    paper_ids: Optional[List[str]] = None,
    llm_provider: str = "gemini",
    llm_model: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, object]]:
    log = logger or logging.getLogger(__name__)

    if paper_ids is None:
        paper_ids = list_paper_ids(human_meta_dir, sea_dir)

    sentence_extractor = SpacySentenceExtractor()
    atomizer = LLMAtomizer(provider=llm_provider, model=llm_model)

    rows: List[Dict[str, object]] = []

    for paper_id in paper_ids:
        human_path = human_meta_dir / f"{paper_id}.json"
        sea_path = sea_dir / f"{paper_id}.txt"

        if not human_path.exists() or not sea_path.exists():
            log.warning("Skip %s because input files are missing", paper_id)
            continue

        human_meta_json = load_json(human_path)
        llm_raw_text = load_text(sea_path)

        reviewer_records = collect_reviewer_sections(human_meta_json, llm_raw_text)
        for reviewer_id, reviewer_type, sections in reviewer_records:
            rows.append(
                build_record(
                    paper_id=paper_id,
                    reviewer_id=reviewer_id,
                    reviewer_type=reviewer_type,
                    sections=sections,
                    sentence_extractor=sentence_extractor,
                    atomizer=atomizer,
                )
            )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log.info("Phase1 completed. records=%s output=%s llm_mode=%s", len(rows), output_jsonl, atomizer.mode)
    return rows


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1: Universal Preprocessing & Atomization")
    parser.add_argument("--human-meta-dir", type=Path, default=HUMAN_META_DIR)
    parser.add_argument("--sea-dir", type=Path, default=SEA_REVIEWS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--paper-id", action="append", default=None, help="can pass multiple times")
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["gemini", "azure", "heuristic"],
        default="gemini",
        help="LLM backend for atomization",
    )
    parser.add_argument("--llm-model", type=str, default=None, help="Optional provider-specific model/deployment name")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()
    log = setup_logger(verbose=args.verbose)

    run_phase1(
        human_meta_dir=args.human_meta_dir,
        sea_dir=args.sea_dir,
        output_jsonl=args.output,
        paper_ids=args.paper_id,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        logger=log,
    )


if __name__ == "__main__":
    main()
