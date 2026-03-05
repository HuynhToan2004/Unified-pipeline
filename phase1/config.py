from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
HUMAN_META_DIR = DATASET_DIR / "Human_and_meta_reviews"
SEA_REVIEWS_DIR = DATASET_DIR / "SEA_reviews"

OUTPUT_JSONL = PROJECT_ROOT / "phase1" / "Phase1.jsonl"

STANDARD_SECTIONS = ["Summary", "Strengths", "Weaknesses", "Questions"]

SECTION_ALIASES = {
    "summary": "Summary",
    "strengths": "Strengths",
    "weaknesses": "Weaknesses",
    "questions": "Questions",
}
