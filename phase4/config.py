from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE2_MASTER_INPUT = PROJECT_ROOT / "phase2" / "output" / "Master_Classification.jsonl"
PHASE3_RETRIEVAL_INPUT = PROJECT_ROOT / "phase3" / "output" / "Novelty_Assessment_Report.json"

OUTPUT_DIR = PROJECT_ROOT / "phase4" / "output"
PHASE4_REPORT_JSON = OUTPUT_DIR / "Phase4_Metrics_Report.json"
PHASE4_REPORT_MD = OUTPUT_DIR / "Phase4_Metrics_Report.md"

SEVERITY_MAP = {
    "Fatal": 3.0,
    "Major": 2.0,
    "Minor": 1.0,
    "None": 0.0,
}
