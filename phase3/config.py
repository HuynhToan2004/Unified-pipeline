from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE2_MASTER_INPUT = PROJECT_ROOT / "phase2" / "output" / "Master_Classification.jsonl"
PHASE2_NOVELTY_INPUT = PROJECT_ROOT / "phase2" / "output" / "Novelty_Verification_Targets.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "phase3" / "output"
NOVELTY_REPORT_JSON = OUTPUT_DIR / "Novelty_Assessment_Report.json"
NOVELTY_REPORT_MD = OUTPUT_DIR / "Novelty_Assessment_Report.md"

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_TIMEOUT = 20
SEMANTIC_DEFAULT_LIMIT = 8

LLM_PROVIDER_DEFAULT = "heuristic"  # heuristic | openai | vllm | azure | gemini
LLM_MAX_CANDIDATES_PER_QUERY = 8
