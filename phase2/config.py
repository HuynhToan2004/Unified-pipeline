from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE1_INPUT = PROJECT_ROOT / "phase1" / "Phase1.jsonl"
MASTER_OUTPUT = PROJECT_ROOT / "phase2" / "output" / "Master_Classification.jsonl"
NOVELTY_OUTPUT = PROJECT_ROOT / "phase2" / "output" / "Novelty_Verification_Targets.jsonl"

SEVERITY_MAP = {
    "Fatal": 3,
    "Major": 2,
    "Minor": 1,
    "None": 0,
}

MACRO_TOPICS = [
    "Novelty & Contribution",
    "Reproducibility & Open Science",
    "Methodology & Theoretical Soundness",
    "Related work & Citations",
    "Experimental Design & Evaluation",
    "Applicability, Scalability & Limitations",
    "Clarity & Presentation",
]

TAXONOMY = {
    "Novelty & Contribution": [
        "Limited Novelty",
        "Incremental Contribution Only",
        "Lack of Significance/Impact",
    ],
    "Clarity & Presentation": [
        "General writing & Clarity issues",
        "Unclear Math/ Notations",
        "Poor Figures/ Tables quality",
        "Grammar & typos",
    ],
    "Applicability, Scalability & Limitations": [
        "General Applicability Issues",
        "Scalability & Complexity Concerns",
        "Lack of Discussion on Limitations",
        "Missing Broader Impact/ Ethical Concerns",
    ],
    "Experimental Design & Evaluation": [
        "Missing/ weak Baselines",
        "Insufficient Experimental Validation",
        "Questionable Evaluation Metrics",
        "Limited/Biased Datasets",
    ],
    "Related work & Citations": [
        "Missing Comparisons with Prior Work",
        "Missing Relevant Citations",
        "Missing Recent/ Concurrent Works",
    ],
    "Methodology & Theoretical Soundness": [
        "Weak Theoretical Justification/Proofs",
        "Methodological Flaws",
        "Strong/Unrealistic Assumptions",
        "Lack of Intuition/Justification",
    ],
    "Reproducibility & Open Science": [
        "General Reproducibility Concerns",
        "Insufficient Implementation Details",
        "Missing Code/Data Repository",
    ],
}

ALL_MICRO_FLAWS = [f for values in TAXONOMY.values() for f in values]
MICRO_TO_MACRO = {micro: macro for macro, micros in TAXONOMY.items() for micro in micros}
