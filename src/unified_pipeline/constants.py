from __future__ import annotations

STANDARD_SECTIONS = [
    "Summary",
    "Strengths",
    "Weaknesses",
    "Questions",
]

MACRO_TOPICS = [
    "Novelty & Contribution",
    "Reproducibility & Open Science",
    "Methodology & Theoretical Soundness",
    "Related work & Citations",
    "Experimental Design & Evaluation",
    "Applicability, Scalability & Limitations",
    "Clarity & Presentation",
]

MICRO_FLAWS_BY_MACRO = {
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

ALL_MICRO_FLAWS = [item for items in MICRO_FLAWS_BY_MACRO.values() for item in items]

SEVERITY_MAP = {
    "Fatal": 3,
    "Major": 2,
    "Minor": 1,
    "None": 0,
}

SEVERITY_LEVELS = list(SEVERITY_MAP.keys())

NOVELTY_CUE_WORDS = [
    "novel",
    "novelty",
    "not novel",
    "incremental",
    "limited novelty",
    "prior work",
    "previous work",
    "already",
    "similar to",
]
