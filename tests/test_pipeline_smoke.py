from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from unified_pipeline.phase1 import run_phase1
from unified_pipeline.phase2 import run_phase2


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_phase1_phase2_single_paper(tmp_path: Path):
    phase1_out = tmp_path / "Phase1.jsonl"
    master_out = tmp_path / "Master_Classification.jsonl"
    novelty_out = tmp_path / "Novelty_Verification_Targets.jsonl"

    rows1 = run_phase1(
        human_meta_dir=PROJECT_ROOT / "dataset" / "Human_and_meta_reviews",
        sea_dir=PROJECT_ROOT / "dataset" / "SEA_reviews",
        output_jsonl=phase1_out,
        paper_ids=["0A5o6dCKeK"],
    )

    assert phase1_out.exists()
    assert len(rows1) == 6

    rows1_file = _read_jsonl(phase1_out)
    assert len(rows1_file) == 6
    assert {r["reviewer_id"] for r in rows1_file} == {"Meta", "Human_1", "Human_2", "Human_3", "Human_4", "LLM_SEA"}

    master_rows, novelty_rows = run_phase2(
        phase1_jsonl=phase1_out,
        master_output_jsonl=master_out,
        novelty_output_jsonl=novelty_out,
    )

    assert master_out.exists()
    assert novelty_out.exists()
    assert len(master_rows) == 6
    assert len(novelty_rows) == 6

    master_file_rows = _read_jsonl(master_out)
    novelty_file_rows = _read_jsonl(novelty_out)
    assert len(master_file_rows) == 6
    assert len(novelty_file_rows) == 6

    sample_master = master_file_rows[0]
    assert "parsed_sections" in sample_master

    sample_novelty = novelty_file_rows[0]
    assert "paper" in sample_novelty
    assert "review_novelty_extracted" in sample_novelty
