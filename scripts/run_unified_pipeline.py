#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logger(verbose: bool) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Unified Pipeline (Phase1 -> Phase2 -> Phase3 -> Phase4)")
    parser.add_argument("--human-meta-dir", type=Path, default=PROJECT_ROOT / "dataset/Human_and_meta_reviews")
    parser.add_argument("--sea-dir", type=Path, default=PROJECT_ROOT / "dataset/SEA_reviews")
    parser.add_argument("--phase1-output", type=Path, default=PROJECT_ROOT / "phase1/Phase1.jsonl")
    parser.add_argument("--master-output", type=Path, default=PROJECT_ROOT / "phase2/output/Master_Classification.jsonl")
    parser.add_argument("--novelty-output", type=Path, default=PROJECT_ROOT / "phase2/output/Novelty_Verification_Targets.jsonl")
    parser.add_argument("--phase3-output-json", type=Path, default=PROJECT_ROOT / "phase3/output/Novelty_Assessment_Report.json")
    parser.add_argument("--phase3-output-md", type=Path, default=PROJECT_ROOT / "phase3/output/Novelty_Assessment_Report.md")
    parser.add_argument("--phase4-output-json", type=Path, default=PROJECT_ROOT / "phase4/output/Phase4_Metrics_Report.json")
    parser.add_argument("--phase4-output-md", type=Path, default=PROJECT_ROOT / "phase4/output/Phase4_Metrics_Report.md")
    parser.add_argument("--paper-id", action="append", default=None)
    parser.add_argument("--phase1-llm-provider", choices=["gemini", "azure", "heuristic"], default="gemini")
    parser.add_argument("--phase1-llm-model", type=str, default=None)
    parser.add_argument("--phase2-llm-provider", choices=["gemini", "azure", "heuristic"], default="gemini")
    parser.add_argument("--phase2-llm-model", type=str, default=None)
    parser.add_argument("--paper-context-dir", type=Path, default=None)
    parser.add_argument("--max-candidates-per-query", type=int, default=8)
    parser.add_argument("--disable-semantic-api", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def _run_step(*, cmd: list[str], env: dict[str, str], log: logging.Logger) -> None:
    log.info("Run: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


def main() -> None:
    args = parse_args()
    log = setup_logger(args.verbose)

    python_exe = sys.executable
    run_env = os.environ.copy()
    if args.disable_semantic_api:
        run_env["DISABLE_SEMANTIC_API"] = "1"

    phase1_cmd = [
        python_exe,
        str(PROJECT_ROOT / "phase1" / "run_phase1.py"),
        "--human-meta-dir",
        str(args.human_meta_dir),
        "--sea-dir",
        str(args.sea_dir),
        "--output",
        str(args.phase1_output),
        "--llm-provider",
        args.phase1_llm_provider,
    ]
    if args.phase1_llm_model:
        phase1_cmd.extend(["--llm-model", args.phase1_llm_model])
    if args.paper_id:
        for item in args.paper_id:
            phase1_cmd.extend(["--paper-id", str(item)])
    if args.verbose:
        phase1_cmd.append("-v")
    _run_step(cmd=phase1_cmd, env=run_env, log=log)

    phase2_cmd = [
        python_exe,
        str(PROJECT_ROOT / "phase2" / "run_phase2.py"),
        "--phase1-input",
        str(args.phase1_output),
        "--master-output",
        str(args.master_output),
        "--novelty-output",
        str(args.novelty_output),
        "--llm-provider",
        args.phase2_llm_provider,
    ]
    if args.phase2_llm_model:
        phase2_cmd.extend(["--llm-model", args.phase2_llm_model])
    if args.paper_context_dir:
        phase2_cmd.extend(["--paper-context-dir", str(args.paper_context_dir)])
    if args.verbose:
        phase2_cmd.append("-v")
    _run_step(cmd=phase2_cmd, env=run_env, log=log)

    phase3_cmd = [
        python_exe,
        str(PROJECT_ROOT / "phase3" / "run_phase3.py"),
        "--phase2-novelty-input",
        str(args.novelty_output),
        "--output-json",
        str(args.phase3_output_json),
        "--output-md",
        str(args.phase3_output_md),
        "--max-candidates-per-query",
        str(args.max_candidates_per_query),
    ]
    if args.verbose:
        phase3_cmd.append("-v")
    _run_step(cmd=phase3_cmd, env=run_env, log=log)

    phase4_cmd = [
        python_exe,
        str(PROJECT_ROOT / "phase4" / "run_phase4.py"),
        "--phase2-master-input",
        str(args.master_output),
        "--phase3-retrieval-input",
        str(args.phase3_output_json),
        "--output-json",
        str(args.phase4_output_json),
        "--output-md",
        str(args.phase4_output_md),
    ]
    if args.verbose:
        phase4_cmd.append("-v")
    _run_step(cmd=phase4_cmd, env=run_env, log=log)

    log.info("Unified pipeline completed through Phase4.")


if __name__ == "__main__":
    main()
