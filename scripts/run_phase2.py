#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from unified_pipeline.phase2 import run_phase2


def setup_logger(verbose: bool) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Unified Pipeline Phase 2")
    parser.add_argument(
        "--phase1-input",
        type=Path,
        default=PROJECT_ROOT / "phase1/Phase1.jsonl",
        help="Input Phase1 JSONL",
    )
    parser.add_argument(
        "--master-output",
        type=Path,
        default=PROJECT_ROOT / "phase2/Master_Classification.jsonl",
        help="Output master classification JSONL",
    )
    parser.add_argument(
        "--novelty-output",
        type=Path,
        default=PROJECT_ROOT / "phase2/Novelty_Verification_Targets.jsonl",
        help="Output novelty targets JSONL",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = setup_logger(args.verbose)

    master, novelty = run_phase2(
        phase1_jsonl=args.phase1_input,
        master_output_jsonl=args.master_output,
        novelty_output_jsonl=args.novelty_output,
        logger=log,
    )
    log.info("Done Phase2. master=%s novelty=%s", len(master), len(novelty))


if __name__ == "__main__":
    main()
