#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from unified_pipeline.phase1 import run_phase1


def setup_logger(verbose: bool) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Unified Pipeline Phase 1")
    parser.add_argument(
        "--human-meta-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset/Human_and_meta_reviews",
        help="Folder containing Human + Meta review JSON files",
    )
    parser.add_argument(
        "--sea-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset/SEA_reviews",
        help="Folder containing LLM SEA review TXT files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "phase1/Phase1.jsonl",
        help="Output JSONL path for Phase 1",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=None,
        help="Run for specific paper_id(s), can pass multiple times",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = setup_logger(args.verbose)

    rows = run_phase1(
        human_meta_dir=args.human_meta_dir,
        sea_dir=args.sea_dir,
        output_jsonl=args.output,
        paper_ids=args.paper_id,
        logger=log,
    )
    log.info("Done Phase1. total records=%s", len(rows))


if __name__ == "__main__":
    main()
