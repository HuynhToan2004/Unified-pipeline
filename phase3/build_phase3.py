from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import NOVELTY_REPORT_JSON, NOVELTY_REPORT_MD, PHASE2_NOVELTY_INPUT
from processor import run_phase3_novelty


def setup_logger(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase3 novelty retrieval-only pipeline")
    parser.add_argument("--phase2-novelty-input", type=Path, default=PHASE2_NOVELTY_INPUT)
    parser.add_argument("--output-json", type=Path, default=NOVELTY_REPORT_JSON)
    parser.add_argument("--output-md", type=Path, default=NOVELTY_REPORT_MD)
    parser.add_argument("--max-candidates-per-query", type=int, default=8)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()
    log = setup_logger(verbose=args.verbose)

    run_phase3_novelty(
        phase2_novelty_input=args.phase2_novelty_input,
        output_json=args.output_json,
        output_md=args.output_md,
        max_candidates_per_query=args.max_candidates_per_query,
        logger=log,
    )


if __name__ == "__main__":
    main()
