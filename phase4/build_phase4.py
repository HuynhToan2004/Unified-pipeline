from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import PHASE2_MASTER_INPUT, PHASE3_RETRIEVAL_INPUT, PHASE4_REPORT_JSON, PHASE4_REPORT_MD
from processor import run_phase4_metrics


def setup_logger(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase4 metrics (CFI/CSP/Novelty)")
    parser.add_argument("--phase2-master-input", type=Path, default=PHASE2_MASTER_INPUT)
    parser.add_argument("--phase3-retrieval-input", type=Path, default=PHASE3_RETRIEVAL_INPUT)
    parser.add_argument("--output-json", type=Path, default=PHASE4_REPORT_JSON)
    parser.add_argument("--output-md", type=Path, default=PHASE4_REPORT_MD)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_cli_parser().parse_args()
    logger = setup_logger(verbose=args.verbose)
    run_phase4_metrics(
        phase2_master_input=args.phase2_master_input,
        phase3_retrieval_input=args.phase3_retrieval_input,
        output_json=args.output_json,
        output_md=args.output_md,
        logger=logger,
    )


if __name__ == "__main__":
    main()
