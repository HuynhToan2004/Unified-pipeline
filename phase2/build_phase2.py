from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import MASTER_OUTPUT, NOVELTY_OUTPUT, PHASE1_INPUT
from processor import run_phase2


def setup_logger(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2: Multi-Dimensional LLM Classification")
    parser.add_argument("--phase1-input", type=Path, default=PHASE1_INPUT)
    parser.add_argument("--master-output", type=Path, default=MASTER_OUTPUT)
    parser.add_argument("--novelty-output", type=Path, default=NOVELTY_OUTPUT)
    parser.add_argument(
        "--paper-context-dir",
        type=Path,
        default=None,
        help="Optional dir containing {paper_id}.txt or .json with paper_abstract_intro",
    )
    parser.add_argument("--llm-provider", type=str, choices=["gemini", "azure", "heuristic"], default="gemini")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()
    log = setup_logger(verbose=args.verbose)

    run_phase2(
        phase1_jsonl=args.phase1_input,
        master_output_jsonl=args.master_output,
        novelty_output_jsonl=args.novelty_output,
        paper_context_dir=args.paper_context_dir,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        logger=log,
    )


if __name__ == "__main__":
    main()
