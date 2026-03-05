from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from build_phase1 import run_phase1, setup_logger
from config import HUMAN_META_DIR, SEA_REVIEWS_DIR


def _count_atomic_args(jsonl_path: Path) -> int:
    total = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sections = row.get("parsed_sections", {})
            for sec in sections.values():
                total += len((sec or {}).get("llm_atomic_arguments", []))
    return total


def benchmark(
    providers: List[str],
    paper_ids: List[str],
    out_dir: Path,
    llm_model: str | None,
    verbose: bool,
) -> None:
    log = setup_logger(verbose=verbose)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for provider in providers:
        output_path = out_dir / f"Phase1_{provider}.jsonl"

        t0 = time.perf_counter()
        rows = run_phase1(
            human_meta_dir=HUMAN_META_DIR,
            sea_dir=SEA_REVIEWS_DIR,
            output_jsonl=output_path,
            paper_ids=paper_ids,
            llm_provider=provider,
            llm_model=llm_model,
            logger=log,
        )
        elapsed = time.perf_counter() - t0

        atomic_count = _count_atomic_args(output_path)
        results.append(
            {
                "provider": provider,
                "records": len(rows),
                "atomic_arguments": atomic_count,
                "seconds": round(elapsed, 3),
                "output": str(output_path),
            }
        )

    summary_path = out_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(json.dumps({"results": results}, indent=2, ensure_ascii=False))
    print(f"Saved summary to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Phase1 providers")
    parser.add_argument("--providers", nargs="+", default=["heuristic", "azure", "gemini"])
    parser.add_argument("--paper-id", action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("output") / "benchmarks")
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    benchmark(
        providers=args.providers,
        paper_ids=args.paper_id or ["0A5o6dCKeK"],
        out_dir=args.out_dir,
        llm_model=args.llm_model,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
