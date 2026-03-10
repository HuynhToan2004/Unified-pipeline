#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_gemini.sh — Run the full Unified Pipeline using Google Gemini
# Usage:
#   ./run_gemini.sh                          # process all papers
#   ./run_gemini.sh --paper-id 0A5o6dCKeK   # single paper
#   ./run_gemini.sh --paper-id A --paper-id B -v   # multiple papers, verbose
#   ./run_gemini.sh --disable-semantic-api   # skip Semantic Scholar
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Move to the project root (regardless of where the script is called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env into environment
if [[ -f ".env" ]]; then
    set -a
    source ".env"
    set +a
    echo "[INFO] Loaded .env"
else
    echo "[WARN] No .env file found — using existing environment variables"
fi

# Activate venv if present
if [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
    echo "[INFO] Activated .venv"
fi

PYTHON="${PYTHON:-python}"

echo "[INFO] Using Python: $($PYTHON --version)"
echo "[INFO] LLM Provider: ${LLM_PROVIDER:-gemini}"
echo "[INFO] Gemini Model: ${GEMINI_MODEL:-models/gemini-2.5-flash-lite}"
echo ""

# ─── Data directories ─────────────────────────────────────────────────────────
# The pipeline needs two parallel datasets with matching paper IDs:
#   HUMAN_META_DIR : one JSON per paper  (keys: "Meta review", "reviews")
#   SEA_DIR        : one TXT  per paper  (SEA LLM review in markdown format)
HUMAN_META_DIR="${HUMAN_META_DIR:-/home/25thanh.tk/paper_data/ICLR_2024/review_json}"
SEA_DIR="${SEA_DIR:-/home/25thanh.tk/SEA/iclr_review_txt}"

echo "[INFO] Human+Meta reviews: $HUMAN_META_DIR"
echo "[INFO] SEA reviews       : $SEA_DIR"
echo ""

# Run unified pipeline — pass all extra args straight through
$PYTHON scripts/run_unified_pipeline.py \
    --phase1-llm-provider gemini \
    --phase2-llm-provider gemini \
    --human-meta-dir "$HUMAN_META_DIR" \
    --sea-dir        "$SEA_DIR" \
    "$@"
