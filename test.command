#!/usr/bin/env bash
# CV to JD — End-to-End Pipeline Test Runner
set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$PROJECT_DIR/data/test_run.log"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  CV ↔ JD Mapping System v2 — Pipeline Test"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "→ Running pipeline test (output also saved to data/test_run.log)..."
echo ""

python tests/demo_results.py 2>&1 | tee "$LOG"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  ✅ Done! Full log: $LOG"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
