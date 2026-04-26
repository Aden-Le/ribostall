#!/bin/bash
#----------------------------------------------------
# Bash script: run merge_global_occupancy_analysis.py
# Concatenates per-site analysis CSVs in
# results/global_occupancy/analysis/{E,P,A}/ into a
# single CSV per analysis at
# results/global_occupancy/analysis_corrected/, adding
# a 'site' column. The merged CSVs are what the
# global_occupancy R plotting scripts now consume.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
ANALYSIS_DIR="./results/global_occupancy/analysis"
OUT_DIR="./results/global_occupancy/analysis_corrected"
SITES=(E P A)
# ===============================================

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "MERGE GLOBAL OCCUPANCY ANALYSIS CSVS"
echo "=============================================="
echo "Analysis directory: $ANALYSIS_DIR"
echo "Output directory:   $OUT_DIR"
echo "Sites:              ${SITES[*]}"
echo "=============================================="

CMD=(python3 scripts/merge_global_occupancy_analysis.py \
  --analysis-dir "$ANALYSIS_DIR" \
  --out-dir "$OUT_DIR" \
  --sites "${SITES[@]}")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done. Merged CSVs in $OUT_DIR"
date
echo "=============================================="
