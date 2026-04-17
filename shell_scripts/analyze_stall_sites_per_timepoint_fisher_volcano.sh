#!/bin/bash
# Per-Timepoint Fisher Volcano Plot Generator
# Launches the R script to create volcano plots from per_timepoint_fisher.csv

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ── CONFIG ───────────────────────────────────────────────────
INPUT_CSV="./results/stall_sites/enrichment/per_timepoint_fisher.csv"
OUTPUT_DIR="./results/stall_sites/plots/per_timepoint_fisher"
FORMAT="png"
DPI=300
# ─────────────────────────────────────────────────────────────

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "PER-TIMEPOINT FISHER VOLCANO PLOTS"
echo "=============================================="
echo "  Input:  $INPUT_CSV"
echo "  Output: $OUTPUT_DIR"
echo "  Format: $FORMAT"
echo "  DPI:    $DPI"
echo "=============================================="

CMD=(Rscript R_scripts/stall_sites_per_timepoint_fisher_volcano.R \
  --input "$INPUT_CSV" \
  --outdir "$OUTPUT_DIR" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
