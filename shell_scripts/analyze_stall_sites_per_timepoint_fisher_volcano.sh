#!/bin/bash
# Per-Timepoint Fisher Volcano Plot Generator (stall_sites)
# Drives R_scripts/fisher_volcano.R on the per_timepoint_fisher CSVs
# emitted by stall_sites_non_consensus_stats.py.

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ── CONFIG ───────────────────────────────────────────────────
INPUT_DIR="./results/stall_sites/enrichment/analysis_stats"
OUTPUT_DIR="./results/stall_sites/plots/per_timepoint_fisher"
FORMAT="png"
DPI=300
# ─────────────────────────────────────────────────────────────

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "STALL SITES PER-TIMEPOINT FISHER VOLCANO PLOTS"
echo "=============================================="
echo "  Input dir:  $INPUT_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Format:     $FORMAT"
echo "  DPI:        $DPI"
echo "=============================================="

# --- AA: Per-timepoint ---
echo ""
echo "--- AA: Per-Timepoint (BWM vs Control) ---"
CMD=(Rscript R_scripts/fisher_volcano.R \
  --input "$INPUT_DIR/per_timepoint_fisher_aa.csv" \
  --outdir "$OUTPUT_DIR" \
  --level aa \
  --group-col "timepoint" \
  --comparison-label "BWM vs Control" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon: Per-timepoint ---
echo ""
echo "--- Codon: Per-Timepoint (BWM vs Control) ---"
CMD=(Rscript R_scripts/fisher_volcano.R \
  --input "$INPUT_DIR/per_timepoint_fisher_codon.csv" \
  --outdir "$OUTPUT_DIR/codon" \
  --level codon \
  --group-col "timepoint" \
  --comparison-label "BWM vs Control" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
