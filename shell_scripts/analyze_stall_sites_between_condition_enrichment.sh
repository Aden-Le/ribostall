#!/bin/bash
#----------------------------------------------------
# Bash script: generate between-condition enrichment
# bar plots via R
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============

# Input CSV from enrichment analysis
INPUT_CSV="./results/stall_sites/enrichment/between_condition_wilcoxon_aa.csv"

# Output directory for plots
OUTPUT_DIR="./results/stall_sites/plots/between_condition"

# Output format: "pdf", "png", or "both"
FORMAT="png"

# DPI for PNG output
DPI=300

# ===============================================

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "BETWEEN-CONDITION ENRICHMENT BAR PLOTS"
echo "=============================================="
echo "Input CSV:        $INPUT_CSV"
echo "Output directory: $OUTPUT_DIR"
echo "Format:           $FORMAT"
echo "DPI:              $DPI"
echo "=============================================="

CMD=(Rscript R_scripts/stall_sites_between_condition_enrichment.R \
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
