#!/bin/bash
#----------------------------------------------------
# Bash script: generate within-condition enrichment
# volcano plots via R
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============

# Input CSV from enrichment analysis
INPUT_CSV="./enrichment_results/within_condition_enrichment.csv"

# Output directory for plots
OUTPUT_DIR="./outputs/within_condition_output"

# Enrichment type: "unweighted", "weighted", or "both"
ENRICHMENT_TYPE="both"

# Show confidence intervals: set to "--show-ci" to enable, leave empty to disable
SHOW_CI="--show-ci"

# Output format: "pdf", "png", or "both"
FORMAT="png"

# DPI for PNG output
DPI=300

# Y-axis cap: clamp -log10(p_adj) values above this to compress the y-axis
# Leave empty ("") to disable capping
Y_CAP=25

# ===============================================

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "WITHIN-CONDITION ENRICHMENT VOLCANO PLOTS"
echo "=============================================="
echo "Input CSV:        $INPUT_CSV"
echo "Output directory: $OUTPUT_DIR"
echo "Enrichment type:  $ENRICHMENT_TYPE"
echo "Format:           $FORMAT"
echo "DPI:              $DPI"
echo "Show CI:          ${SHOW_CI:-no}"
echo "Y-axis cap:       ${Y_CAP:-none}"
echo "=============================================="

# Build optional flags
OPTIONAL_FLAGS="$SHOW_CI"
if [ -n "$Y_CAP" ]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --y-cap $Y_CAP"
fi

CMD=(Rscript R_scripts/stall_sites_within_condition_enrichment.R \
  --input "$INPUT_CSV" \
  --outdir "$OUTPUT_DIR" \
  --enrichment-type "$ENRICHMENT_TYPE" \
  --format "$FORMAT" \
  --dpi "$DPI" \
  $OPTIONAL_FLAGS)

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
