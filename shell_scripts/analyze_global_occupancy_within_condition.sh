#!/bin/bash
#----------------------------------------------------
# Bash script: generate within-condition bar plots
# for global codon and amino acid occupancy enrichment
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============

# Input directory containing the binomial CSVs
INPUT_DIR="./global_occupancy_results/analysis"

# Output directory for plots
OUTPUT_DIR="./global_occupancy_results/within_condition_output"

# Enrichment type: "unweighted", "weighted", or "both"
ENRICHMENT_TYPE="both"

# Output format: "pdf", "png", or "both"
FORMAT="png"

# DPI for PNG output
DPI=300

# Y-axis cap: clamp y-axis to ±this value
# Leave empty ("") to disable
Y_CAP=""

# ===============================================

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "GLOBAL OCCUPANCY WITHIN-CONDITION BAR PLOTS"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Enrichment type:  $ENRICHMENT_TYPE"
echo "Format:           $FORMAT"
echo "DPI:              $DPI"
echo "Y-axis cap:       ${Y_CAP:-none}"
echo "=============================================="

# Build optional flags
OPTIONAL_FLAGS=()
if [ -n "$Y_CAP" ]; then
  OPTIONAL_FLAGS+=(--y-cap "$Y_CAP")
fi

# --- Amino acid level ---
echo ""
echo "--- Amino Acid Level ---"
CMD=(Rscript R_scripts/global_occupancy_within_condition.R \
  --input "$INPUT_DIR/aa_within_condition_binomial.csv" \
  --outdir "$OUTPUT_DIR/aa" \
  --level aa \
  --enrichment-type "$ENRICHMENT_TYPE" \
  --format "$FORMAT" \
  --dpi "$DPI" \
  "${OPTIONAL_FLAGS[@]}")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon level ---
echo ""
echo "--- Codon Level ---"
CMD=(Rscript R_scripts/global_occupancy_within_condition.R \
  --input "$INPUT_DIR/codon_within_condition_binomial.csv" \
  --outdir "$OUTPUT_DIR/codon" \
  --level codon \
  --enrichment-type "$ENRICHMENT_TYPE" \
  --format "$FORMAT" \
  --dpi "$DPI" \
  "${OPTIONAL_FLAGS[@]}")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
