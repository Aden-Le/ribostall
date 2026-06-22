#!/bin/bash
#----------------------------------------------------
# Bash script: generate within-condition bar plots
# for global codon and amino acid occupancy enrichment
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============

# Input directory containing the binomial CSVs
INPUT_DIR="./results/c_elegans/global_occupancy/analysis_corrected"

# Output directory for plots
OUTPUT_DIR="./results/c_elegans/global_occupancy/plots/within_condition"

# Enrichment type: "unweighted", "weighted", or "both"
ENRICHMENT_TYPE="both"

# Output format: "pdf", "png", or "both"
FORMAT="both"

# DPI for PNG output
DPI=300

# Y-axis cap: clamp y-axis to ±this value
# Leave empty ("") to disable
Y_CAP=""

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

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

# Build optional flags. Mega-composite preserves global's prior behaviour
# (the old global bar plot rendered an all-groups grid; the unified volcano
# script gates this behind a flag so stall_sites runs do not produce it).
OPTIONAL_FLAGS=(--mega-composite)
if [ -n "$Y_CAP" ]; then
  OPTIONAL_FLAGS+=(--y-cap "$Y_CAP")
fi

# --- Amino acid level (AA plots at the within_condition/ root) ---
echo ""
echo "--- Amino Acid Level ---"
CMD=(Rscript R_scripts/within_condition_volcano.R \
  --input "$INPUT_DIR/aa_within_condition_binomial.csv" \
  --outdir "$OUTPUT_DIR" \
  --level aa \
  --enrichment-type "$ENRICHMENT_TYPE" \
  --format "$FORMAT" \
  --dpi "$DPI" \
  "${OPTIONAL_FLAGS[@]}")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon level (codon/ subdir) ---
echo ""
echo "--- Codon Level ---"
CMD=(Rscript R_scripts/within_condition_volcano.R \
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
