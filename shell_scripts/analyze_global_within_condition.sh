#!/bin/bash
#----------------------------------------------------
# Bash script: generate within-condition volcano plots
# for global codon and amino acid occupancy enrichment
#----------------------------------------------------

# ============== CONFIG: edit these ==============
INPUT_DIR="./global_occupancy_results"
OUTPUT_DIR="./global_occupancy_results/within_condition_output"
ENRICHMENT_TYPE="both"   # unweighted, weighted, or both
FORMAT="png"             # pdf, png, or both
DPI=300
Y_CAP=""                 # set to a number (e.g. 25) to cap y-axis, or leave empty
# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "GLOBAL OCCUPANCY WITHIN-CONDITION VOLCANO PLOTS"
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
