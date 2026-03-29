#!/bin/bash
#----------------------------------------------------
# Bash script: run global_codon_occ.py with
# amino acid aggregation and statistical tests
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing _coverage.pkl.gz files
RIBO_DIR="./all_ribo_file"
RIBO_FILE="$RIBO_DIR/C_elegan_all_02_04_2026.ribo"

# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Trimming parameters
TRIM_START=20
TRIM_STOP=10

# Offset type
OFFSET="P"

# Reference file
REFERENCE_FILE="./C_elegan_reference/appris_celegans_v1_selected_new.fa"

# Output directory
OUT_DIR="./global_occupancy_results"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Find coverage pickle
PICKLE=$(ls "$RIBO_DIR"/*_coverage.pkl.gz 2>/dev/null | head -1)

if [ -z "$PICKLE" ]; then
  echo "Error: No coverage pickle files found in $RIBO_DIR"
  exit 1
fi

echo "=============================================="
echo "GLOBAL CODON & AMINO ACID OCCUPANCY ANALYSIS"
echo "=============================================="
echo "Coverage pickle: $PICKLE"
echo "Ribo file: $RIBO_FILE"
echo "Reference: $REFERENCE_FILE"
echo "Groups: $EXP_GROUPS"
echo "Parameters: trim_start=$TRIM_START, trim_stop=$TRIM_STOP, offset=$OFFSET"
echo "Output directory: $OUT_DIR"
echo "=============================================="

CMD=(python3 global_codon_occ.py \
  --pickle "$PICKLE" \
  --ribo "$RIBO_FILE" \
  --reference "$REFERENCE_FILE" \
  --ofset "$OFFSET" \
  --groups "$EXP_GROUPS" \
  --trim-start "$TRIM_START" \
  --trim-stop "$TRIM_STOP" \
  --out-dir "$OUT_DIR" \
  --run-stats)

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
