#!/bin/bash
#----------------------------------------------------
# Bash script: run global_codon_occ.py
# Computes per-experiment codon and amino acid occupancy
# and saves base CSVs to out_dir/raw/
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing .ribo file and _coverage.pkl.gz
RIBO_DIR="./all_ribo_file"
RIBO_FILE="$RIBO_DIR/C_elegan_all_02_04_2026.ribo"

# Format: "group1:rep1,rep2;group2:rep1,rep2"
# Used to filter the coverage dict to declared replicates only
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Trimming parameters
TRIM_START=20
TRIM_STOP=10

# Reference file
REFERENCE_FILE="./reference/appris_celegans_v1_selected_new.fa"

# Output directory (must match --out-dir used in run_global_codon_occ_stats.sh)
OUT_DIR="./results/global_occupancy"

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
echo "GLOBAL CODON & AMINO ACID OCCUPANCY — STEP 1"
echo "Compute occupancy tables and save base CSVs"
echo "=============================================="
echo "Coverage pickle: $PICKLE"
echo "Ribo file:       $RIBO_FILE"
echo "Reference:       $REFERENCE_FILE"
echo "Groups:          $EXP_GROUPS"
echo "Parameters:      trim_start=$TRIM_START, trim_stop=$TRIM_STOP"
echo "Output directory: $OUT_DIR"
echo "=============================================="

CMD=(python3 scripts/global_codon_occ.py \
  --pickle "$PICKLE" \
  --ribo "$RIBO_FILE" \
  --reference "$REFERENCE_FILE" \
  --groups "$EXP_GROUPS" \
  --trim-start "$TRIM_START" \
  --trim-stop "$TRIM_STOP" \
  --out-dir "$OUT_DIR")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done. Base CSVs written to $OUT_DIR/raw/"
echo "Run run_global_codon_occ_stats.sh to run statistical tests."
date
echo "=============================================="
