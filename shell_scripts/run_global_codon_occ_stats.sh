#!/bin/bash
#----------------------------------------------------
# Bash script: run global_codon_occ_stats.py
# Runs statistical tests on the base CSVs produced
# by run_global_codon_occ.sh (reads from out_dir/raw/,
# writes results to out_dir/analysis/)
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Must match --out-dir used in run_global_codon_occ.sh
OUT_DIR="./results/global_occupancy"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "GLOBAL CODON & AMINO ACID OCCUPANCY — STEP 2"
echo "Statistical tests on base CSVs"
echo "=============================================="
echo "Input directory:  $OUT_DIR/raw/"
echo "Output directory: $OUT_DIR/analysis/"
echo "Groups:           $EXP_GROUPS"
echo "=============================================="

CMD=(python3 scripts/global_codon_occ_stats.py \
  --out-dir "$OUT_DIR" \
  --groups "$EXP_GROUPS")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done. Results written to $OUT_DIR/analysis/"
date
echo "=============================================="
