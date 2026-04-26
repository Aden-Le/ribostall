#!/bin/bash
#----------------------------------------------------
# Bash script: run global_codon_occ_stats.py
# Runs statistical tests on the base CSVs produced
# by run_global_codon_occ.sh (reads from out_dir/raw/,
# writes results to out_dir/analysis/{E,P,A}/).
#
# The Python script is general-purpose: it processes one
# CSV (one site, one level) per invocation. This shell
# script loops over all 6 combinations (3 sites x 2 levels).
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Must match --out-dir used in run_global_codon_occ.sh
OUT_DIR="./results/global_occupancy"

# Ribosome sites and occupancy levels to process
SITES=(E P A)
LEVELS=(codon aa)

# ===============================================

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "GLOBAL CODON & AMINO ACID OCCUPANCY — STEP 2"
echo "Statistical tests on base CSVs"
echo "=============================================="
echo "Input directory:  $OUT_DIR/raw/"
echo "Output directory: $OUT_DIR/analysis/{${SITES[*]// /,}}/"
echo "Sites:            ${SITES[*]}"
echo "Levels:           ${LEVELS[*]}"
echo "Groups:           $EXP_GROUPS"
echo "=============================================="

for site in "${SITES[@]}"; do
  for level in "${LEVELS[@]}"; do
    INPUT_CSV="$OUT_DIR/raw/${level}_occupancy_${site}.csv"
    SITE_OUT_DIR="$OUT_DIR/analysis/$site"

    echo ""
    echo "----------------------------------------------"
    echo "Site: $site  |  Level: $level"
    echo "Input:  $INPUT_CSV"
    echo "Output: $SITE_OUT_DIR/"
    echo "----------------------------------------------"

    CMD=(python3 scripts/global_codon_occ_stats.py \
      --input-csv "$INPUT_CSV" \
      --out-dir "$SITE_OUT_DIR" \
      --groups "$EXP_GROUPS" \
      --prefix "$level")

    echo "Running: ${CMD[@]}"
    "${CMD[@]}"
  done
done

echo ""
echo "=============================================="
echo "Done. Results written to $OUT_DIR/analysis/{${SITES[*]// /,}}/"
date
echo "=============================================="
