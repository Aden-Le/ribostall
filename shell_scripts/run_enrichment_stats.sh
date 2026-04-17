#!/bin/bash
#----------------------------------------------------
# Bash script: enrichment tests on the stall-site CSVs
# (stall_sites_non_consensus_stats.py)
#
# Consumes stall_sites_{codon,aa}.csv produced by
# run_enrichment.sh and writes within-condition,
# between-condition Wilcoxon, and per-timepoint
# Fisher result CSVs into $OUT_ENRICHMENT.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing the .ribo file
RIBO_DIR="./all_ribo_file"
RIBO_FILE="$RIBO_DIR/C_elegan_all_02_04_2026.ribo"

# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Reference file
REFERENCE_FILE="./reference/appris_celegans_v1_selected_new.fa"

# Directory containing stall_sites_{codon,aa}.csv from run_enrichment.sh
OUT_ENRICHMENT="./results/stall_sites/enrichment"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "RIBOSOME STALL SITE ENRICHMENT STATS"
echo "=============================================="
echo "Ribo file: $RIBO_FILE"
echo "Reference: $REFERENCE_FILE"
echo "Groups: $EXP_GROUPS"
echo "Input/Output: $OUT_ENRICHMENT"
echo "=============================================="

for LEVEL in aa codon; do
  python3 scripts/stall_sites_non_consensus_stats.py \
    --stall-sites "$OUT_ENRICHMENT/stall_sites_${LEVEL}.csv" \
    --ribo "$RIBO_FILE" \
    --reference "$REFERENCE_FILE" \
    --groups "$EXP_GROUPS" \
    --out-dir "$OUT_ENRICHMENT"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
