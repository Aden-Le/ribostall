#!/bin/bash
#----------------------------------------------------
# Bash script: enrichment tests on the stall-site CSVs
# (stall_sites_non_consensus_stats.py)
#
# Ribopy-free. Consumes stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv produced by
# run_enrichment.sh and writes within-condition,
# between-condition Wilcoxon, and per-timepoint Fisher
# result CSVs into $OUT_ENRICHMENT.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Directory containing stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv from run_enrichment.sh
OUT_ENRICHMENT="./results/stall_sites/enrichment"
OUT_DIR="./results/stall_sites/enrichment/analysis_stats"

# ===============================================

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "RIBOSOME STALL SITE ENRICHMENT STATS"
echo "=============================================="
echo "Groups: $EXP_GROUPS"
echo "Input/Output: $OUT_ENRICHMENT"
echo "=============================================="

for LEVEL in aa codon; do
  python3 scripts/stall_sites_non_consensus_stats.py \
    --stall-sites "$OUT_ENRICHMENT/stall_sites_${LEVEL}.csv" \
    --background "$OUT_ENRICHMENT/per_group_background_${LEVEL}.csv" \
    --groups "$EXP_GROUPS" \
    --out-dir "$OUT_DIR"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
