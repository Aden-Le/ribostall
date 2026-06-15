#!/bin/bash
#----------------------------------------------------
# Bash script: enrichment tests on the stall-site CSVs
# (stall_sites_non_consensus_stats.py)
#
# Ribopy-free. Consumes stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv produced by
# run_stall_sites_non_consensus.sh and writes within-condition binomial,
# between-condition Wilcoxon, between-timepoint Wilcoxon + Fisher,
# per-timepoint Fisher, and per-timepoint background-aware diff result
# CSVs into $OUT_DIR.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Directory containing stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv from run_stall_sites_non_consensus.sh
OUT_ENRICHMENT="./results/stall_sites/enrichment"
OUT_DIR="./results/stall_sites/enrichment/analysis_stats"

# Headline condition for the between-condition tests: the between-condition
# Wilcoxon (Analysis 2), the per-timepoint Fisher (Analysis 4), and the
# per-timepoint background-aware diff (Analysis 5). A positive effect size
# (log2_FC / log2 odds ratio / delta_log2_enrichment) means enriched in THIS
# condition. Must match one of the condition labels above (the part before the
# first underscore, e.g. BWM or control). Set to "BWM" so positive = BWM-
# enriched. Leave empty ("") to fall back to alphabetical ordering.
HEADLINE_CONDITION="BWM"

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=============================================="
echo "RIBOSOME STALL SITE ENRICHMENT STATS"
echo "=============================================="
echo "Groups: $EXP_GROUPS"
echo "Input: $OUT_ENRICHMENT"
echo "Output: $OUT_DIR"
echo "Headline condition: ${HEADLINE_CONDITION:-alphabetical default}"
echo "=============================================="

# Pass --headline-condition only when set, so an empty value falls back to the
# script's alphabetical default. -n "$HEADLINE_CONDITION" is true when non-empty:
# if set, the flag is added; if empty, the whole argument is omitted below.
HEADLINE_FLAG=()
[ -n "$HEADLINE_CONDITION" ] && HEADLINE_FLAG=(--headline-condition "$HEADLINE_CONDITION")

for LEVEL in aa codon; do
  python3 scripts/stall_sites_non_consensus_stats.py \
    --stall-sites "$OUT_ENRICHMENT/stall_sites_${LEVEL}.csv" \
    --background "$OUT_ENRICHMENT/per_group_background_${LEVEL}.csv" \
    --groups "$EXP_GROUPS" \
    --out-dir "$OUT_DIR" \
    "${HEADLINE_FLAG[@]}"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
