#!/bin/bash
#----------------------------------------------------
# Bash script: enrichment tests on the CONSENSUS stall-site CSVs
# (stall_sites_consensus_stats.py)
#
# Ribopy-free. Consumes stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv produced by
# run_stall_sites_consensus.sh and writes within-condition
# binomial + between-condition Fisher result CSVs into $OUT_DIR.
#
# Flat control-vs-treatment design (no timepoints): the consensus
# call collapses replicates into one set per group and writes
# replicate == group, so the reps below are the group names.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Format: "group1:rep1,rep2;group2:rep1,rep2"
# Consensus sets replicate == group, so each group's "rep" is the group name.
EXP_GROUPS='control:control;treatment:treatment'

# Directory containing stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv from run_stall_sites_consensus.sh
OUT_ENRICHMENT="./results/stall_sites/enrichment"
OUT_DIR="./results/stall_sites/enrichment/analysis_stats"

# Headline condition for the between-condition Fisher test: a positive
# log2(odds ratio) means the codon/AA is enriched in THIS condition. Must match
# one of the group labels above. Set to "treatment" so positive = treatment-
# enriched. Leave empty ("") to fall back to alphabetical ordering.
HEADLINE_CONDITION="treatment"

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=============================================="
echo "CONSENSUS STALL SITE ENRICHMENT STATS"
echo "=============================================="
echo "Groups: $EXP_GROUPS"
echo "Input: $OUT_ENRICHMENT"
echo "Output: $OUT_DIR"
echo "Headline condition: ${HEADLINE_CONDITION:-alphabetical default}"
echo "=============================================="

# Pass --headline-condition only when set, so an empty value falls back to the
# script's alphabetical default.
# -n $HEADLINE_CONDITION checks if the variable is non-empty, if it is, then the flag is added.
# Its like an if else statement, if HEADLINE_CONDITION is not empty, then add the flag.
# If not then the whole argument is omitted from the python script below
HEADLINE_FLAG=()
[ -n "$HEADLINE_CONDITION" ] && HEADLINE_FLAG=(--headline-condition "$HEADLINE_CONDITION")

for LEVEL in aa codon; do
  python3 scripts/stall_sites_consensus_stats.py \
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
