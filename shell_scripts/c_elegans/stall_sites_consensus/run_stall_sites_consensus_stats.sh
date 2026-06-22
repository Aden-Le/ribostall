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

# --- Which analyses to run -------------------------------------------------
# Each analysis defaults to true (runs). Set one to false to skip it; leaving it
# true (or unset) runs it. A skipped analysis writes no output CSV.
RUN_WITHIN_CONDITION=true                     # Analysis 1: within-condition binomial
RUN_BETWEEN_CONDITION_FISHER=true             # Analysis 2: between-condition Fisher's exact
RUN_BETWEEN_CONDITION_BACKGROUND_DIFF=true    # Analysis 3: between-condition background-aware diff

# Headline condition for the between-condition tests (Fisher + background-aware
# diff) lives in the shared _headline_config.sh, which the plot launchers also
# source — so the stats direction and the plot labels come from ONE place and
# cannot drift. A positive log2(odds ratio) / delta_log2_enrichment means enriched
# in HEADLINE_CONDITION. Leave it empty there to fall back to alphabetical.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "CONSENSUS STALL SITE ENRICHMENT STATS"
echo "=============================================="
echo "Groups: $EXP_GROUPS"
echo "Input: $OUT_ENRICHMENT"
echo "Output: $OUT_DIR"
echo "Headline condition: ${HEADLINE_CONDITION:-alphabetical default}"
echo "Analyses: within=$RUN_WITHIN_CONDITION  fisher=$RUN_BETWEEN_CONDITION_FISHER  bgdiff=$RUN_BETWEEN_CONDITION_BACKGROUND_DIFF"
echo "=============================================="

# Pass --headline-condition only when set, so an empty value falls back to the
# script's alphabetical default.
# -n $HEADLINE_CONDITION checks if the variable is non-empty, if it is, then the flag is added.
# Its like an if else statement, if HEADLINE_CONDITION is not empty, then add the flag.
# If not then the whole argument is omitted from the python script below
HEADLINE_FLAG=()
[ -n "$HEADLINE_CONDITION" ] && HEADLINE_FLAG=(--headline-condition "$HEADLINE_CONDITION")

# Each RUN_* config var is passed straight through as a true/false value; the
# Python script skips any analysis given false. ${VAR:-true} keeps an unset var
# running, matching the CONFIG defaults above.
for LEVEL in aa codon; do
  python3 scripts/stall_sites_consensus_stats.py \
    --stall-sites "$OUT_ENRICHMENT/stall_sites_${LEVEL}.csv" \
    --background "$OUT_ENRICHMENT/per_group_background_${LEVEL}.csv" \
    --groups "$EXP_GROUPS" \
    --out-dir "$OUT_DIR" \
    "${HEADLINE_FLAG[@]}" \
    --within-condition "${RUN_WITHIN_CONDITION:-true}" \
    --between-condition-fisher "${RUN_BETWEEN_CONDITION_FISHER:-true}" \
    --between-condition-background-diff "${RUN_BETWEEN_CONDITION_BACKGROUND_DIFF:-true}"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
