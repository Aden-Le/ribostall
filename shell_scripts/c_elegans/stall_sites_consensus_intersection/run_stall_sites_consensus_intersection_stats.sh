#!/bin/bash
#----------------------------------------------------
# Bash script: Fisher count-collapsing enrichment tests on the
# CONSENSUS INTERSECTION stall-site CSVs (stall_sites_consensus_intersection_stats.py)
#
# Ribopy-free. Consumes stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv produced by
# run_stall_sites_consensus_intersection.sh and writes:
#   A1 — within-condition binomial enrichment
#   A3 — between-condition Fisher (flat) or per-timepoint Fisher (with --timepoints)
#   A6 — between-timepoint Fisher within each condition (--timepoints only)
#
# In the INTERSECTION design every group shares one transcript universe, so raw
# stall-site shares are apples-to-apples and Fisher's exact is the valid
# between-group comparison. (The per-group backgrounds are identical, which
# makes the background-aware diff degenerate — those tests A4/A7 live in the
# union stats runner.) The within-condition binomial A1 still uses the per-group
# background.
#
# The consensus call collapses replicates into one set per group and writes
# replicate == group, so each group's "rep" is the group name.
#
# Flat design (no timepoints): omit TIMEPOINTS or leave it empty. A3 emits one
# between-condition Fisher CSV; A6 is skipped.
# Timepoint design: set TIMEPOINTS (chronological order) and set EXP_GROUPS to
# timepoint-bearing names, e.g.
#   treatment_day_0:treatment_day_0;control_day_0:control_day_0;...
# A3 then slices per-timepoint and A6 runs across every day-pair.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Format: "group1:rep1,rep2;group2:rep1,rep2"
# Consensus sets replicate == group, so each group's "rep" is the group name.
EXP_GROUPS='control:control;treatment:treatment'

# Timepoint labels in chronological order (earliest first). Leave empty for a
# flat control-vs-treatment design (today's C. elegans consensus setup). Set to
# a comma-separated list when groups carry timepoints (e.g. treatment_day_0).
# Timepoints are NOT sorted automatically (a string sort places "day_10" before "day_5").
TIMEPOINTS=''

# Directory containing stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv from run_stall_sites_consensus_intersection.sh
RAW_DIR="./results/c_elegans/stall_sites_consensus_intersection/raw"
OUT_DIR="./results/c_elegans/stall_sites_consensus_intersection/analysis"

# --- Which analyses to run -------------------------------------------------
# Each analysis defaults to true (runs). Set one to false to skip it; leaving it
# true (or unset) runs it. A skipped analysis writes no output CSV.
# A6 is automatically skipped when TIMEPOINTS is empty.
RUN_WITHIN_CONDITION=true                       # A1: within-condition binomial
RUN_BETWEEN_CONDITION_FISHER=true               # A3: between-condition Fisher (or per-tp with --timepoints)
RUN_BETWEEN_TIMEPOINT_FISHER=true               # A6: between-timepoint Fisher within condition (tp only)

# Headline condition for the between-condition Fisher test lives in the shared
# _headline_config.sh, which the plot launchers also source — so the stats
# direction and the plot labels come from ONE place and cannot drift. A positive
# log2(odds ratio) means enriched in HEADLINE_CONDITION. Leave it empty there to
# fall back to alphabetical.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "CONSENSUS INTERSECTION STALL SITE ENRICHMENT STATS"
echo "=============================================="
echo "Groups: $EXP_GROUPS"
echo "Timepoints: ${TIMEPOINTS:-none (flat design)}"
echo "Input: $RAW_DIR"
echo "Output: $OUT_DIR"
echo "Headline condition: ${HEADLINE_CONDITION:-alphabetical default}"
echo "Analyses: within=$RUN_WITHIN_CONDITION  fisher=$RUN_BETWEEN_CONDITION_FISHER  bt_fisher=$RUN_BETWEEN_TIMEPOINT_FISHER"
echo "=============================================="

# Pass --headline-condition only when set, so an empty value falls back to the
# script's alphabetical default.
HEADLINE_FLAG=()
[ -n "$HEADLINE_CONDITION" ] && HEADLINE_FLAG=(--headline-condition "$HEADLINE_CONDITION")

# Pass --timepoints only when TIMEPOINTS is non-empty (flat designs omit it).
TIMEPOINTS_FLAG=()
[ -n "$TIMEPOINTS" ] && TIMEPOINTS_FLAG=(--timepoints "$TIMEPOINTS")

# Each RUN_* config var is passed straight through as a true/false value; the
# Python script skips any analysis given false. ${VAR:-true} keeps an unset var
# running, matching the CONFIG defaults above.
for LEVEL in aa codon; do
  python3 scripts/stall_sites_consensus_intersection_stats.py \
    --stall-sites "$RAW_DIR/stall_sites_${LEVEL}.csv" \
    --background "$RAW_DIR/per_group_background_${LEVEL}.csv" \
    --groups "$EXP_GROUPS" \
    --out-dir "$OUT_DIR" \
    "${HEADLINE_FLAG[@]}" \
    "${TIMEPOINTS_FLAG[@]}" \
    --within-condition "${RUN_WITHIN_CONDITION:-true}" \
    --between-condition-fisher "${RUN_BETWEEN_CONDITION_FISHER:-true}" \
    --between-timepoint-fisher "${RUN_BETWEEN_TIMEPOINT_FISHER:-true}"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
