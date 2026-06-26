#!/bin/bash
#----------------------------------------------------
# Bash script: background-aware count-collapsing enrichment tests on the
# CONSENSUS UNION stall-site CSVs (stall_sites_consensus_union_stats.py)
#
# Ribopy-free. Consumes stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv produced by
# run_stall_sites_consensus_union.sh and writes:
#   A1 — within-condition binomial enrichment
#   A4 — between-condition background-aware diff (flat) or per-timepoint (with --timepoints)
#   A7 — between-timepoint background-aware diff, pooled (--timepoints only)
#
# In the UNION design each group keeps its own filtered transcript set, so the
# per-group backgrounds differ between conditions; the background-aware diff
# (each condition normalized to its OWN background) is the valid between-group
# comparison. The Fisher tests (A3/A6) live in the intersection stats runner.
#
# The consensus call collapses replicates into one set per group and writes
# replicate == group, so each group's "rep" is the group name.
#
# Flat design (no timepoints): omit TIMEPOINTS or leave it empty. A4 emits one
# between-condition CSV; A7 is skipped.
# Timepoint design: set TIMEPOINTS (chronological order) and set EXP_GROUPS to
# timepoint-bearing names, e.g.
#   treatment_day_0:treatment_day_0;control_day_0:control_day_0;...
# A4 then slices per-timepoint and A7 runs across every day-pair.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Format: "group1:rep1,rep2;group2:rep1,rep2"
# Consensus sets replicate == group, so each group's "rep" is the group name.
# Timepoint design: one group per (condition, timepoint) cell, replicate == group.
EXP_GROUPS='control_day_0:control_day_0;control_day_5:control_day_5;control_day_10:control_day_10;BWM_day_0:BWM_day_0;BWM_day_5:BWM_day_5;BWM_day_10:BWM_day_10'

# Timepoint labels in chronological order (earliest first). Set when the groups
# carry timepoints (e.g. BWM_day_0); leave empty for a flat control-vs-treatment
# design. Timepoints are NOT sorted automatically (a string sort places "day_10"
# before "day_5").
TIMEPOINTS='day_0,day_5,day_10'

# Directory containing stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv from run_stall_sites_consensus_union.sh
RAW_DIR="./results/c_elegans/stall_sites_consensus_union/raw"
OUT_DIR="./results/c_elegans/stall_sites_consensus_union/analysis"

# --- Which analyses to run -------------------------------------------------
# Each analysis defaults to true (runs). Set one to false to skip it; leaving it
# true (or unset) runs it. A skipped analysis writes no output CSV.
# A7 is automatically skipped when TIMEPOINTS is empty.
RUN_WITHIN_CONDITION=true                       # A1: within-condition binomial
RUN_BETWEEN_CONDITION_BACKGROUND_DIFF=true      # A4: between-condition background-aware diff (or per-tp)
RUN_BETWEEN_TIMEPOINT_BACKGROUND_DIFF=true      # A7: between-timepoint background-aware diff (tp only)

# Headline condition for the between-condition background-aware diff lives in the
# shared _headline_config.sh, which the plot launchers also source — so the stats
# direction and the plot labels come from ONE place and cannot drift. A positive
# delta_log2_enrichment means enriched vs background in HEADLINE_CONDITION. Leave
# it empty there to fall back to alphabetical.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "CONSENSUS UNION STALL SITE ENRICHMENT STATS"
echo "=============================================="
echo "Groups: $EXP_GROUPS"
echo "Timepoints: ${TIMEPOINTS:-none (flat design)}"
echo "Input: $RAW_DIR"
echo "Output: $OUT_DIR"
echo "Headline condition: ${HEADLINE_CONDITION:-alphabetical default}"
echo "Analyses: within=$RUN_WITHIN_CONDITION  bgdiff=$RUN_BETWEEN_CONDITION_BACKGROUND_DIFF  bt_bgdiff=$RUN_BETWEEN_TIMEPOINT_BACKGROUND_DIFF"
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
  python3 scripts/stall_sites_consensus_union_stats.py \
    --stall-sites "$RAW_DIR/stall_sites_${LEVEL}.csv" \
    --background "$RAW_DIR/per_group_background_${LEVEL}.csv" \
    --groups "$EXP_GROUPS" \
    --out-dir "$OUT_DIR" \
    "${HEADLINE_FLAG[@]}" \
    "${TIMEPOINTS_FLAG[@]}" \
    --within-condition "${RUN_WITHIN_CONDITION:-true}" \
    --between-condition-background-diff "${RUN_BETWEEN_CONDITION_BACKGROUND_DIFF:-true}" \
    --between-timepoint-background-diff "${RUN_BETWEEN_TIMEPOINT_BACKGROUND_DIFF:-true}"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
