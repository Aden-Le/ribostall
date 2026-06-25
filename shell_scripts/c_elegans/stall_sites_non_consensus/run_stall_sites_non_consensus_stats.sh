#!/bin/bash
#----------------------------------------------------
# Bash script: per-replicate Wilcoxon tests on the stall-site CSVs
# (stall_sites_non_consensus_stats.py)
#
# Ribopy-free. Consumes stall_sites_{codon,aa}.csv produced by
# run_stall_sites_non_consensus.sh and writes:
#   A2 — between-condition Wilcoxon (per-replicate frequencies)
#   A5 — between-timepoint Wilcoxon (per-replicate, one CSV per day-pair;
#         only emitted when --timepoints is given with >=2 timepoints)
#
# The count-collapsing tests (A1 within-condition binomial, A3/A4
# between-condition Fisher + background-diff, A6/A7 between-timepoint
# Fisher + background-diff) pool biological replicates, which is
# pseudoreplication on per-replicate data. They live exclusively in the
# consensus stats runners (run_stall_sites_consensus_union_stats.sh for
# A1/A4/A7, run_stall_sites_consensus_intersection_stats.sh for A1/A3/A6),
# which run them at n=1 per cell on reproducibility-filtered sets. The split is
# structural — this script contains no count-collapsing code — so
# pseudoreplication is impossible by construction, not merely disabled.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Timepoint labels in chronological order (earliest first). Drives the
# later-vs-earlier day-pairs for A5; timepoints are NOT sorted automatically
# (a string sort places "day_10" before "day_5"). Leave empty to skip A5.
TIMEPOINTS='day_0,day_5,day_10'

# Directory containing stall_sites_{codon,aa}.csv from run_stall_sites_non_consensus.sh
RAW_DIR="./results/c_elegans/stall_sites_non_consensus/raw"
OUT_DIR="./results/c_elegans/stall_sites_non_consensus/analysis"

# --- Which analyses to run -------------------------------------------------
# Each analysis defaults to true (runs). Set one to false to skip it; leaving it
# true (or unset) runs it. A skipped analysis writes no output CSV.
RUN_BETWEEN_CONDITION_WILCOXON=true     # A2: between-condition Wilcoxon
RUN_BETWEEN_TIMEPOINT_WILCOXON=true     # A5: between-timepoint Wilcoxon (pooled across conditions)

# Headline condition for the between-condition Wilcoxon (A2) lives in the shared
# _headline_config.sh, which the plot launchers also source — so the stats
# direction and the plot labels come from ONE place and cannot drift.
# A positive log2_FC means enriched in HEADLINE_CONDITION.
# Leave it empty there to fall back to alphabetical.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "RIBOSOME STALL SITE ENRICHMENT STATS"
echo "=============================================="
echo "Groups: $EXP_GROUPS"
echo "Timepoints: $TIMEPOINTS"
echo "Input: $RAW_DIR"
echo "Output: $OUT_DIR"
echo "Headline condition: ${HEADLINE_CONDITION:-alphabetical default}"
echo "Analyses: bc_wilcoxon=$RUN_BETWEEN_CONDITION_WILCOXON  bt_wilcoxon=$RUN_BETWEEN_TIMEPOINT_WILCOXON"
echo "=============================================="

# Pass --headline-condition only when set, so an empty value falls back to the
# script's alphabetical default. -n "$HEADLINE_CONDITION" is true when non-empty:
# if set, the flag is added; if empty, the whole argument is omitted below.
HEADLINE_FLAG=()
[ -n "$HEADLINE_CONDITION" ] && HEADLINE_FLAG=(--headline-condition "$HEADLINE_CONDITION")

# Each RUN_* config var is passed straight through as a true/false value; the
# Python script skips any analysis given false. ${VAR:-true} keeps an unset var
# running, matching the CONFIG defaults above.
for LEVEL in aa codon; do
  python3 scripts/stall_sites_non_consensus_stats.py \
    --stall-sites "$RAW_DIR/stall_sites_${LEVEL}.csv" \
    --groups "$EXP_GROUPS" \
    --timepoints "$TIMEPOINTS" \
    --out-dir "$OUT_DIR" \
    "${HEADLINE_FLAG[@]}" \
    --between-condition-wilcoxon "${RUN_BETWEEN_CONDITION_WILCOXON:-true}" \
    --between-timepoint-wilcoxon "${RUN_BETWEEN_TIMEPOINT_WILCOXON:-true}"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
