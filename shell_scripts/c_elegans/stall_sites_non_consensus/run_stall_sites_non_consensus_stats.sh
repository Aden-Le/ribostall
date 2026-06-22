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

# --- Which analyses to run -------------------------------------------------
# Each analysis defaults to true (runs). Set one to false to skip it; leaving it
# true (or unset) runs it. A skipped analysis writes no output CSV. The
# between-timepoint block (Analysis 3) is split into its two sub-tests so each
# can be toggled independently.
RUN_WITHIN_CONDITION=true               # Analysis 1: within-condition binomial
RUN_BETWEEN_CONDITION_WILCOXON=true     # Analysis 2: between-condition Wilcoxon
RUN_BETWEEN_TIMEPOINT_WILCOXON=true     # Analysis 3a: between-timepoint Wilcoxon (pooled)
RUN_BETWEEN_TIMEPOINT_FISHER=true       # Analysis 3b: between-timepoint Fisher (within condition)
RUN_PER_TIMEPOINT_FISHER=true           # Analysis 4: per-timepoint Fisher's exact
RUN_PER_TIMEPOINT_BACKGROUND_DIFF=true  # Analysis 5: per-timepoint background-aware diff

# Headline condition for the between-condition tests (Wilcoxon Analysis 2,
# per-timepoint Fisher Analysis 4, per-timepoint background-aware diff Analysis 5)
# lives in the shared _headline_config.sh, which the plot launchers also source —
# so the stats direction and the plot labels come from ONE place and cannot drift.
# A positive effect (log2_FC / log2 odds ratio / delta_log2_enrichment) means
# enriched in HEADLINE_CONDITION. Leave it empty there to fall back to alphabetical.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "RIBOSOME STALL SITE ENRICHMENT STATS"
echo "=============================================="
echo "Groups: $EXP_GROUPS"
echo "Input: $OUT_ENRICHMENT"
echo "Output: $OUT_DIR"
echo "Headline condition: ${HEADLINE_CONDITION:-alphabetical default}"
echo "Analyses: within=$RUN_WITHIN_CONDITION  bc_wilcoxon=$RUN_BETWEEN_CONDITION_WILCOXON  bt_wilcoxon=$RUN_BETWEEN_TIMEPOINT_WILCOXON  bt_fisher=$RUN_BETWEEN_TIMEPOINT_FISHER  pt_fisher=$RUN_PER_TIMEPOINT_FISHER  pt_bgdiff=$RUN_PER_TIMEPOINT_BACKGROUND_DIFF"
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
    --stall-sites "$OUT_ENRICHMENT/stall_sites_${LEVEL}.csv" \
    --background "$OUT_ENRICHMENT/per_group_background_${LEVEL}.csv" \
    --groups "$EXP_GROUPS" \
    --out-dir "$OUT_DIR" \
    "${HEADLINE_FLAG[@]}" \
    --within-condition "${RUN_WITHIN_CONDITION:-true}" \
    --between-condition-wilcoxon "${RUN_BETWEEN_CONDITION_WILCOXON:-true}" \
    --between-timepoint-wilcoxon "${RUN_BETWEEN_TIMEPOINT_WILCOXON:-true}" \
    --between-timepoint-fisher "${RUN_BETWEEN_TIMEPOINT_FISHER:-true}" \
    --per-timepoint-fisher "${RUN_PER_TIMEPOINT_FISHER:-true}" \
    --per-timepoint-background-diff "${RUN_PER_TIMEPOINT_BACKGROUND_DIFF:-true}"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
