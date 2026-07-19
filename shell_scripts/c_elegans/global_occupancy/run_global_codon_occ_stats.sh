#!/bin/bash
#----------------------------------------------------
# Bash script: run global_codon_occ_stats.py
# Runs statistical tests on the base CSVs produced
# by run_global_codon_occ.sh (reads from out_dir/raw/).
#
# The Python script processes all 3 sites (E/P/A) for one
# level per invocation and writes the merged CSVs to
# out_dir/analysis/ (the E/P/A per-site frames
# concatenated with a 'site' column — the old
# merge_global_occupancy_analysis.py step is folded in). The
# per-site frames are an internal intermediate and are NOT
# exported. This shell script loops the 2 levels.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Timepoint labels in chronological order (earliest first). Drives the order of
# the per-timepoint Fisher (A3) and the later-vs-earlier between-timepoint pairs
# (A5/A6); timepoints are NOT sorted automatically (a string sort places
# "day_10" before "day_5").
TIMEPOINTS='day_0,day_5,day_10'

# Must match --out-dir used in run_global_codon_occ.sh
OUT_DIR="./results/c_elegans/global_occupancy"

# Ribosome sites and occupancy levels to process
SITES=(E P A)
LEVELS=(codon aa)

# --- Which analyses to run -------------------------------------------------
# Each analysis defaults to true (runs). Set one to false to skip it; leaving it
# true (or unset) runs it. A skipped analysis writes no per-site CSV and is
# therefore absent from the merged analysis/ tree.
RUN_WITHIN_CONDITION=true                   # A1: within-condition binomial
RUN_BETWEEN_CONDITION_WILCOXON=false         # A2: between-condition Wilcoxon
RUN_BETWEEN_CONDITION_FISHER=true           # A3: between-condition Fisher (flat) or per-tp (with --timepoints)
RUN_BETWEEN_TIMEPOINT_WILCOXON=false         # A5: between-timepoint Wilcoxon (pooled, tp only)
RUN_BETWEEN_TIMEPOINT_FISHER=true           # A6: between-timepoint Fisher within condition (tp only)

# Headline condition for the between-condition tests (Wilcoxon A2,
# per-timepoint Fisher A3) lives in the shared _headline_config.sh, which
# the plot launchers also source — so the stats direction and the plot labels come
# from ONE place and cannot drift. A positive effect (log2_FC / log2 odds ratio)
# means enriched in HEADLINE_CONDITION. Leave it empty there for alphabetical.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# Pass --headline-condition only when set, so an empty value falls back to the
# script's alphabetical default.
HEADLINE_FLAG=()
[ -n "$HEADLINE_CONDITION" ] && HEADLINE_FLAG=(--headline-condition "$HEADLINE_CONDITION")

# Each RUN_* config var is passed straight through as a true/false value (built
# into the CMD array below); the Python script skips any analysis given false.
# ${VAR:-true} keeps an unset var running, matching the CONFIG defaults above.

echo "=============================================="
echo "GLOBAL CODON & AMINO ACID OCCUPANCY — STEP 2"
echo "Statistical tests on base CSVs"
echo "=============================================="
echo "Raw directory:        $OUT_DIR/raw/"
echo "Merged output:        $OUT_DIR/analysis/"
echo "Sites:                ${SITES[*]}"
echo "Levels:               ${LEVELS[*]}"
echo "Groups:               $EXP_GROUPS"
echo "Timepoints:           ${TIMEPOINTS:-none (flat design)}"
echo "Analyses:             within=$RUN_WITHIN_CONDITION  bc_wilcoxon=$RUN_BETWEEN_CONDITION_WILCOXON  bc_fisher=$RUN_BETWEEN_CONDITION_FISHER  bt_wilcoxon=$RUN_BETWEEN_TIMEPOINT_WILCOXON  bt_fisher=$RUN_BETWEEN_TIMEPOINT_FISHER"
echo "=============================================="

# Pass --timepoints only when TIMEPOINTS is non-empty.
TIMEPOINTS_FLAG=()
[ -n "$TIMEPOINTS" ] && TIMEPOINTS_FLAG=(--timepoints "$TIMEPOINTS")

for level in "${LEVELS[@]}"; do
  echo ""
  echo "----------------------------------------------"
  echo "Level: $level  |  Sites: ${SITES[*]}"
  echo "----------------------------------------------"

  CMD=(python3 scripts/global_codon_occ_stats.py \
    --raw-dir "$OUT_DIR/raw" \
    --analysis-dir "$OUT_DIR/analysis" \
    --level "$level" \
    --sites "${SITES[@]}" \
    --groups "$EXP_GROUPS" \
    "${TIMEPOINTS_FLAG[@]}" \
    "${HEADLINE_FLAG[@]}" \
    --within-condition "${RUN_WITHIN_CONDITION:-true}" \
    --between-condition-wilcoxon "${RUN_BETWEEN_CONDITION_WILCOXON:-true}" \
    --between-condition-fisher "${RUN_BETWEEN_CONDITION_FISHER:-true}" \
    --between-timepoint-wilcoxon "${RUN_BETWEEN_TIMEPOINT_WILCOXON:-true}" \
    --between-timepoint-fisher "${RUN_BETWEEN_TIMEPOINT_FISHER:-true}")

  echo "Running: ${CMD[@]}"
  "${CMD[@]}"
done

echo ""
echo "=============================================="
echo "Done. Merged CSVs in $OUT_DIR/analysis/"
date
echo "=============================================="
