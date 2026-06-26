#!/bin/bash
#----------------------------------------------------
# Fisher Volcano Plot Generator (CONSENSUS INTERSECTION stall_sites)
# Drives R_scripts/between_group_volcano.R on the Fisher CSVs emitted by
# stall_sites_consensus_intersection_stats.py.
#
# In the intersection design every group shares one transcript universe, so the
# raw stall-site shares Fisher compares are apples-to-apples.
#
# This launcher targets the TIMEPOINT stats output (the stats runner sets
# TIMEPOINTS='day_0,day_5,day_10'), which writes:
#   A3  per_timepoint_fisher_{aa,codon}.csv
#         BWM-vs-control Fisher at each timepoint   → faceted by `timepoint`
#   A6  timepoint_fisher_within_condition_{cmp}_{aa,codon}.csv
#         later-vs-earlier day Fisher within each condition → faceted by `condition`
# Both CSVs already carry the grouping column (timepoint / condition) plus
# `site`, `odds_ratio`, `p_adj`, and the feature column, so between_group_volcano.R
# consumes them directly — no derived/preprocessed CSV is needed.
#
# FLAT design note: if the stats were instead run WITHOUT --timepoints, A3 emits
# a single between_condition_fisher_{aa,codon}.csv (no per-timepoint axis) and A6
# is skipped. Those files are not produced by the current timepoint run, so the
# per-timepoint / within-condition-timepoint blocks below will simply skip them
# (existence-guarded). To plot a flat run, point between_group_volcano.R at
# between_condition_fisher_*.csv with a constant --group-col instead.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
INPUT_DIR="./results/c_elegans/stall_sites_consensus_intersection/analysis"
PLOTS_DIR="./results/c_elegans/stall_sites_consensus_intersection/plots"
# Shared headline/direction config (same file the stats runner sources) so the
# per-timepoint BWM-vs-control label matches the stats numerator and cannot
# drift. A positive log2(odds ratio) = enriched in the headline condition.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"
FORMAT="both"       # pdf, png, or both
DPI=300
# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "CONSENSUS INTERSECTION STALL SITES FISHER VOLCANO PLOTS"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $PLOTS_DIR"
echo "Comparison:       $COMPARISON_LABEL"
echo "Format:           $FORMAT  | DPI: $DPI"
echo "=============================================="

# Run between_group_volcano.R on one CSV, skipping (with a note) if the input is
# absent — e.g. when the stats were run flat (no --timepoints), in which case the
# per-timepoint / within-condition-timepoint CSVs this launcher targets do not
# exist. See the FLAT design note in the header.
run_volcano() {
  local input="$1" outdir="$2" level="$3" group_col="$4" label="$5"
  if [ ! -f "$input" ]; then
    echo "  (skip) input not found: $input"
    return
  fi
  echo "Running: between_group_volcano.R --input $input --group-col $group_col"
  Rscript R_scripts/between_group_volcano.R \
    --input "$input" \
    --outdir "$outdir" \
    --level "$level" \
    --group-col "$group_col" \
    --comparison-label "$label" \
    --composite-tag "fisher" \
    --format "$FORMAT" --dpi "$DPI"
}

# =============================================
# A3 — Per-Timepoint Fisher (BWM vs Control at each day)
# Output: plots/per_timepoint_fisher/{,codon/}
# =============================================
PT_OUT="$PLOTS_DIR/per_timepoint_fisher"

echo ""
echo "--- AA: Per-Timepoint Fisher ($COMPARISON_LABEL) ---"
run_volcano "$INPUT_DIR/per_timepoint_fisher_aa.csv" \
  "$PT_OUT" aa "timepoint" "$COMPARISON_LABEL"

echo ""
echo "--- Codon: Per-Timepoint Fisher ($COMPARISON_LABEL) ---"
run_volcano "$INPUT_DIR/per_timepoint_fisher_codon.csv" \
  "$PT_OUT/codon" codon "timepoint" "$COMPARISON_LABEL"

# =============================================
# A6 — Within-Condition Timepoint Fisher (later vs earlier day, within condition)
# Output: plots/within_condition_timepoint_fisher/{comparison}/{,codon/}
# =============================================
WCT_OUT="$PLOTS_DIR/within_condition_timepoint_fisher"

for comparison in d10_vs_d0 d10_vs_d5 d5_vs_d0; do
  pretty=$(echo "$comparison" | sed 's/d\([0-9]\+\)/Day \1/g; s/_vs_/ vs /')

  echo ""
  echo "--- AA: Within-Condition Timepoint Fisher ($pretty) ---"
  run_volcano "$INPUT_DIR/timepoint_fisher_within_condition_${comparison}_aa.csv" \
    "$WCT_OUT/${comparison}" aa "condition" "$pretty"

  echo ""
  echo "--- Codon: Within-Condition Timepoint Fisher ($pretty) ---"
  run_volcano "$INPUT_DIR/timepoint_fisher_within_condition_${comparison}_codon.csv" \
    "$WCT_OUT/${comparison}/codon" codon "condition" "$pretty"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
