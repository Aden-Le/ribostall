#!/bin/bash
#----------------------------------------------------
# Background-Aware Volcano Plots (CONSENSUS UNION stall_sites) — TIMEPOINT MODE
# Drives R_scripts/between_group_volcano.R on the two background-aware CSVs that
# stall_sites_consensus_union_stats.py emits when run with --timepoints:
#   A4  per_timepoint_background_diff_{aa,codon}.csv    (BWM vs control at each day)
#   A7  between_timepoint_background_diff_{aa,codon}.csv (later vs earlier day, pooled)
#
# In the UNION design each group keeps its own filtered transcript set (and hence
# its own background), so these tests compare each condition/timepoint's
# enrichment OVER ITS OWN background. The x-axis effect size is
# `delta_log2_enrichment` — already log2, an enrichment RATIO, not an odds ratio —
# so between_group_volcano.R is driven via its generalized options:
#   --effect-col delta_log2_enrichment  (which column is the x-axis)
#   --effect-is-log2                    (it is already log2; do not re-log)
#   --x-label "..."                     (honest axis label, not 'Odds Ratio')
#
# Both CSVs already carry their own splitting column, so (unlike the old flat
# launcher) NO derived "comparison" column is injected:
#   A4 splits on `timepoint`   (day_0 / day_5 / day_10)
#   A7 splits on `comparison`  (d10_vs_d0 / d10_vs_d5 / d5_vs_d0)
#
# This is the timepoint-mode launcher matching the current analysis/ output. If
# the stats are ever re-run flat (no --timepoints), they emit
# between_condition_background_diff_* instead and this launcher should be revised.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
INPUT_DIR="./results/c_elegans/stall_sites_consensus_union/analysis"
PLOTS_DIR="./results/c_elegans/stall_sites_consensus_union/plots"
# Shared headline/direction config (same file the stats runner sources). The
# A4 comparison label and x-axis direction (BWM / control enrichment ratio) are
# derived from the headline there, so they match the stats numerator and cannot
# drift. A7 is fixed later-vs-earlier and ignores the headline.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"
FORMAT="both"
DPI=300
# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "CONSENSUS UNION STALL SITES BACKGROUND-AWARE VOLCANO PLOTS (TIMEPOINT MODE)"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $PLOTS_DIR"
echo "A4 comparison:    $COMPARISON_LABEL  | x-axis: $X_LABEL_RATIO"
echo "A7 comparison:    later vs earlier timepoint (pooled across conditions)"
echo "Format:           $FORMAT  | DPI: $DPI"
echo "=============================================="

# =============================================
# A4: Per-timepoint background-aware diff (BWM vs control at each day)
# Splits on `timepoint`. Output: plots/per_timepoint_background_diff/{,codon/}
# =============================================
PT_OUT="$PLOTS_DIR/per_timepoint_background_diff"

for LEVEL in aa codon; do
  SRC="$INPUT_DIR/per_timepoint_background_diff_${LEVEL}.csv"
  if [ ! -f "$SRC" ]; then
    echo "Error: input CSV not found: $SRC"
    exit 1
  fi
  if [ "$LEVEL" = "aa" ]; then OUT="$PT_OUT"; else OUT="$PT_OUT/codon"; fi

  echo ""
  echo "--- $LEVEL: Per-Timepoint Background-Aware Diff ($COMPARISON_LABEL) ---"
  CMD=(Rscript R_scripts/between_group_volcano.R \
    --input "$SRC" \
    --outdir "$OUT" \
    --level "$LEVEL" \
    --group-col "timepoint" \
    --comparison-label "$COMPARISON_LABEL" \
    --effect-col "delta_log2_enrichment" \
    --effect-is-log2 \
    --x-label "$X_LABEL_RATIO" \
    --title-test-label "Background-Aware Enrichment" \
    --composite-tag "binomial" \
    --format "$FORMAT" --dpi "$DPI")
  echo "Running: ${CMD[@]}"
  "${CMD[@]}"
done

# =============================================
# A7: Between-timepoint background-aware diff (later vs earlier, pooled)
# Splits on `comparison`. Output: plots/between_timepoint_background_diff/{,codon/}
# Direction is later-vs-earlier (positive = more enriched at the LATER day), so
# the x-axis label is fixed and independent of the headline condition.
# =============================================
BT_OUT="$PLOTS_DIR/between_timepoint_background_diff"
BT_LABEL="Later vs Earlier Timepoint"
BT_XLABEL="Log2 Enrichment Ratio (later / earlier)"

for LEVEL in aa codon; do
  SRC="$INPUT_DIR/between_timepoint_background_diff_${LEVEL}.csv"
  if [ ! -f "$SRC" ]; then
    echo "Error: input CSV not found: $SRC"
    exit 1
  fi
  if [ "$LEVEL" = "aa" ]; then OUT="$BT_OUT"; else OUT="$BT_OUT/codon"; fi

  echo ""
  echo "--- $LEVEL: Between-Timepoint Background-Aware Diff (later vs earlier) ---"
  CMD=(Rscript R_scripts/between_group_volcano.R \
    --input "$SRC" \
    --outdir "$OUT" \
    --level "$LEVEL" \
    --group-col "comparison" \
    --comparison-label "$BT_LABEL" \
    --effect-col "delta_log2_enrichment" \
    --effect-is-log2 \
    --x-label "$BT_XLABEL" \
    --title-test-label "Background-Aware Enrichment" \
    --composite-tag "binomial" \
    --format "$FORMAT" --dpi "$DPI")
  echo "Running: ${CMD[@]}"
  "${CMD[@]}"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
