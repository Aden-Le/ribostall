#!/bin/bash
#----------------------------------------------------
# Background-Aware Volcano Plots (CONSENSUS UNION stall_sites) — MOUSE, FLAT DESIGN
# Drives R_scripts/between_group_volcano.R on the background-aware CSV that
# stall_sites_consensus_union_stats.py emits for a flat run:
#   A4  between_condition_background_diff_{aa,codon}.csv  (treatment vs control)
# (A7, the between-timepoint diff, is skipped — the mouse run has no timepoints.)
#
# In the UNION design each group keeps its own filtered transcript set (and hence
# its own background), so this test compares each condition's enrichment OVER ITS
# OWN background. The x-axis effect size is `delta_log2_enrichment` — already
# log2, an enrichment RATIO, not an odds ratio — so between_group_volcano.R is
# driven via its generalized options:
#   --effect-col delta_log2_enrichment  (which column is the x-axis)
#   --effect-is-log2                    (it is already log2; do not re-log)
#   --x-label "..."                     (honest axis label, not 'Odds Ratio')
#
# The flat CSV carries no timepoint/condition grouping column — only `site` — so
# we pass --flat-design: between_group_volcano.R then plots one composite row of
# A/P/E site panels for the single comparison and ignores --group-col.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
INPUT_DIR="./results/mouse/stall_sites_consensus_union/analysis"
PLOTS_DIR="./results/mouse/stall_sites_consensus_union/plots"
# Shared headline/direction config (same file the stats runner sources). The
# comparison label and x-axis direction (treatment / control enrichment ratio)
# are derived from the headline there, so they match the stats numerator and
# cannot drift.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"
FORMAT="both"
DPI=300
# ===============================================

# Navigate to repo root (three levels up from shell_scripts/<organism>/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "CONSENSUS UNION STALL SITES BACKGROUND-AWARE VOLCANO PLOTS (MOUSE, FLAT DESIGN)"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $PLOTS_DIR"
echo "Comparison:       $COMPARISON_LABEL  | x-axis: $X_LABEL_RATIO"
echo "Format:           $FORMAT  | DPI: $DPI"
echo "=============================================="

# =============================================
# A4: Between-condition background-aware diff (treatment vs control)
# Splits on `site` only (flat). Output: plots/between_condition_background_diff/{,codon/}
# =============================================
BC_OUT="$PLOTS_DIR/between_condition_background_diff"

for LEVEL in aa codon; do
  SRC="$INPUT_DIR/between_condition_background_diff_${LEVEL}.csv"
  if [ ! -f "$SRC" ]; then
    echo "  (skip) input not found: $SRC"
    continue
  fi
  if [ "$LEVEL" = "aa" ]; then OUT="$BC_OUT"; else OUT="$BC_OUT/codon"; fi

  echo ""
  echo "--- $LEVEL: Between-Condition Background-Aware Diff ($COMPARISON_LABEL) ---"
  CMD=(Rscript R_scripts/between_group_volcano.R \
    --input "$SRC" \
    --outdir "$OUT" \
    --level "$LEVEL" \
    --flat-design \
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

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
