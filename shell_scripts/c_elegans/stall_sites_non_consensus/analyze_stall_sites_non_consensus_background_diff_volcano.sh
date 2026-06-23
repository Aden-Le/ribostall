#!/bin/bash
#----------------------------------------------------
# Per-Timepoint Background-Aware Between-Condition Volcano Plots
# (NON-CONSENSUS stall_sites)
#
# Drives R_scripts/between_group_volcano.R on the per-timepoint background-aware
# between-condition CSVs emitted by stall_sites_non_consensus_stats.py
# (Analysis 5):
#   per_timepoint_background_diff_{aa,codon}.csv
#
# This is the background-aware counterpart of
# analyze_stall_sites_non_consensus_fisher_volcano.sh (per-timepoint Fisher,
# Analysis 4). Fisher compares raw stall-site shares between conditions; this
# test compares each condition's enrichment OVER ITS OWN background, so the
# x-axis effect size is `delta_log2_enrichment` (already log2 — an enrichment
# RATIO, not an odds ratio). between_group_volcano.R is reused via its generalized
# options:
#   --effect-col delta_log2_enrichment  (which column is the x-axis)
#   --effect-is-log2                    (it is already log2; do not re-log)
#   --x-label "..."                     (honest axis label, not 'Odds Ratio')
#
# Unlike the consensus background-diff launcher (which injects a constant
# `comparison` column because the consensus CSV has no grouping column), the
# non-consensus CSV already carries a `timepoint` column — one comparison per
# day — so we split into plots with --group-col timepoint directly, exactly like
# the non-consensus Fisher launcher. The stats CSV is read as-is.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ── CONFIG ───────────────────────────────────────────────────
INPUT_DIR="./results/c_elegans/stall_sites_non_consensus/analysis"
PLOTS_DIR="./results/c_elegans/stall_sites_non_consensus/plots"
# Shared headline/direction config (same file the stats runner sources). The
# comparison label and the x-axis direction (enrichment ratio, headline / other)
# are derived from the headline there, so they match the stats numerator and
# cannot drift. A positive delta_log2_enrichment = more enriched vs background in
# the headline condition.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"
X_LABEL="$X_LABEL_RATIO"
FORMAT="both"
DPI=300
# ─────────────────────────────────────────────────────────────

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

PT_OUT="$PLOTS_DIR/per_timepoint_background_diff"

echo "=============================================="
echo "STALL SITES PER-TIMEPOINT BACKGROUND-AWARE VOLCANO PLOTS"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $PT_OUT"
echo "Comparison:       $COMPARISON_LABEL"
echo "X-axis label:     $X_LABEL"
echo "Format:           $FORMAT  | DPI: $DPI"
echo "=============================================="

# =============================================
# Amino acid level
# Output: plots/per_timepoint_background_diff/{,codon/}
# =============================================
AA_SRC="$INPUT_DIR/per_timepoint_background_diff_aa.csv"
if [ ! -f "$AA_SRC" ]; then
  echo "Error: input CSV not found: $AA_SRC"
  exit 1
fi

echo ""
echo "--- AA: Per-Timepoint Background-Aware (BWM vs Control) ---"
CMD=(Rscript R_scripts/between_group_volcano.R \
  --input "$AA_SRC" \
  --outdir "$PT_OUT" \
  --level aa \
  --group-col "timepoint" \
  --comparison-label "$COMPARISON_LABEL" \
  --effect-col "delta_log2_enrichment" \
  --effect-is-log2 \
  --x-label "$X_LABEL" \
  --title-test-label "Background-Aware Enrichment" \
  --format "$FORMAT" --dpi "$DPI")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

# =============================================
# Codon level
# Output: plots/per_timepoint_background_diff/codon/
# =============================================
CODON_SRC="$INPUT_DIR/per_timepoint_background_diff_codon.csv"
if [ ! -f "$CODON_SRC" ]; then
  echo "Error: input CSV not found: $CODON_SRC"
  exit 1
fi

echo ""
echo "--- Codon: Per-Timepoint Background-Aware (BWM vs Control) ---"
CMD=(Rscript R_scripts/between_group_volcano.R \
  --input "$CODON_SRC" \
  --outdir "$PT_OUT/codon" \
  --level codon \
  --group-col "timepoint" \
  --comparison-label "$COMPARISON_LABEL" \
  --effect-col "delta_log2_enrichment" \
  --effect-is-log2 \
  --x-label "$X_LABEL" \
  --title-test-label "Background-Aware Enrichment" \
  --format "$FORMAT" --dpi "$DPI")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
