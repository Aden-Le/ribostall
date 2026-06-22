#!/bin/bash
#----------------------------------------------------
# Background-Aware Between-Condition Volcano Plots (CONSENSUS stall_sites)
# Drives R_scripts/between_group_volcano.R on the background-aware between-condition
# CSVs emitted by stall_sites_consensus_stats.py:
#   between_condition_background_diff_{aa,codon}.csv
#
# This is the background-aware counterpart of
# analyze_stall_sites_consensus_fisher_volcano.sh. Fisher compares raw
# stall-site shares between conditions; this test compares each condition's
# enrichment OVER ITS OWN background, so the x-axis effect size is
# `delta_log2_enrichment` (already log2 — an enrichment RATIO, not an odds
# ratio). between_group_volcano.R is reused via its generalized options:
#   --effect-col delta_log2_enrichment  (which column is the x-axis)
#   --effect-is-log2                    (it is already log2; do not re-log)
#   --x-label "..."                     (honest axis label, not 'Odds Ratio')
#
# Like the Fisher launcher, the consensus CSV holds a SINGLE control-vs-
# treatment comparison with no timepoint/condition grouping column, and
# between_group_volcano.R needs a --group-col to split into plots, so this launcher
# injects a constant `comparison` column into a derived CSV (in the plots dir)
# before plotting. The original stats CSV is left untouched.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
INPUT_DIR="./results/stall_sites/enrichment/analysis_stats"
PLOTS_DIR="./results/stall_sites/plots/between_condition_background_diff"
# Shared headline/direction config (same file the stats runner sources). The
# comparison label, x-axis direction (enrichment ratio, headline / other), and
# injected grouping tag are derived from the headline there, so they match the
# stats numerator and cannot drift.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"
X_LABEL="$X_LABEL_RATIO"
FORMAT="both"
DPI=300
# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "CONSENSUS STALL SITES BACKGROUND-AWARE VOLCANO PLOTS"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $PLOTS_DIR"
echo "Comparison:       $COMPARISON_LABEL"
echo "X-axis label:     $X_LABEL"
echo "Format:           $FORMAT  | DPI: $DPI"
echo "=============================================="

# =============================================
# Amino acid level
# Output: plots/between_condition_background_diff/
# =============================================
AA_SRC="$INPUT_DIR/between_condition_background_diff_aa.csv"
if [ ! -f "$AA_SRC" ]; then
  echo "Error: input CSV not found: $AA_SRC"
  exit 1
fi
AA_OUT="$PLOTS_DIR"
mkdir -p "$AA_OUT"

# Derive a copy with a constant `comparison` column so between_group_volcano.R has
# a --group-col to split on. Header gets ",comparison"; data rows get the tag.
AA_DERIVED="$AA_OUT/_input_with_comparison_aa.csv"
awk -v tag="$COMPARISON_TAG" 'NR==1 {print $0",comparison"; next} {print $0","tag}' \
  "$AA_SRC" > "$AA_DERIVED"

echo ""
echo "--- AA: Background-Aware Between-Condition (Treatment vs Control) ---"
CMD=(Rscript R_scripts/between_group_volcano.R \
  --input "$AA_DERIVED" \
  --outdir "$AA_OUT" \
  --level aa \
  --group-col "comparison" \
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
# Output: plots/between_condition_background_diff/codon/
# =============================================
CODON_SRC="$INPUT_DIR/between_condition_background_diff_codon.csv"
if [ ! -f "$CODON_SRC" ]; then
  echo "Error: input CSV not found: $CODON_SRC"
  exit 1
fi
CODON_OUT="$PLOTS_DIR/codon"
mkdir -p "$CODON_OUT"

# Derive a copy with a constant `comparison` column so between_group_volcano.R has
# a --group-col to split on. Header gets ",comparison"; data rows get the tag.
CODON_DERIVED="$CODON_OUT/_input_with_comparison_codon.csv"
awk -v tag="$COMPARISON_TAG" 'NR==1 {print $0",comparison"; next} {print $0","tag}' \
  "$CODON_SRC" > "$CODON_DERIVED"

echo ""
echo "--- Codon: Background-Aware Between-Condition (Treatment vs Control) ---"
CMD=(Rscript R_scripts/between_group_volcano.R \
  --input "$CODON_DERIVED" \
  --outdir "$CODON_OUT" \
  --level codon \
  --group-col "comparison" \
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
