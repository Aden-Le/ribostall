#!/bin/bash
#----------------------------------------------------
# Fisher Volcano Plot Generator (CONSENSUS stall_sites)
# Drives R_scripts/between_group_volcano.R on the between-condition
# Fisher CSVs emitted by stall_sites_consensus_stats.py:
#   between_condition_fisher_{aa,codon}.csv
#
# Unlike the non-consensus per_timepoint_fisher output, the consensus
# between-condition CSV holds a SINGLE control-vs-treatment comparison
# and therefore has no timepoint/condition grouping column. fisher_volcano.R
# requires a --group-col to split into individual plots, so this launcher
# injects a constant `comparison` column into a derived CSV (in the plots
# dir) before plotting. The original stats CSV is left untouched.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
INPUT_DIR="./results/stall_sites/enrichment/analysis_stats"
PLOTS_DIR="./results/stall_sites/plots/between_condition_fisher"
# Shared headline/direction config (same file the stats runner sources) so the
# comparison label + injected grouping tag match the stats numerator and cannot
# drift. A positive log2(odds ratio) = enriched in the headline condition.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"
FORMAT="both"
DPI=300
# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=============================================="
echo "CONSENSUS STALL SITES FISHER VOLCANO PLOTS"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $PLOTS_DIR"
echo "Comparison:       $COMPARISON_LABEL"
echo "Format:           $FORMAT  | DPI: $DPI"
echo "=============================================="

# =============================================
# Amino acid level
# Output: plots/between_condition_fisher/
# =============================================
AA_SRC="$INPUT_DIR/between_condition_fisher_aa.csv"
if [ ! -f "$AA_SRC" ]; then
  echo "Error: input CSV not found: $AA_SRC"
  exit 1
fi
AA_OUT="$PLOTS_DIR"
mkdir -p "$AA_OUT"

# Derive a copy with a constant `comparison` column so fisher_volcano.R has
# a --group-col to split on. Header gets ",comparison"; data rows get the tag.
AA_DERIVED="$AA_OUT/_input_with_comparison_aa.csv"
awk -v tag="$COMPARISON_TAG" 'NR==1 {print $0",comparison"; next} {print $0","tag}' \
  "$AA_SRC" > "$AA_DERIVED"

echo ""
echo "--- AA: Between-Condition Fisher (Treatment vs Control) ---"
CMD=(Rscript R_scripts/between_group_volcano.R \
  --input "$AA_DERIVED" \
  --outdir "$AA_OUT" \
  --level aa \
  --group-col "comparison" \
  --comparison-label "$COMPARISON_LABEL" \
  --format "$FORMAT" --dpi "$DPI")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

# =============================================
# Codon level
# Output: plots/between_condition_fisher/codon/
# =============================================
CODON_SRC="$INPUT_DIR/between_condition_fisher_codon.csv"
if [ ! -f "$CODON_SRC" ]; then
  echo "Error: input CSV not found: $CODON_SRC"
  exit 1
fi
CODON_OUT="$PLOTS_DIR/codon"
mkdir -p "$CODON_OUT"

# Derive a copy with a constant `comparison` column so fisher_volcano.R has
# a --group-col to split on. Header gets ",comparison"; data rows get the tag.
CODON_DERIVED="$CODON_OUT/_input_with_comparison_codon.csv"
awk -v tag="$COMPARISON_TAG" 'NR==1 {print $0",comparison"; next} {print $0","tag}' \
  "$CODON_SRC" > "$CODON_DERIVED"

echo ""
echo "--- Codon: Between-Condition Fisher (Treatment vs Control) ---"
CMD=(Rscript R_scripts/between_group_volcano.R \
  --input "$CODON_DERIVED" \
  --outdir "$CODON_OUT" \
  --level codon \
  --group-col "comparison" \
  --comparison-label "$COMPARISON_LABEL" \
  --format "$FORMAT" --dpi "$DPI")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
