#!/bin/bash
#----------------------------------------------------
# Bash script: generate Wilcoxon bar plots for global
# codon and amino acid occupancy fold-change.
# Runs both between-condition and between-timepoint Wilcoxons.
# Drives R_scripts/wilcoxon_barplot.R.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
INPUT_DIR="./results/global_occupancy/analysis_corrected"
PLOTS_DIR="./results/global_occupancy/plots"
# Shared headline/direction config (same file the stats runner sources) so the
# between-condition comparison tag matches the stats numerator and cannot drift.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"
FORMAT="both"       # pdf, png, or both
DPI=300
# ===============================================

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=============================================="
echo "GLOBAL OCCUPANCY WILCOXON BAR PLOTS"
echo "=============================================="

# =============================================
# Between-Condition (BWM vs Control, pooled across timepoints)
# Output: plots/between_condition/{,codon/}
# =============================================

BC_OUT="$PLOTS_DIR/between_condition"

echo ""
echo "--- AA: BWM vs Control ---"
CMD=(Rscript R_scripts/wilcoxon_barplot.R \
  --input "$INPUT_DIR/aa_wilcoxon_condition.csv" \
  --outdir "$BC_OUT" \
  --level aa \
  --comparison "$COMPARISON_TAG" \
  --format "$FORMAT" --dpi "$DPI")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "--- Codon: BWM vs Control ---"
CMD=(Rscript R_scripts/wilcoxon_barplot.R \
  --input "$INPUT_DIR/codon_wilcoxon_condition.csv" \
  --outdir "$BC_OUT/codon" \
  --level codon \
  --comparison "$COMPARISON_TAG" \
  --format "$FORMAT" --dpi "$DPI")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

# =============================================
# Between-Timepoint (day vs day, pooled across conditions)
# Output: plots/between_timepoint/{comparison}/{,codon/}
# =============================================

BT_OUT="$PLOTS_DIR/between_timepoint"

for comparison in d10_vs_d0 d10_vs_d5 d5_vs_d0; do
  pretty=$(echo "$comparison" | sed 's/d\([0-9]\+\)/Day\1/g')

  echo ""
  echo "--- AA: $pretty ---"
  CMD=(Rscript R_scripts/wilcoxon_barplot.R \
    --input "$INPUT_DIR/aa_wilcoxon_timepoint_${comparison}.csv" \
    --outdir "$BT_OUT/${comparison}" \
    --level aa \
    --comparison "$pretty" \
    --format "$FORMAT" --dpi "$DPI")
  echo "Running: ${CMD[@]}"
  "${CMD[@]}"

  echo ""
  echo "--- Codon: $pretty ---"
  CMD=(Rscript R_scripts/wilcoxon_barplot.R \
    --input "$INPUT_DIR/codon_wilcoxon_timepoint_${comparison}.csv" \
    --outdir "$BT_OUT/${comparison}/codon" \
    --level codon \
    --comparison "$pretty" \
    --format "$FORMAT" --dpi "$DPI")
  echo "Running: ${CMD[@]}"
  "${CMD[@]}"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
