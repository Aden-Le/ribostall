#!/bin/bash
#----------------------------------------------------
# Bash script: generate Wilcoxon bar plots via R for
# stall_sites enrichment (between-condition AND
# between-timepoint comparisons).
# Drives R_scripts/wilcoxon_barplot.R.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============

# Input directory (containing between_condition_wilcoxon_* and
# between_timepoint_wilcoxon_* CSVs)
INPUT_DIR="./results/stall_sites/enrichment/analysis_stats"

# Output directory for plots
OUTPUT_DIR="./results/stall_sites/plots/between_condition"

# Shared headline/direction config (same file the stats runner sources) so the
# between-condition comparison tag matches the stats numerator and cannot drift.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"

# Output format: "pdf", "png", or "both"
FORMAT="both"

# DPI for PNG output
DPI=300

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=============================================="
echo "STALL SITES WILCOXON BAR PLOTS"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Format:           $FORMAT"
echo "DPI:              $DPI"
echo "=============================================="

# =============================================
# Between-Condition (BWM vs Control)
# =============================================

# --- AA: BWM vs Control ---
echo ""
echo "--- AA: BWM vs Control ---"
CMD=(Rscript R_scripts/wilcoxon_barplot.R \
  --input "$INPUT_DIR/between_condition_wilcoxon_aa.csv" \
  --outdir "$OUTPUT_DIR" \
  --level aa \
  --comparison "$COMPARISON_TAG" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon: BWM vs Control ---
echo ""
echo "--- Codon: BWM vs Control ---"
CMD=(Rscript R_scripts/wilcoxon_barplot.R \
  --input "$INPUT_DIR/between_condition_wilcoxon_codon.csv" \
  --outdir "$OUTPUT_DIR/codon" \
  --level codon \
  --comparison "$COMPARISON_TAG" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# =============================================
# Between-Timepoint (day vs day, both levels)
# =============================================

BT_OUTPUT_DIR="./results/stall_sites/plots/between_timepoint"

for comparison in d10_vs_d0 d10_vs_d5 d5_vs_d0; do
  # Human-readable label: d10_vs_d0 → Day10_vs_Day0
  pretty=$(echo "$comparison" | sed 's/d\([0-9]\+\)/Day\1/g')

  # --- AA ---
  echo ""
  echo "--- AA: $pretty ---"
  CMD=(Rscript R_scripts/wilcoxon_barplot.R \
    --input "$INPUT_DIR/between_timepoint_wilcoxon_${comparison}_aa.csv" \
    --outdir "$BT_OUTPUT_DIR/${comparison}" \
    --level aa \
    --comparison "$pretty" \
    --format "$FORMAT" \
    --dpi "$DPI")
  echo "Running: ${CMD[@]}"
  "${CMD[@]}"

  # --- Codon ---
  echo ""
  echo "--- Codon: $pretty ---"
  CMD=(Rscript R_scripts/wilcoxon_barplot.R \
    --input "$INPUT_DIR/between_timepoint_wilcoxon_${comparison}_codon.csv" \
    --outdir "$BT_OUTPUT_DIR/${comparison}/codon" \
    --level codon \
    --comparison "$pretty" \
    --format "$FORMAT" \
    --dpi "$DPI")
  echo "Running: ${CMD[@]}"
  "${CMD[@]}"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
