#!/bin/bash
#----------------------------------------------------
# Bash script: generate between-condition enrichment
# bar plots via R (stall_sites; BWM vs Control).
# Drives R_scripts/wilcoxon_barplot.R.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============

# Input directory (containing between_condition_wilcoxon_{aa,codon}.csv)
INPUT_DIR="./results/stall_sites/enrichment/analysis_stats"

# Output directory for plots
OUTPUT_DIR="./results/stall_sites/plots/between_condition"

# Output format: "pdf", "png", or "both"
FORMAT="png"

# DPI for PNG output
DPI=300

# ===============================================

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "STALL SITES BETWEEN-CONDITION BAR PLOTS"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Format:           $FORMAT"
echo "DPI:              $DPI"
echo "=============================================="

# --- AA: BWM vs Control ---
echo ""
echo "--- AA: BWM vs Control ---"
CMD=(Rscript R_scripts/wilcoxon_barplot.R \
  --input "$INPUT_DIR/between_condition_wilcoxon_aa.csv" \
  --outdir "$OUTPUT_DIR" \
  --level aa \
  --comparison "BWM_vs_Control" \
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
  --comparison "BWM_vs_Control" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
