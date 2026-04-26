#!/bin/bash
#----------------------------------------------------
# Bash script: generate Wilcoxon bar plots for global
# codon and amino acid occupancy fold-change
# Runs 4 combinations: (codon/aa) x (condition/timepoint)
#----------------------------------------------------

# ============== CONFIG: edit these ==============
INPUT_DIR="./results/global_occupancy/analysis_corrected"
OUTPUT_DIR="./results/global_occupancy/plots/wilcoxon"
FORMAT="png"       # pdf, png, or both
DPI=300
# ===============================================

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "GLOBAL OCCUPANCY WILCOXON BAR PLOTS"
echo "=============================================="

# --- AA: BWM vs Control ---
echo ""
echo "--- AA: BWM vs Control ---"
CMD=(Rscript R_scripts/global_occupancy_wilcoxon.R \
  --input "$INPUT_DIR/aa_wilcoxon_condition.csv" \
  --outdir "$OUTPUT_DIR/aa_condition" \
  --level aa \
  --comparison "BWM_vs_Control" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon: BWM vs Control ---
echo ""
echo "--- Codon: BWM vs Control ---"
CMD=(Rscript R_scripts/global_occupancy_wilcoxon.R \
  --input "$INPUT_DIR/codon_wilcoxon_condition.csv" \
  --outdir "$OUTPUT_DIR/codon_condition" \
  --level codon \
  --comparison "BWM_vs_Control" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- AA: Day 0 vs Day 10 ---
echo ""
echo "--- AA: Day 0 vs Day 10 ---"
CMD=(Rscript R_scripts/global_occupancy_wilcoxon.R \
  --input "$INPUT_DIR/aa_wilcoxon_timepoint.csv" \
  --outdir "$OUTPUT_DIR/aa_timepoint" \
  --level aa \
  --comparison "Day0_vs_Day10" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon: Day 0 vs Day 10 ---
echo ""
echo "--- Codon: Day 0 vs Day 10 ---"
CMD=(Rscript R_scripts/global_occupancy_wilcoxon.R \
  --input "$INPUT_DIR/codon_wilcoxon_timepoint.csv" \
  --outdir "$OUTPUT_DIR/codon_timepoint" \
  --level codon \
  --comparison "Day0_vs_Day10" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
