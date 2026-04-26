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

# --- AA: Day 10 vs Day 0 ---
echo ""
echo "--- AA: Day 10 vs Day 0 ---"
CMD=(Rscript R_scripts/global_occupancy_wilcoxon.R \
  --input "$INPUT_DIR/aa_wilcoxon_timepoint_d10_vs_d0.csv" \
  --outdir "$OUTPUT_DIR/aa_timepoint_d10_vs_d0" \
  --level aa \
  --comparison "Day10_vs_Day0" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon: Day 10 vs Day 0 ---
echo ""
echo "--- Codon: Day 10 vs Day 0 ---"
CMD=(Rscript R_scripts/global_occupancy_wilcoxon.R \
  --input "$INPUT_DIR/codon_wilcoxon_timepoint_d10_vs_d0.csv" \
  --outdir "$OUTPUT_DIR/codon_timepoint_d10_vs_d0" \
  --level codon \
  --comparison "Day10_vs_Day0" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- AA: Day 10 vs Day 5 ---
echo ""
echo "--- AA: Day 10 vs Day 5 ---"
CMD=(Rscript R_scripts/global_occupancy_wilcoxon.R \
  --input "$INPUT_DIR/aa_wilcoxon_timepoint_d10_vs_d5.csv" \
  --outdir "$OUTPUT_DIR/aa_timepoint_d10_vs_d5" \
  --level aa \
  --comparison "Day10_vs_Day5" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon: Day 10 vs Day 5 ---
echo ""
echo "--- Codon: Day 10 vs Day 5 ---"
CMD=(Rscript R_scripts/global_occupancy_wilcoxon.R \
  --input "$INPUT_DIR/codon_wilcoxon_timepoint_d10_vs_d5.csv" \
  --outdir "$OUTPUT_DIR/codon_timepoint_d10_vs_d5" \
  --level codon \
  --comparison "Day10_vs_Day5" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- AA: Day 5 vs Day 0 ---
echo ""
echo "--- AA: Day 5 vs Day 0 ---"
CMD=(Rscript R_scripts/global_occupancy_wilcoxon.R \
  --input "$INPUT_DIR/aa_wilcoxon_timepoint_d5_vs_d0.csv" \
  --outdir "$OUTPUT_DIR/aa_timepoint_d5_vs_d0" \
  --level aa \
  --comparison "Day5_vs_Day0" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon: Day 5 vs Day 0 ---
echo ""
echo "--- Codon: Day 5 vs Day 0 ---"
CMD=(Rscript R_scripts/global_occupancy_wilcoxon.R \
  --input "$INPUT_DIR/codon_wilcoxon_timepoint_d5_vs_d0.csv" \
  --outdir "$OUTPUT_DIR/codon_timepoint_d5_vs_d0" \
  --level codon \
  --comparison "Day5_vs_Day0" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
