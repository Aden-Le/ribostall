#!/bin/bash
#----------------------------------------------------
# Bash script: generate Fisher's exact test volcano plots
# for global codon and amino acid occupancy
# Runs both per-timepoint and within-condition-timepoint analyses
#----------------------------------------------------

# ============== CONFIG: edit these ==============
INPUT_DIR="./results/global_occupancy/analysis_corrected"
OUTPUT_DIR="./results/global_occupancy/plots/fisher"
FORMAT="png"       # pdf, png, or both
DPI=300
# ===============================================

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "GLOBAL OCCUPANCY FISHER VOLCANO PLOTS"
echo "=============================================="

# =============================================
# Per-Timepoint (BWM vs Control at each day)
# =============================================

# --- AA: Per-timepoint ---
echo ""
echo "--- AA: Per-Timepoint (BWM vs Control) ---"
CMD=(Rscript R_scripts/global_occupancy_fisher_volcano.R \
  --input "$INPUT_DIR/aa_per_timepoint_fisher.csv" \
  --outdir "$OUTPUT_DIR/aa_per_timepoint" \
  --level aa \
  --group-col "timepoint" \
  --comparison-label "BWM vs Control" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon: Per-timepoint ---
echo ""
echo "--- Codon: Per-Timepoint (BWM vs Control) ---"
CMD=(Rscript R_scripts/global_occupancy_fisher_volcano.R \
  --input "$INPUT_DIR/codon_per_timepoint_fisher.csv" \
  --outdir "$OUTPUT_DIR/codon_per_timepoint" \
  --level codon \
  --group-col "timepoint" \
  --comparison-label "BWM vs Control" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# =============================================
# Within-Condition Timepoint (Day 0 vs Day 10)
# =============================================

# --- AA: Within-condition timepoint ---
echo ""
echo "--- AA: Within-Condition Timepoint (Day 0 vs Day 10) ---"
CMD=(Rscript R_scripts/global_occupancy_fisher_volcano.R \
  --input "$INPUT_DIR/aa_timepoint_fisher_within_condition.csv" \
  --outdir "$OUTPUT_DIR/aa_within_condition_timepoint" \
  --level aa \
  --group-col "condition" \
  --comparison-label "Day 0 vs Day 10" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon: Within-condition timepoint ---
echo ""
echo "--- Codon: Within-Condition Timepoint (Day 0 vs Day 10) ---"
CMD=(Rscript R_scripts/global_occupancy_fisher_volcano.R \
  --input "$INPUT_DIR/codon_timepoint_fisher_within_condition.csv" \
  --outdir "$OUTPUT_DIR/codon_within_condition_timepoint" \
  --level codon \
  --group-col "condition" \
  --comparison-label "Day 0 vs Day 10" \
  --format "$FORMAT" \
  --dpi "$DPI")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
