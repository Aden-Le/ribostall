#!/bin/bash
#----------------------------------------------------
# Bash script: generate Fisher's exact test volcano plots
# for global codon and amino acid occupancy.
# Runs both per-timepoint and within-condition-timepoint Fishers.
# Drives R_scripts/fisher_volcano.R.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
INPUT_DIR="./results/global_occupancy/analysis_corrected"
PLOTS_DIR="./results/global_occupancy/plots"
FORMAT="both"       # pdf, png, or both
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
# Output: plots/per_timepoint_fisher/{,codon/}
# =============================================

PT_OUT="$PLOTS_DIR/per_timepoint_fisher"

echo ""
echo "--- AA: Per-Timepoint (BWM vs Control) ---"
CMD=(Rscript R_scripts/fisher_volcano.R \
  --input "$INPUT_DIR/aa_per_timepoint_fisher.csv" \
  --outdir "$PT_OUT" \
  --level aa \
  --group-col "timepoint" \
  --comparison-label "BWM vs Control" \
  --format "$FORMAT" --dpi "$DPI")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "--- Codon: Per-Timepoint (BWM vs Control) ---"
CMD=(Rscript R_scripts/fisher_volcano.R \
  --input "$INPUT_DIR/codon_per_timepoint_fisher.csv" \
  --outdir "$PT_OUT/codon" \
  --level codon \
  --group-col "timepoint" \
  --comparison-label "BWM vs Control" \
  --format "$FORMAT" --dpi "$DPI")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

# =============================================
# Within-Condition Timepoint Fisher (day vs day, within each condition)
# Output: plots/within_condition_timepoint_fisher/{comparison}/{,codon/}
# =============================================

WCT_OUT="$PLOTS_DIR/within_condition_timepoint_fisher"

for comparison in d10_vs_d0 d10_vs_d5 d5_vs_d0; do
  pretty=$(echo "$comparison" | sed 's/d\([0-9]\+\)/Day \1/g; s/_vs_/ vs /')

  echo ""
  echo "--- AA: Within-Condition Timepoint ($pretty) ---"
  CMD=(Rscript R_scripts/fisher_volcano.R \
    --input "$INPUT_DIR/aa_timepoint_fisher_within_condition_${comparison}.csv" \
    --outdir "$WCT_OUT/${comparison}" \
    --level aa \
    --group-col "condition" \
    --comparison-label "$pretty" \
    --format "$FORMAT" --dpi "$DPI")
  echo "Running: ${CMD[@]}"
  "${CMD[@]}"

  echo ""
  echo "--- Codon: Within-Condition Timepoint ($pretty) ---"
  CMD=(Rscript R_scripts/fisher_volcano.R \
    --input "$INPUT_DIR/codon_timepoint_fisher_within_condition_${comparison}.csv" \
    --outdir "$WCT_OUT/${comparison}/codon" \
    --level codon \
    --group-col "condition" \
    --comparison-label "$pretty" \
    --format "$FORMAT" --dpi "$DPI")
  echo "Running: ${CMD[@]}"
  "${CMD[@]}"
done

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
