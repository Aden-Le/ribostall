#!/bin/bash
# Fisher Volcano Plot Generator (stall_sites)
# Drives R_scripts/between_group_volcano.R on the per_timepoint_fisher AND
# timepoint_fisher_within_condition CSVs emitted by
# stall_sites_non_consensus_stats.py.

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ── CONFIG ───────────────────────────────────────────────────
INPUT_DIR="./results/stall_sites/enrichment/analysis_stats"
PLOTS_DIR="./results/stall_sites/plots"
# Shared headline/direction config (same file the stats runner sources) so the
# per-timepoint BWM-vs-Control label matches the stats numerator and cannot drift.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"
FORMAT="both"
DPI=300
# ─────────────────────────────────────────────────────────────

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "STALL SITES FISHER VOLCANO PLOTS"
echo "=============================================="

# =============================================
# Per-Timepoint (BWM vs Control at each day)
# Output: plots/per_timepoint_fisher/{,codon/}
# =============================================

PT_OUT="$PLOTS_DIR/per_timepoint_fisher"

echo ""
echo "--- AA: Per-Timepoint (BWM vs Control) ---"
CMD=(Rscript R_scripts/between_group_volcano.R \
  --input "$INPUT_DIR/per_timepoint_fisher_aa.csv" \
  --outdir "$PT_OUT" \
  --level aa \
  --group-col "timepoint" \
  --comparison-label "$COMPARISON_LABEL" \
  --format "$FORMAT" --dpi "$DPI")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "--- Codon: Per-Timepoint (BWM vs Control) ---"
CMD=(Rscript R_scripts/between_group_volcano.R \
  --input "$INPUT_DIR/per_timepoint_fisher_codon.csv" \
  --outdir "$PT_OUT/codon" \
  --level codon \
  --group-col "timepoint" \
  --comparison-label "$COMPARISON_LABEL" \
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
  CMD=(Rscript R_scripts/between_group_volcano.R \
    --input "$INPUT_DIR/timepoint_fisher_within_condition_${comparison}_aa.csv" \
    --outdir "$WCT_OUT/${comparison}" \
    --level aa \
    --group-col "condition" \
    --comparison-label "$pretty" \
    --format "$FORMAT" --dpi "$DPI")
  echo "Running: ${CMD[@]}"
  "${CMD[@]}"

  echo ""
  echo "--- Codon: Within-Condition Timepoint ($pretty) ---"
  CMD=(Rscript R_scripts/between_group_volcano.R \
    --input "$INPUT_DIR/timepoint_fisher_within_condition_${comparison}_codon.csv" \
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
