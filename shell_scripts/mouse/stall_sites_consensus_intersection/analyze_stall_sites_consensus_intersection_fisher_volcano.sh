#!/bin/bash
#----------------------------------------------------
# Fisher Volcano Plot Generator (CONSENSUS INTERSECTION stall_sites) — MOUSE, FLAT DESIGN
# Drives R_scripts/between_group_volcano.R on the Fisher CSVs emitted by
# stall_sites_consensus_intersection_stats.py.
#
# In the intersection design every group shares one transcript universe, so the
# raw stall-site shares Fisher compares are apples-to-apples.
#
# The mouse run is a FLAT control-vs-treatment design (no --timepoints), so the
# stats emit a single between-condition Fisher CSV per level (A3):
#   between_condition_fisher_{aa,codon}.csv
# and A6 (within-condition-timepoint Fisher) is skipped. These flat CSVs carry no
# timepoint/condition grouping column — only `site` — so we pass --flat-design:
# between_group_volcano.R then plots one composite row of A/P/E site panels for
# the single comparison and ignores --group-col.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
INPUT_DIR="./results/mouse/stall_sites_consensus_intersection/analysis"
PLOTS_DIR="./results/mouse/stall_sites_consensus_intersection/plots"
# Shared headline/direction config (same file the stats runner sources) so the
# between-condition label matches the stats numerator and cannot drift. A
# positive log2(odds ratio) = enriched in the headline condition.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"
FORMAT="both"       # pdf, png, or both
DPI=300
# ===============================================

# Navigate to repo root (three levels up from shell_scripts/<organism>/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "CONSENSUS INTERSECTION STALL SITES FISHER VOLCANO PLOTS (MOUSE, FLAT DESIGN)"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $PLOTS_DIR"
echo "Comparison:       $COMPARISON_LABEL"
echo "Format:           $FORMAT  | DPI: $DPI"
echo "=============================================="

# Run between_group_volcano.R on one flat between-condition Fisher CSV, skipping
# (with a note) if the input is absent — e.g. when the stats have not been run.
run_volcano() {
  local input="$1" outdir="$2" level="$3" label="$4"
  if [ ! -f "$input" ]; then
    echo "  (skip) input not found: $input"
    return
  fi
  echo "Running: between_group_volcano.R --input $input --flat-design"
  Rscript R_scripts/between_group_volcano.R \
    --input "$input" \
    --outdir "$outdir" \
    --level "$level" \
    --flat-design \
    --comparison-label "$label" \
    --composite-tag "fisher" \
    --format "$FORMAT" --dpi "$DPI"
}

# =============================================
# A3 — Between-Condition Fisher (treatment vs control)
# Output: plots/between_condition_fisher/{,codon/}
# =============================================
BC_OUT="$PLOTS_DIR/between_condition_fisher"

echo ""
echo "--- AA: Between-Condition Fisher ($COMPARISON_LABEL) ---"
run_volcano "$INPUT_DIR/between_condition_fisher_aa.csv" \
  "$BC_OUT" aa "$COMPARISON_LABEL"

echo ""
echo "--- Codon: Between-Condition Fisher ($COMPARISON_LABEL) ---"
run_volcano "$INPUT_DIR/between_condition_fisher_codon.csv" \
  "$BC_OUT/codon" codon "$COMPARISON_LABEL"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
