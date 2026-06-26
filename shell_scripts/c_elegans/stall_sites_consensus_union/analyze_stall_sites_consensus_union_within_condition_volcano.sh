#!/bin/bash
#----------------------------------------------------
# Within-Condition Enrichment Volcano Plots (CONSENSUS UNION stall_sites) — TIMEPOINT MODE
# Drives R_scripts/within_condition_volcano.R on the within-condition
# binomial CSVs emitted by stall_sites_consensus_union_stats.py:
#   within_condition_binomial_{aa,codon}.csv
#
# Run with --timepoints, the stats carry REAL timepoints, so the CSV's
# site/group/condition/timepoint columns span condition x timepoint
# (BWM/control x day_0/5/10). The R script consumes it directly with no
# preprocessing and builds the by-condition and by-day composites (NOT the flat
# per-group layout — so --flat-design is intentionally omitted here).
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
# Input directory (containing within_condition_binomial_{aa,codon}.csv)
INPUT_DIR="./results/c_elegans/stall_sites_consensus_union/analysis"

# Output directory for plots
OUTPUT_DIR="./results/c_elegans/stall_sites_consensus_union/plots/within_condition"

# Enrichment type: "unweighted", "weighted", or "both"
ENRICHMENT_TYPE="both"

# Show confidence intervals: set to "--show-ci" to enable, leave empty to disable
SHOW_CI="--show-ci"

# Output format: "pdf", "png", or "both"
FORMAT="both"

# DPI for PNG output
DPI=300

# Y-axis cap: clamp -log10(p_adj) values above this to compress the y-axis.
# Leave empty ("") to use the R script default (50).
Y_CAP=25
# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "CONSENSUS UNION STALL SITES WITHIN-CONDITION VOLCANO PLOTS"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Enrichment type:  $ENRICHMENT_TYPE"
echo "Format:           $FORMAT"
echo "DPI:              $DPI"
echo "Show CI:          ${SHOW_CI:-no}"
echo "Y-axis cap:       ${Y_CAP:-default}"
echo "=============================================="

# Build optional flags.
# --mega-composite: also emit the all-groups grid (rows = condition x timepoint,
# cols = sites). NO --flat-design: the stats are run with --timepoints, so the
# CSV spans condition x timepoint and the R script builds the by-condition and
# by-day composites over that cross-product.
OPTIONAL_FLAGS=(--mega-composite)
[ -n "$SHOW_CI" ] && OPTIONAL_FLAGS+=("$SHOW_CI")
[ -n "$Y_CAP" ]   && OPTIONAL_FLAGS+=(--y-cap "$Y_CAP")

# --- Amino acid level ---
echo ""
echo "--- Amino Acid Level ---"
CMD=(Rscript R_scripts/within_condition_volcano.R \
  --input "$INPUT_DIR/within_condition_binomial_aa.csv" \
  --outdir "$OUTPUT_DIR" \
  --level aa \
  --enrichment-type "$ENRICHMENT_TYPE" \
  --format "$FORMAT" \
  --dpi "$DPI" \
  "${OPTIONAL_FLAGS[@]}")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon level ---
echo ""
echo "--- Codon Level ---"
CMD=(Rscript R_scripts/within_condition_volcano.R \
  --input "$INPUT_DIR/within_condition_binomial_codon.csv" \
  --outdir "$OUTPUT_DIR/codon" \
  --level codon \
  --enrichment-type "$ENRICHMENT_TYPE" \
  --format "$FORMAT" \
  --dpi "$DPI" \
  "${OPTIONAL_FLAGS[@]}")
echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
