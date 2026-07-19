#!/bin/bash
#----------------------------------------------------
# Within-Condition Enrichment Volcano Plots (CONSENSUS UNION stall_sites) — MOUSE, FLAT DESIGN
# Drives R_scripts/within_condition_volcano.R on the within-condition
# binomial CSVs (A1) emitted by stall_sites_consensus_union_stats.py:
#   within_condition_binomial_{aa,codon}.csv
#
# The mouse run is a FLAT control-vs-treatment design (no --timepoints), so the
# CSV carries group == condition and a degenerate timepoint column. We pass
# --flat-design: the R script labels each composite by the group (condition)
# alone and builds the flat per-group layout (rows = group, cols = sites) rather
# than the by-condition / by-day grids it builds for a timepoint design.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
# Input directory (containing within_condition_binomial_{aa,codon}.csv)
INPUT_DIR="./results/mouse/stall_sites_consensus_union/analysis"

# Output directory for plots
OUTPUT_DIR="./results/mouse/stall_sites_consensus_union/plots/within_condition"

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

# Navigate to repo root (three levels up from shell_scripts/<organism>/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "CONSENSUS UNION STALL SITES WITHIN-CONDITION VOLCANO PLOTS (MOUSE, FLAT DESIGN)"
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
# --flat-design: no timepoint axis (group == condition), so build the flat
#   per-group composites (rows = group, cols = sites), not the timepoint grids.
# --mega-composite: also emit the all-groups grid.
OPTIONAL_FLAGS=(--flat-design --mega-composite)
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
