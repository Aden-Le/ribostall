#!/bin/bash
#----------------------------------------------------
# Within-Condition Enrichment Volcano Plots (CONSENSUS INTERSECTION stall_sites)
# Drives R_scripts/within_condition_volcano.R on the within-condition
# binomial CSVs emitted by stall_sites_consensus_intersection_stats.py:
#   within_condition_binomial_{aa,codon}.csv
#
# The consensus within-condition CSV already carries site/group/condition/
# timepoint columns (timepoint == condition for this flat control-vs-treatment
# design), so the R script consumes it directly with no preprocessing.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============
# Input directory (containing within_condition_binomial_{aa,codon}.csv)
INPUT_DIR="./results/c_elegans/stall_sites_consensus_intersection/analysis"

# Output directory for plots
OUTPUT_DIR="./results/c_elegans/stall_sites_consensus_intersection/plots/within_condition"

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
echo "CONSENSUS INTERSECTION STALL SITES WITHIN-CONDITION VOLCANO PLOTS"
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
# --flat-design: the consensus stats output is a flat control-vs-treatment
# design with no timepoint axis (group == condition, timepoint == condition).
# It tells within_condition_volcano.R to build composites per group rather than
# over the condition x timepoint cross-product (which would be empty here).
OPTIONAL_FLAGS=(--mega-composite --flat-design)
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
