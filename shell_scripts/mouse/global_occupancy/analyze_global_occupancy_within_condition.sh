#!/bin/bash
#----------------------------------------------------
# Bash script: generate within-condition bar plots
# for global codon and amino acid occupancy enrichment (MOUSE).
#
# The mouse design is a FLAT control-vs-treatment comparison with no timepoints,
# so the within_condition_binomial CSVs carry group == condition and a degenerate
# timepoint column. We pass --flat-design so within_condition_volcano.R labels by
# the group (condition) alone and builds the flat per-group composites instead of
# the by-condition/by-day grids it would build for a timepoint design.
#----------------------------------------------------

# Add R to PATH (Windows)
export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"

# ============== CONFIG: edit these ==============

# Input directory containing the binomial CSVs
INPUT_DIR="./results/mouse/global_occupancy/analysis"

# Output directory for plots
OUTPUT_DIR="./results/mouse/global_occupancy/plots/within_condition"

# Enrichment type: "unweighted", "weighted", or "both"
# Global occupancy normalizes every condition to one shared transcriptome
# background, so the frequency-weighted enrichment adds little here — keep it
# unweighted-only (the weighted volcanoes were dropped from the global reports).
ENRICHMENT_TYPE="unweighted"

# Output format: "pdf", "png", or "both"
FORMAT="both"

# DPI for PNG output
DPI=300

# Y-axis cap: clamp y-axis to ±this value
# Leave empty ("") to disable
Y_CAP=""

# ===============================================

# Navigate to repo root (three levels up from shell_scripts/<organism>/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "=============================================="
echo "GLOBAL OCCUPANCY WITHIN-CONDITION BAR PLOTS (MOUSE, FLAT DESIGN)"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Enrichment type:  $ENRICHMENT_TYPE"
echo "Format:           $FORMAT"
echo "DPI:              $DPI"
echo "Y-axis cap:       ${Y_CAP:-none}"
echo "=============================================="

# Build optional flags.
# --flat-design: no timepoint axis (group == condition), so build the flat
#   per-group composites, not the timepoint grids.
# --mega-composite: preserve global's prior behaviour (an all-groups grid; the
#   unified volcano script gates this behind a flag so stall_sites runs do not
#   produce it).
OPTIONAL_FLAGS=(--flat-design --mega-composite)
if [ -n "$Y_CAP" ]; then
  OPTIONAL_FLAGS+=(--y-cap "$Y_CAP")
fi

# --- Amino acid level (AA plots at the within_condition/ root) ---
echo ""
echo "--- Amino Acid Level ---"
CMD=(Rscript R_scripts/within_condition_volcano.R \
  --input "$INPUT_DIR/aa_within_condition_binomial.csv" \
  --outdir "$OUTPUT_DIR" \
  --level aa \
  --enrichment-type "$ENRICHMENT_TYPE" \
  --format "$FORMAT" \
  --dpi "$DPI" \
  "${OPTIONAL_FLAGS[@]}")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

# --- Codon level (codon/ subdir) ---
echo ""
echo "--- Codon Level ---"
CMD=(Rscript R_scripts/within_condition_volcano.R \
  --input "$INPUT_DIR/codon_within_condition_binomial.csv" \
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
