#!/bin/bash
#----------------------------------------------------
# Bash script: run internal_stop_codons.py (mouse)
# Scans every CDS for in-frame internal stop codons (TAA/TAG/TGA) left in the
# coding body after the terminal stop is trimmed off, and pulls the per-replicate
# P-site read count at each one. Read-only diagnostic; writes two CSVs.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing the .ribo file and its _coverage.pkl.gz
RIBO_DIR="./all_ribo_file"
RIBO_FILE="$RIBO_DIR/mouse_all.ribo"

# Format: "group1:rep1,rep2;group2:rep1,rep2"
# Filters the coverage dict to declared replicates only (the read-count columns).
# Mouse run: 2-vs-1 design — control has 2 reps (AA_3, AA_4), treatment has 1 (Ch_WAA2).
EXP_GROUPS='control:AA_3,AA_4;treatment:Ch_WAA2'

# Scan window (codons trimmed before scanning). The terminal stop is the last
# CDS codon, so TRIM_STOP=1 removes it and any stop still found sits inside the
# coding body (a true internal stop). Set TRIM_STOP=0 to keep the terminal stop
# (then every transcript is reported).
TRIM_START=20
TRIM_STOP=10

# Reference file
REFERENCE_FILE="./reference/appris_mouse_v2_selected.fa.gz"

# Output directory
OUT_DIR="./results/mouse/internal_stop_codons"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root (three levels up from shell_scripts/<organism>/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# Coverage pickle for the exact ribo file above. adj_coverage.py writes
# <basename>_coverage.pkl.gz next to the .ribo, so derive it from RIBO_FILE
# rather than globbing (a glob would pick the wrong organism's pickle).
PICKLE="${RIBO_DIR}/$(basename "$RIBO_FILE" .ribo)_coverage.pkl.gz"

if [ ! -f "$PICKLE" ]; then
  echo "Error: coverage pickle not found: $PICKLE"
  exit 1
fi

echo "=============================================="
echo "INTERNAL STOP CODON SCAN (mouse)"
echo "=============================================="
echo "Coverage pickle: $PICKLE"
echo "Ribo file:       $RIBO_FILE"
echo "Reference:       $REFERENCE_FILE"
echo "Groups:          $EXP_GROUPS"
echo "Scan window:     trim_start=$TRIM_START, trim_stop=$TRIM_STOP codons"
echo "Output dir:      $OUT_DIR"
echo "=============================================="

CMD=(python3 scripts/internal_stop_codons.py \
  --ribo "$RIBO_FILE" \
  --pickle "$PICKLE" \
  --reference "$REFERENCE_FILE" \
  --groups "$EXP_GROUPS" \
  --trim-start "$TRIM_START" \
  --trim-stop "$TRIM_STOP" \
  --out-dir "$OUT_DIR")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done. CSVs written to $OUT_DIR"
date
echo "=============================================="
