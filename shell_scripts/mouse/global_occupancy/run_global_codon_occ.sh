#!/bin/bash
#----------------------------------------------------
# Bash script: run global_codon_occ.py (MOUSE)
# Computes per-experiment codon and amino acid occupancy
# and saves base CSVs to out_dir/raw/
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing .ribo file and _coverage.pkl.gz
RIBO_DIR="./all_ribo_file"
RIBO_FILE="$RIBO_DIR/mouse_all.ribo"

# Format: "group1:rep1,rep2;group2:rep1,rep2"
# Used to filter the coverage dict to declared replicates only.
# Flat control-vs-treatment design: control has 2 reps (AA_3, AA_4), treatment 1 (Ch_WAA2).
EXP_GROUPS='control:AA_3,AA_4;treatment:Ch_WAA2'

# Trimming parameters
TRIM_START=20
TRIM_STOP=10

# Output filtering: exclude stop codons (TAA/TAG/TGA) before computing occupancy
# ("True"/"False"). Default True drops them from background, totals, rates,
# proportions, and rpm. Pass "False" to keep them.
DROP_STOP_CODONS="True"

# Reference file
REFERENCE_FILE="./reference/appris_mouse_v2_selected.fa.gz"

# Output directory (must match --out-dir used in run_global_codon_occ_stats.sh)
OUT_DIR="./results/mouse/global_occupancy"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root (three levels up from shell_scripts/<organism>/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# Coverage pickle for the exact ribo file above (not a directory glob).
# adj_coverage.py writes <basename>_coverage.pkl.gz next to the .ribo, so derive
# it from RIBO_FILE. With multiple *_coverage.pkl.gz present (e.g. C. elegans +
# mouse), a glob would pick the wrong one.
PICKLE="${RIBO_DIR}/$(basename "$RIBO_FILE" .ribo)_coverage.pkl.gz"

if [ ! -f "$PICKLE" ]; then
  echo "Error: coverage pickle not found: $PICKLE"
  exit 1
fi

echo "=============================================="
echo "GLOBAL CODON & AMINO ACID OCCUPANCY — STEP 1"
echo "Compute occupancy tables and save base CSVs"
echo "=============================================="
echo "Coverage pickle: $PICKLE"
echo "Ribo file:       $RIBO_FILE"
echo "Reference:       $REFERENCE_FILE"
echo "Groups:          $EXP_GROUPS"
echo "Parameters:      trim_start=$TRIM_START, trim_stop=$TRIM_STOP, drop_stop_codons=$DROP_STOP_CODONS"
echo "Output directory: $OUT_DIR"
echo "=============================================="

CMD=(python3 scripts/global_codon_occ.py \
  --pickle "$PICKLE" \
  --ribo "$RIBO_FILE" \
  --reference "$REFERENCE_FILE" \
  --groups "$EXP_GROUPS" \
  --trim-start "$TRIM_START" \
  --trim-stop "$TRIM_STOP" \
  --drop-stop-codons "$DROP_STOP_CODONS" \
  --out-dir "$OUT_DIR")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done. Base CSVs written to $OUT_DIR/raw/"
echo "Run run_global_codon_occ_stats.sh to run statistical tests."
date
echo "=============================================="
