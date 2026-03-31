#!/bin/bash
#----------------------------------------------------
# Bash script: run stall_sites_consensus.py with
# consensus stall site calling across replicates
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing _coverage.pkl.gz files
RIBO_DIR="./all_ribo_file"
RIBO_FILE="$RIBO_DIR/C_elegan_all_02_04_2026.ribo"

# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Transcript filtering thresholds
TX_THRESHOLD=1.0
TX_MIN_REPS=2

# Stall site calling thresholds
MIN_Z=1.0
MIN_READS=2
TRIM_EDGES=10
PSEUDOCOUNT=0.5

# Consensus calling parameters
STALL_MIN_REPS=2
TOL=0
MIN_SEP=7

# Reference file (required for motif analysis)
REFERENCE_FILE="./C_elegan_reference/appris_celegans_v1_selected_new.fa"

# Output files
OUT_CSV="../ribostall_results/stall_sites.csv"
OUT_PNG="../ribostall_results/motif.png"
OUT_CSV="../ribostall_results/motif_csv"

# Set to "--motif" to enable motif plotting, or "" to skip
MOTIF_FLAG="--motif"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root (one level up from shell_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Find coverage pickle
PICKLE=$(ls "$RIBO_DIR"/*_coverage.pkl.gz 2>/dev/null | head -1)

if [ -z "$PICKLE" ]; then
  echo "Error: No coverage pickle files found in $RIBO_DIR"
  exit 1
fi

echo "=============================================="
echo "RIBOSOME STALL SITE CONSENSUS ANALYSIS"
echo "=============================================="
echo "Coverage pickle: $PICKLE"
echo "Ribo file: $RIBO_FILE"
echo "Reference: $REFERENCE_FILE"
echo "Groups: $EXP_GROUPS"
echo "Parameters: min_z=$MIN_Z, min_reads=$MIN_READS, trim_edges=$TRIM_EDGES, pseudocount=$PSEUDOCOUNT"
echo "Consensus: stall_min_reps=$STALL_MIN_REPS, tol=$TOL, min_sep=$MIN_SEP"
echo "Output CSV: $OUT_CSV"
echo "=============================================="

CMD=(python3 stall_sites_consensus.py \
  --pickle "$PICKLE" \
  --ribo "$RIBO_FILE" \
  --groups "$EXP_GROUPS" \
  --tx_threshold "$TX_THRESHOLD" \
  --tx_min_reps "$TX_MIN_REPS" \
  --min_z "$MIN_Z" \
  --min_reads "$MIN_READS" \
  --trim_edges "$TRIM_EDGES" \
  --pseudocount "$PSEUDOCOUNT" \
  --stall_min_reps "$STALL_MIN_REPS" \
  --tol "$TOL" \
  --min_sep "$MIN_SEP" \
  --out-csv "$OUT_CSV" \
  --reference "$REFERENCE_FILE" \
  --out-png "$OUT_PNG" \
  --out-csv "$OUT_CSV")

# Append motif flag if set
if [ -n "$MOTIF_FLAG" ]; then
  CMD+=($MOTIF_FLAG)
fi

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
