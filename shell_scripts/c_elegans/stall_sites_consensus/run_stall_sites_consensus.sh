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
# Flat control-vs-treatment design (no timepoints). Edit the replicate sample
# names to match the experiments in the coverage pickle.
EXP_GROUPS='control:control_rep1,control_rep2;treatment:treatment_rep1,treatment_rep2'

# Transcript filtering thresholds
TX_THRESHOLD=1.0
TX_MIN_REPS=2

# Stall site calling thresholds
MIN_Z=1.0
MIN_READS=2
TRIM_START=20
TRIM_STOP=10
PSEUDOCOUNT=0.5

# Consensus calling parameters
# Per-group consensus support: must name EVERY declared group (no global fallback).
# control needs both reps; treatment has only 1 rep.
STALL_MIN_REPS_PER_GROUP='control:2;treatment:1'
TOL=0
MIN_SEP=7

# Reference file (required for E/P/A annotation)
REFERENCE_FILE="./reference/appris_celegans_v1_selected_new.fa"

# E/P/A annotation parameters
BASIS="P"            # register for E/P/A offsets (P or A)
PSITE_OFFSET=0       # codon offset applied to each stall index before deriving E/P/A

# Output directory for stats-ready stall-site CSVs
# (stall_sites_{codon,aa}.csv + per_group_background_{codon,aa}.csv)
OUT_DIR="results/stall_sites/enrichment"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

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
echo "Parameters: min_z=$MIN_Z, min_reads=$MIN_READS, trim_start=$TRIM_START, trim_stop=$TRIM_STOP, pseudocount=$PSEUDOCOUNT"
echo "Consensus: per-group min_reps=$STALL_MIN_REPS_PER_GROUP, tol=$TOL, min_sep=$MIN_SEP"
echo "E/P/A: basis=$BASIS, psite_offset=$PSITE_OFFSET"
echo "Output dir: $OUT_DIR"
echo "=============================================="

CMD=(python3 scripts/stall_sites_consensus.py \
  --pickle "$PICKLE" \
  --ribo "$RIBO_FILE" \
  --reference "$REFERENCE_FILE" \
  --groups "$EXP_GROUPS" \
  --tx_threshold "$TX_THRESHOLD" \
  --tx_min_reps "$TX_MIN_REPS" \
  --min_z "$MIN_Z" \
  --min_reads "$MIN_READS" \
  --trim-start "$TRIM_START" \
  --trim-stop "$TRIM_STOP" \
  --pseudocount "$PSEUDOCOUNT" \
  --stall_min_reps_per_group "$STALL_MIN_REPS_PER_GROUP" \
  --tol "$TOL" \
  --min_sep "$MIN_SEP" \
  --basis "$BASIS" \
  --psite-offset "$PSITE_OFFSET" \
  --out-dir "$OUT_DIR")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
