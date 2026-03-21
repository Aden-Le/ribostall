#!/bin/bash
#----------------------------------------------------
# Bash script: run stall_sites_non_consensus.py with
# enrichment analysis on coverage files
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

# Stall site calling thresholds (raised defaults)
MIN_Z=2.0
MIN_READS=5
TRIM_START=20
TRIM_STOP=10
PSEUDOCOUNT=0.5

# Reference file (required for enrichment analysis)
REFERENCE_FILE="./C_elegan_reference/appris_celegans_v1_selected_new.fa"

# Output files
OUT_JSON="stall_sites.jsonl"
OUT_ENRICHMENT="./enrichment_results"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find coverage pickle
PICKLE=$(ls "$RIBO_DIR"/*_coverage.pkl.gz 2>/dev/null | head -1)

if [ -z "$PICKLE" ]; then
  echo "Error: No coverage pickle files found in $RIBO_DIR"
  exit 1
fi

echo "=============================================="
echo "RIBOSOME STALL SITE ENRICHMENT ANALYSIS"
echo "=============================================="
echo "Coverage pickle: $PICKLE"
echo "Ribo file: $RIBO_FILE"
echo "Reference: $REFERENCE_FILE"
echo "Groups: $EXP_GROUPS"
echo "Parameters: min_z=$MIN_Z, min_reads=$MIN_READS, trim_start=$TRIM_START, trim_stop=$TRIM_STOP"
echo "Output enrichment: $OUT_ENRICHMENT"
echo "=============================================="

CMD=(python3 stall_sites_non_consensus.py \
  --pickle "$PICKLE" \
  --ribo "$RIBO_FILE" \
  --groups "$EXP_GROUPS" \
  --tx_threshold "$TX_THRESHOLD" \
  --tx_min_reps "$TX_MIN_REPS" \
  --min_z "$MIN_Z" \
  --min_reads "$MIN_READS" \
  --trim-start "$TRIM_START" \
  --trim-stop "$TRIM_STOP" \
  --pseudocount "$PSEUDOCOUNT" \
  --out-json "$OUT_JSON" \
  --reference "$REFERENCE_FILE" \
  --enrichment \
  --out-enrichment "$OUT_ENRICHMENT")

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
