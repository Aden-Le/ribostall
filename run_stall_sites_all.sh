#!/bin/bash
#----------------------------------------------------
# Bash script: run stall_sites.py on coverage files
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing _coverage.pkl.gz files
RIBO_DIR="./ribo_all_26_38"

# stall_sites.py arguments (required)
# Format: "group1:rep1,rep2,rep3;group2:rep1,rep2,rep3"
# IMPORTANT: replace rep names with actual replicate names from your .ribo file
GROUPS="control_day_0:control_day0_rep1,control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep1,control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep1,control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep1,BWM_day0_rep2,BWM_day0_rep3;BWM_dayG_5:BWM_day5_rep1,BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep1,BWM_day10_rep2,BWM_day10_rep3"

# Optional: transcript filtering thresholds
TX_THRESHOLD=1.0
TX_MIN_REPS=2

# Optional: stall site calling thresholds
MIN_Z=1.0
MIN_READS=2
STALL_MIN_REPS=2
TRIM_EDGES=10
MIN_SEP=7
PSEUDOCOUNT=0.5

# Optional: motif analysis (set to "yes" to enable)
RUN_MOTIF="no"
# If using motif, set reference file path:
REFERENCE_FILE=""  # e.g., "./reference_files/appris_mouse_v2_selected.fa.gz"
FLANK_LEFT=10
FLANK_RIGHT=6

# Output files
OUT_JSON="stall_sites.jsonl"
OUT_PNG="motif.png"
OUT_CSV="motif_csv"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find all _coverage.pkl.gz files - we'll use one representative pickle per RIBO_DIR
PICKLE=$(ls "$RIBO_DIR"/*_coverage.pkl.gz 2>/dev/null | head -1)

if [ -z "$PICKLE" ]; then
  echo "Error: No coverage pickle files found in $RIBO_DIR"
  exit 1
fi

echo "Using coverage pickle: $PICKLE"

# Build command array
CMD=(python3 stall_sites.py --pickle "$PICKLE" --ribo "$RIBO_DIR" --groups "$GROUPS" --tx_threshold "$TX_THRESHOLD" --tx_min_reps "$TX_MIN_REPS" --min_z "$MIN_Z" --min_reads "$MIN_READS" --stall_min_reps "$STALL_MIN_REPS" --trim_edges "$TRIM_EDGES" --min_sep "$MIN_SEP" --pseudocount "$PSEUDOCOUNT" --out-json "$OUT_JSON" --out-csv "$OUT_CSV")

# Add motif analysis if enabled
if [ "$RUN_MOTIF" = "yes" ]; then
  if [ -z "$REFERENCE_FILE" ]; then
    echo "Error: RUN_MOTIF is yes but REFERENCE_FILE is not set"
    exit 1
  fi
  CMD+=(--motif --reference "$REFERENCE_FILE" --flank-left "$FLANK_LEFT" --flank-right "$FLANK_RIGHT" --out-png "$OUT_PNG")
fi

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo "Done."
date
