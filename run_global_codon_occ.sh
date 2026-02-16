#!/bin/bash
#----------------------------------------------------
# Bash script: run global_codon_occ.py on coverage pickle
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to .ribo file (required)
RIBO_FILE="./ribo_files_29/ribo_29.ribo"

# Path to pickled coverage dict (from adj_coverage.py output, required)
COVERAGE_PICKLE="./ribo_files_29/ribo_29_coverage.pkl.gz"

# Path to reference FASTA/2bit (required)
REFERENCE="/path/to/reference.fasta"

# Offset applied to coverage (P or A, required)
OFFSET="P"

# Output CSV file (optional, default: codon_occupancy.csv)
OUTPUT="codon_occupancy_29.csv"

# Trimming options (optional)
TRIM_START=0
TRIM_STOP=0

# Use human/mouse alias (set to "yes" if your .ribo uses apris_human_alias)
USE_ALIAS="no"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if required files exist
if [ ! -f "$RIBO_FILE" ]; then
  echo "ERROR: .ribo file not found: $RIBO_FILE"
  exit 1
fi

if [ ! -f "$COVERAGE_PICKLE" ]; then
  echo "ERROR: Coverage pickle not found: $COVERAGE_PICKLE"
  exit 1
fi

if [ ! -f "$REFERENCE" ]; then
  echo "ERROR: Reference file not found: $REFERENCE"
  exit 1
fi

# Check if global_codon_occ.py exists
if [ ! -f "global_codon_occ.py" ]; then
  echo "ERROR: Script not found: global_codon_occ.py"
  exit 1
fi

# Build command
CMD="python global_codon_occ.py"
CMD="$CMD --ribo $RIBO_FILE"
CMD="$CMD --pickle $COVERAGE_PICKLE"
CMD="$CMD --reference $REFERENCE"
CMD="$CMD --ofset $OFFSET"
CMD="$CMD --out $OUTPUT"
CMD="$CMD --trim-start $TRIM_START"
CMD="$CMD --trim-stop $TRIM_STOP"

# Add optional alias flag
if [ "$USE_ALIAS" = "yes" ]; then
  CMD="$CMD --use-human-alias"
fi

echo "Running: $CMD"
echo "=========================================="

# Execute
$CMD

if [ $? -eq 0 ]; then
  echo "=========================================="
  echo "SUCCESS: Output written to $OUTPUT"
else
  echo "=========================================="
  echo "ERROR: Script failed"
  exit 1
fi
