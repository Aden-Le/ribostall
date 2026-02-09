#!/bin/bash
#----------------------------------------------------
# Bash script: run adj_coverage.py on ribo files
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing .ribo files (absolute or relative to current directory)
RIBO_DIR="./ribo_files_29"

# adj_coverage.py arguments (required)
MIN_LEN=26
MAX_LEN=32
RETURN_SITE="P"    # P-site or A-site

# Optional: site type and search window for offset (uncomment and set if needed)
SITE_TYPE="start"
# SEARCH_WINDOW="-60 -30"

# Optional: use human/mouse alias (set to "yes" if your .ribo uses apris_human_alias)
USE_ALIAS="no"

# Optional: parallel workers per ribo file (experiments run in parallel within one .ribo)
PROCS=64

# Output: will write to ${RIBO_DIR}/<basename>_coverage.pkl.gz per file
# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if RIBO_DIR exists
if [ ! -d "$RIBO_DIR" ]; then
  echo "ERROR: Directory not found: $RIBO_DIR"
  exit 1
fi

# Check if adj_coverage.py exists
if [ ! -f "adj_coverage.py" ]; then
  echo "ERROR: Script not found: adj_coverage.py"
  exit 1
fi

# Count .ribo files
RIBO_COUNT=$(find "$RIBO_DIR" -maxdepth 1 -name "*.ribo" -type f | wc -l)
if [ $RIBO_COUNT -eq 0 ]; then
  echo "ERROR: No .ribo files found in $RIBO_DIR"
  exit 1
fi

echo "Found $RIBO_COUNT .ribo file(s) in $RIBO_DIR"

# Process all .ribo files
PROCESSED=0
FAILED=0

for RIBO in "$RIBO_DIR"/*.ribo; do
  [ -f "$RIBO" ] || continue
  BASENAME=$(basename "$RIBO" .ribo)
  OUT="${RIBO_DIR}/${BASENAME}_coverage.pkl.gz"

  CMD="python3 adj_coverage.py --ribo $RIBO --min-len $MIN_LEN --max-len $MAX_LEN --return-site $RETURN_SITE --out $OUT --procs $PROCS"
  [ "$USE_ALIAS" = "yes" ] && CMD="$CMD --alias"
  CMD="$CMD --site-type $SITE_TYPE"
  # Uncomment if using custom search window:
  # CMD="$CMD --search-window $SEARCH_WINDOW"

  echo "Running: $CMD"
  if eval $CMD; then
    echo "✓ Successfully processed: $BASENAME"
    ((PROCESSED++))
  else
    echo "✗ FAILED to process: $BASENAME (exit code: $?)"
    ((FAILED++))
  fi
done

echo ""
echo "Summary: Processed $PROCESSED/$RIBO_COUNT files successfully"
if [ $FAILED -gt 0 ]; then
  echo "WARNING: $FAILED file(s) failed"
  exit 1
fi

echo "Done."
date

