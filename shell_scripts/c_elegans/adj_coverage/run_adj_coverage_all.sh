#!/bin/bash
#----------------------------------------------------
# Bash script: run adj_coverage.py on ribo files
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing .ribo files (absolute or relative to current directory)
RIBO_DIR="./all_ribo_file"

# Exact input .ribo for this run (a single file, not a directory glob).
# all_ribo_file/ holds multiple organisms' .ribo files, so target the C. elegans
# one explicitly -- a *.ribo glob would also pick up mouse_all.ribo.
RIBO_FILE="$RIBO_DIR/C_elegan_all_02_04_2026.ribo"

# adj_coverage.py arguments (required)
MIN_LEN=26
MAX_LEN=38
RETURN_SITE="P"    # P-site or A-site

# Site type and search window for offset (explicit; do not rely on defaults).
# For landmark "start" the offset peak window is -25..-10 (5' end).
SITE_TYPE="start"
SEARCH_WINDOW="-25 -10"

# Optional: use human/mouse alias (set to "yes" if your .ribo uses apris_human_alias)
USE_ALIAS="no"

# Optional: parallel workers per ribo file (experiments run in parallel within one .ribo)
PROCS=64

# Output: will write to ${RIBO_DIR}/<basename>_coverage.pkl.gz per file
# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root (three levels up from shell_scripts/<organism>/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# Process the single specified .ribo file
RIBO="$RIBO_FILE"
if [ ! -f "$RIBO" ]; then
  echo "Error: ribo file not found: $RIBO"
  exit 1
fi
BASENAME=$(basename "$RIBO" .ribo)
OUT="${RIBO_DIR}/${BASENAME}_coverage.pkl.gz"

CMD="python3 scripts/adj_coverage.py --ribo $RIBO --min-len $MIN_LEN --max-len $MAX_LEN --return-site $RETURN_SITE --out $OUT --procs $PROCS"
[ "$USE_ALIAS" = "yes" ] && CMD="$CMD --alias"
CMD="$CMD --site-type $SITE_TYPE"
CMD="$CMD --search-window $SEARCH_WINDOW"

echo "Running: $CMD"
eval $CMD

echo "Done."
date