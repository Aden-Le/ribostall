#!/bin/bash
#----------------------------------------------------
# Slurm job script: run adj_coverage.py on ribo files
#   for TACC Lonestar6 AMD Milan nodes
#
# Usage:
#   Edit the CONFIG section below, then:
#   sbatch run_adj_coverage.slurm
#
#  -- Runs one Python process per .ribo file (serial per file).
#  -- To run multiple ribo files in parallel, use pylauncher:
#       module load pylauncher
#       and generate a task file; see TACC pylauncher docs.
#----------------------------------------------------

#SBATCH -J adj_coverage       # Job name
#SBATCH -o adj_coverage.o%j   # Name of stdout output file
#SBATCH -e adj_coverage.e%j   # Name of stderr error file
#SBATCH -p normal             # Queue (partition) name
#SBATCH -N 1                  # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                  # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00           # Run time (hh:mm:ss) - adjust if needed
#SBATCH --mail-type=all       # Send email at begin and end of job
#SBATCH -A myproject          # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=adenl56@utexas.edu  # email address to send notifications to

# ============== CONFIG: edit these ==============
# Path to directory containing .ribo files (absolute or relative to $SLURM_SUBMIT_DIR)
RIBO_DIR="${SLURM_SUBMIT_DIR:-.}/ribo_files"

# adj_coverage.py arguments (required)
MIN_LEN=25
MAX_LEN=35
RETURN_SITE="P"    # P-site or A-site

# Optional: site type and search window for offset (uncomment and set if needed)
SITE_TYPE="stop"
# SEARCH_WINDOW="-60 -30"

# Optional: use human/mouse alias (set to "yes" if your .ribo uses apris_human_alias)
USE_ALIAS="no"

# Optional: parallel workers per ribo file (experiments run in parallel within one .ribo)
PROCS=64

# Output: will write to ${RIBO_DIR}/<basename>_coverage.pkl.gz per file
# ===============================================

# Activate conda/env if you use one (uncomment and set path)
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda create -n ribostall_env python=3.8 -y
conda activate ribostall_env

# Install required packages
pip install -r requirements.txt

# Or load a Python module if on TACC
# module load python3

cd "${SLURM_SUBMIT_DIR:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for RIBO in "$RIBO_DIR"/*.ribo; do
  [ -f "$RIBO" ] || continue
  BASENAME=$(basename "$RIBO" .ribo)
  OUT="${RIBO_DIR}/${BASENAME}_coverage.pkl.gz"

  CMD="python3 adj_coverage.py --ribo $RIBO --min-len $MIN_LEN --max-len $MAX_LEN --return-site $RETURN_SITE --out $OUT --procs $PROCS"
  [ "$USE_ALIAS" = "yes" ] && CMD="$CMD --alias"
  # Uncomment if using custom site type / search window:
  # CMD="$CMD --site-type $SITE_TYPE --search-window $SEARCH_WINDOW"

  echo "Running: $CMD"
  eval $CMD
done

echo "Done."
date
