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
#SBATCH -A Ceniklab-CPRIT         # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=adenl56@utexas.edu  # email address to send notifications to

# ============== CONFIG: edit these ==============
# Navigate to script directory to ensure relative paths work correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

sh run_adj_coverage_29.sh
sh run_adj_coverage_34.sh
# ===============================================