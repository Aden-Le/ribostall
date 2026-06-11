#!/bin/bash
#----------------------------------------------------
# Bash script: per-replicate stall calling
# (stall_sites_non_consensus_call.py)
#
# Produces stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv in $OUT_ENRICHMENT.
# Run run_stall_sites_non_consensus_stats.sh afterwards for the
# enrichment tests — it is ribopy-free and consumes
# only those CSVs.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing _coverage.pkl.gz files
RIBO_DIR="./all_ribo_file"
RIBO_FILE="$RIBO_DIR/C_elegan_all_02_04_2026.ribo"

# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Transcript filtering thresholds
# Parameter set v2 (2026-04-27): TX_THRESHOLD lowered 1.0 -> 0.5 to retain more
# transcripts in low-coverage groups (BWM_day_0, BWM_day_10 had only 51 / 36 tx
# under v1). See docs/analysis_interpretation/parameter_decisions.md.
TX_THRESHOLD=0.5
TX_MIN_REPS=2

# Stall site calling thresholds
# Parameter set v2 (2026-04-27):
#   MIN_READS raised 2 -> 5 (move off the noise floor)
#   TRIM_START raised 10 -> 20 (match global script + better excludes initiation ramp)
# See docs/analysis_interpretation/parameter_decisions.md.
MIN_Z=1.0
MIN_READS=5
TRIM_START=20
TRIM_STOP=10
PSEUDOCOUNT=0.5

# E/P/A site geometry
# BASIS: register for E/P/A offsets
#   "P" -> E=-1 codon, P=0,    A=+1 codon (P-site coverage)
#   "A" -> E=-2,       P=-1,   A=0       (A-site coverage)
# PSITE_OFFSET: codon shift applied to each stall index before deriving E/P/A
BASIS="P"
PSITE_OFFSET=0

# Reference file (required for enrichment analysis)
REFERENCE_FILE="./reference/appris_celegans_v1_selected_new.fa"

# Output files
OUT_ENRICHMENT="./results/stall_sites/enrichment"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Find coverage pickle
PICKLE=$(ls "$RIBO_DIR"/*_coverage.pkl.gz 2>/dev/null | head -1)

if [ -z "$PICKLE" ]; then
  echo "Error: No coverage pickle files found in $RIBO_DIR"
  exit 1
fi

echo "=============================================="
echo "RIBOSOME STALL SITE CALLING"
echo "=============================================="
echo "Coverage pickle: $PICKLE"
echo "Ribo file: $RIBO_FILE"
echo "Reference: $REFERENCE_FILE"
echo "Groups: $EXP_GROUPS"
echo "Parameters: min_z=$MIN_Z, min_reads=$MIN_READS, trim_start=$TRIM_START, trim_stop=$TRIM_STOP"
echo "E/P/A geometry: basis=$BASIS, psite_offset=$PSITE_OFFSET"
echo "Output enrichment: $OUT_ENRICHMENT"
echo "=============================================="

python3 scripts/stall_sites_non_consensus_call.py \
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
  --basis "$BASIS" \
  --psite-offset "$PSITE_OFFSET" \
  --out-dir "$OUT_ENRICHMENT"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
