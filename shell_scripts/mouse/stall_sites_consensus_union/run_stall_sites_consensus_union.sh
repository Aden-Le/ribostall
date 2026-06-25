#!/bin/bash
#----------------------------------------------------
# Bash script: run stall_sites_consensus_union.py with
# consensus stall site calling across replicates (UNION variant:
# each group keeps its own filtered transcript set)
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Path to directory containing _coverage.pkl.gz files
RIBO_DIR="./all_ribo_file"
RIBO_FILE="$RIBO_DIR/mouse_all.ribo"

# Format: "group1:rep1,rep2;group2:rep1,rep2"
# Flat control-vs-treatment design (no timepoints). Replicate names must match
# the experiments in the coverage pickle; group labels are free-form.
# Mouse run: 2-vs-1 design — control has 2 reps (AA_3, AA_4), treatment has 1 (Ch_WAA2).
EXP_GROUPS='control:AA_3,AA_4;treatment:Ch_WAA2'

# Transcript filtering thresholds
# Aligned with the C. elegans non-consensus "Parameter set v2 (2026-04-27)":
# TX_THRESHOLD 1.0 -> 0.5 to retain more transcripts in low-coverage groups.
TX_THRESHOLD=0.5
# Global gate (no per-group override exists). MUST be 1 here: the 1-rep treatment
# group can never reach 2, so a value of 2 would leave it with zero transcripts.
# Trade-off: this also relaxes the 2-rep control filter to ">=1 of 2 reps".
TX_MIN_REPS=1

# Stall site calling thresholds
# Aligned with the C. elegans non-consensus v2 set: MIN_READS 2 -> 5 (off the
# noise floor); TRIM_START already 20. Keeps consensus comparable to non-consensus.
MIN_Z=1.0
MIN_READS=5
TRIM_START=20
TRIM_STOP=10
PSEUDOCOUNT=0.5

# Consensus calling parameters
# Per-group consensus support: must name EVERY declared group (no global fallback).
# control needs both reps; treatment has only 1 rep.
STALL_MIN_REPS_PER_GROUP='control:2;treatment:1'
TOL=0
MIN_SEP=0

# Reference file (required for E/P/A annotation)
REFERENCE_FILE="./reference/appris_mouse_v2_selected.fa.gz"

# E/P/A annotation parameters
BASIS="P"            # register for E/P/A offsets (P or A)
PSITE_OFFSET=0       # codon offset applied to each stall index before deriving E/P/A

# Output directory for stats-ready stall-site CSVs
# (stall_sites_{codon,aa}.csv + per_group_background_{codon,aa}.csv)
OUT_DIR="results/mouse/stall_sites_consensus_union/raw"

# ===============================================

# Activate conda/env if you use one
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ribostall_env

# Navigate to repo root (three levels up from shell_scripts/<organism>/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# Coverage pickle for the exact ribo file above (not a directory glob).
# adj_coverage.py writes <basename>_coverage.pkl.gz next to the .ribo, so derive
# it from RIBO_FILE. With multiple *_coverage.pkl.gz present (e.g. C. elegans +
# mouse), a glob would pick the wrong one.
PICKLE="${RIBO_DIR}/$(basename "$RIBO_FILE" .ribo)_coverage.pkl.gz"

if [ ! -f "$PICKLE" ]; then
  echo "Error: coverage pickle not found: $PICKLE"
  exit 1
fi

echo "=============================================="
echo "RIBOSOME STALL SITE CONSENSUS (UNION) ANALYSIS"
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

CMD=(python3 scripts/stall_sites_consensus_union.py \
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
