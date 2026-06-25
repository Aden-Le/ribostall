#!/bin/bash
#----------------------------------------------------
# Bash script: per-replicate stall calling for MOUSE
# (stall_sites_non_consensus.py)
#
# Produces stall_sites_{codon,aa}.csv and
# per_group_background_{codon,aa}.csv in $RAW_DIR.
# Run run_stall_sites_non_consensus_stats.sh afterwards for the
# enrichment tests — it is ribopy-free and consumes
# only those CSVs.
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
# Per-group transcript-filter support (replaces the old global TX_MIN_REPS): must
# name EVERY declared group, no global fallback. Same 'group:int;...' format as
# the consensus scripts' STALL_MIN_REPS_PER_GROUP. control requires both reps;
# treatment has only its single rep.
TX_MIN_REPS_PER_GROUP='control:2;treatment:1'

# Stall site calling thresholds
# Aligned with the C. elegans non-consensus v2 set / mouse consensus runners:
#   MIN_READS 2 -> 5 (off the noise floor); TRIM_START already 20.
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

# Output filtering: drop stall windows whose E/P/A site hits a stop codon
# (TAA/TAG/TGA) from the output CSVs ("True"/"False"). Default True; pass
# "False" to keep them.
DROP_STOP_CODONS="True"

# Reference file (required for enrichment analysis)
REFERENCE_FILE="./reference/appris_mouse_v2_selected.fa.gz"

# Output files
RAW_DIR="./results/mouse/stall_sites_non_consensus/raw"

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
echo "RIBOSOME STALL SITE CALLING"
echo "=============================================="
echo "Coverage pickle: $PICKLE"
echo "Ribo file: $RIBO_FILE"
echo "Reference: $REFERENCE_FILE"
echo "Groups: $EXP_GROUPS"
echo "Transcript filter: per-group min_reps=$TX_MIN_REPS_PER_GROUP, tx_threshold=$TX_THRESHOLD"
echo "Parameters: min_z=$MIN_Z, min_reads=$MIN_READS, trim_start=$TRIM_START, trim_stop=$TRIM_STOP"
echo "E/P/A geometry: basis=$BASIS, psite_offset=$PSITE_OFFSET"
echo "Output filter: drop_stop_codons=$DROP_STOP_CODONS"
echo "Output raw CSVs: $RAW_DIR"
echo "=============================================="

python3 scripts/stall_sites_non_consensus.py \
  --pickle "$PICKLE" \
  --ribo "$RIBO_FILE" \
  --reference "$REFERENCE_FILE" \
  --groups "$EXP_GROUPS" \
  --tx_threshold "$TX_THRESHOLD" \
  --tx_min_reps_per_group "$TX_MIN_REPS_PER_GROUP" \
  --min_z "$MIN_Z" \
  --min_reads "$MIN_READS" \
  --trim-start "$TRIM_START" \
  --trim-stop "$TRIM_STOP" \
  --pseudocount "$PSEUDOCOUNT" \
  --basis "$BASIS" \
  --psite-offset "$PSITE_OFFSET" \
  --drop-stop-codons "$DROP_STOP_CODONS" \
  --out-dir "$RAW_DIR"

echo ""
echo "=============================================="
echo "Done."
date
echo "=============================================="
