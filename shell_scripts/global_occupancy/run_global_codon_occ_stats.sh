#!/bin/bash
#----------------------------------------------------
# Bash script: run global_codon_occ_stats.py
# Runs statistical tests on the base CSVs produced
# by run_global_codon_occ.sh (reads from out_dir/raw/).
#
# The Python script now processes all 3 sites (E/P/A)
# for one level per invocation, and writes BOTH:
#   - per-site CSVs to out_dir/analysis/{E,P,A}/
#   - merged CSVs   to out_dir/analysis_corrected/
#     (per-site frames concatenated with a 'site' column —
#      the old merge_global_occupancy_analysis.py step is
#      now folded in). This shell script loops the 2 levels.
#----------------------------------------------------

# ============== CONFIG: edit these ==============
# Format: "group1:rep1,rep2;group2:rep1,rep2"
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'

# Must match --out-dir used in run_global_codon_occ.sh
OUT_DIR="./results/global_occupancy"

# Ribosome sites and occupancy levels to process
SITES=(E P A)
LEVELS=(codon aa)

# Headline condition for the between-condition tests (Wilcoxon Analysis 2,
# per-timepoint Fisher Analysis 4) lives in the shared _headline_config.sh, which
# the plot launchers also source — so the stats direction and the plot labels come
# from ONE place and cannot drift. A positive effect (log2_FC / log2 odds ratio)
# means enriched in HEADLINE_CONDITION. Leave it empty there for alphabetical.
source "$(dirname "${BASH_SOURCE[0]}")/_headline_config.sh"

# ===============================================

# Navigate to repo root (two levels up from shell_scripts/<subdir>/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Pass --headline-condition only when set, so an empty value falls back to the
# script's alphabetical default.
HEADLINE_FLAG=()
[ -n "$HEADLINE_CONDITION" ] && HEADLINE_FLAG=(--headline-condition "$HEADLINE_CONDITION")

echo "=============================================="
echo "GLOBAL CODON & AMINO ACID OCCUPANCY — STEP 2"
echo "Statistical tests on base CSVs"
echo "=============================================="
echo "Raw directory:        $OUT_DIR/raw/"
echo "Per-site output:      $OUT_DIR/analysis/{${SITES[*]// /,}}/"
echo "Merged output:        $OUT_DIR/analysis_corrected/"
echo "Sites:                ${SITES[*]}"
echo "Levels:               ${LEVELS[*]}"
echo "Groups:               $EXP_GROUPS"
echo "=============================================="

for level in "${LEVELS[@]}"; do
  echo ""
  echo "----------------------------------------------"
  echo "Level: $level  |  Sites: ${SITES[*]}"
  echo "----------------------------------------------"

  CMD=(python3 scripts/global_codon_occ_stats.py \
    --raw-dir "$OUT_DIR/raw" \
    --analysis-dir "$OUT_DIR/analysis" \
    --corrected-dir "$OUT_DIR/analysis_corrected" \
    --level "$level" \
    --sites "${SITES[@]}" \
    --groups "$EXP_GROUPS" \
    "${HEADLINE_FLAG[@]}")

  echo "Running: ${CMD[@]}"
  "${CMD[@]}"
done

echo ""
echo "=============================================="
echo "Done. Per-site CSVs in $OUT_DIR/analysis/{${SITES[*]// /,}}/"
echo "      Merged CSVs in   $OUT_DIR/analysis_corrected/"
date
echo "=============================================="
