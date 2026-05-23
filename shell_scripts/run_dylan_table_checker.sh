#!/bin/bash
#----------------------------------------------------
# Run dylan_table_checker.py on each analysis_stats CSV.
#
# WORKFLOW
#   - Full run    : `bash shell_scripts/run_dylan_table_checker.sh`
#   - Single one  : copy the two-line block under any numbered banner below
#                   and paste it into your shell (bash OR PowerShell).
#                   Each block uses a literal relative path, so as long as
#                   your CWD is the repo root, it just works.
#
# Output prints to the terminal; nothing is written to disk. Compare each
# block's printed picks against the corresponding Olive report's Top-hits
# sub-tables to verify Dylan's selection rule was applied correctly.
#
# Numbering matches `_MANUAL_REVIEW.md` / `_OLIVE_PLAN.md`.
#
# Wilcoxon blocks (1-8) also print a `low-count audit` section before the
# Top hits. A row is flagged `low-count` when min(median_arm_A,
# median_arm_B) < 0.005. The per-site `Top hits` sub-tables include the
# `min_median` + `low_count` columns so you can directly compare the
# script's classification against the `low-count` flag column in the
# matching Olive .qmd. Override the threshold per block with
# `--low-count-threshold 0.002` if you want to tighten it.
#
# Blocks 9b and 10b run `cross_tp_summary_checker.py` on the per_timepoint
# fisher CSVs (#9 + #10) to reproduce the two summary tables that appear
# under "Cross-timepoint summary (two ranking tables)" in the matching
# Olive report: (1) Cross-timepoint direction concordance and (2)
# Direction-flip cells across timepoints. The checker is feature-agnostic
# (works for any CSV with `site, timepoint, <amino_acid|codon>,
# odds_ratio, p_adj, BWM_count, control_count`) so any future per-timepoint
# fisher CSV can be added by copying one of the two-line blocks below and
# swapping the path.
#
# The `rare-aa` / `rare-codon` flag printed by 9b/10b in those summary
# tables is suffixed with the chronological list of timepoints that trip
# the rare-k threshold (e.g. `rare-aa (d0, d10)`), so the resolved label
# can be pasted straight back into the matching Concordance / Direction-
# flip rows in the .qmd without a second lookup. Per-timepoint Top-hits
# sub-tables (8/9/10/...) keep the bare `rare-aa` / `rare-codon` flag --
# the section header already names the TP.
#
# tfwc blocks (11-16) also print a `rare-low-count audit` section before
# the Top hits. A row is flagged `rare_low_count` when at least one of its
# two `day_X_count` columns falls below a per-condition threshold (default
# < 100 in BWM and < 200 in control at AA resolution; < 50 in both at codon
# resolution -- matches the qmd `rare-aa-low-count` / `rare-codon-low-count`
# flag glossary). The per-(condition, site) Top hits sub-tables include the
# two `day_X_count` columns plus `min_day_count` and `rare_low_count` so
# the script's classification can be compared directly against the matching
# Olive .qmd. Override per block with `--rare-low-bwm-threshold 100
# --rare-low-control-threshold 200` if you want to retune.
#
# tfwc Top-hits selection: hard cutoff at `p_adj < 0.05` -- every row that
# clears the threshold is printed (no top-N cap, no fallback). Within each
# (condition, site, direction) cell, rows are ranked by p_adj ascending
# with |log2_OR| descending as tiebreaker. Override the cutoff per block
# with `--tfwc-p-adj-threshold 0.10`. Note that this rule diverges from
# the older qmd intro paragraph ("Up to 5 rows ... p_adj < 0.10 ... with
# fallback") -- the qmd will need a parallel update before the script's
# picks can be eyeballed against the report 1:1.
#----------------------------------------------------

# When running the whole script via bash, cd to repo root so the relative
# paths below resolve. (Bash-only — irrelevant for copy-paste.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ============================================================
# 1. between_condition_wilcoxon_aa  (family: wilcoxon)
# ============================================================
echo "---- 1. between_condition_wilcoxon_aa ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/between_condition_wilcoxon_aa.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 2. between_condition_wilcoxon_codon  (family: wilcoxon)
# ============================================================
echo "---- 2. between_condition_wilcoxon_codon ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/between_condition_wilcoxon_codon.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 3. between_timepoint_wilcoxon_d10_vs_d0_aa  (family: wilcoxon)
# ============================================================
echo "---- 3. between_timepoint_wilcoxon_d10_vs_d0_aa ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d0_aa.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 4. between_timepoint_wilcoxon_d10_vs_d0_codon  (family: wilcoxon)
# ============================================================
echo "---- 4. between_timepoint_wilcoxon_d10_vs_d0_codon ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d0_codon.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 5. between_timepoint_wilcoxon_d10_vs_d5_aa  (family: wilcoxon)
# ============================================================
echo "---- 5. between_timepoint_wilcoxon_d10_vs_d5_aa ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d5_aa.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 6. between_timepoint_wilcoxon_d10_vs_d5_codon  (family: wilcoxon)
# ============================================================
echo "---- 6. between_timepoint_wilcoxon_d10_vs_d5_codon ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d5_codon.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 7. between_timepoint_wilcoxon_d5_vs_d0_aa  (family: wilcoxon)
# ============================================================
echo "---- 7. between_timepoint_wilcoxon_d5_vs_d0_aa ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d5_vs_d0_aa.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 8. between_timepoint_wilcoxon_d5_vs_d0_codon  (family: wilcoxon)
# ============================================================
echo "---- 8. between_timepoint_wilcoxon_d5_vs_d0_codon ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d5_vs_d0_codon.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 9. per_timepoint_fisher_aa  (family: fisher_aa)
# ============================================================
echo "---- 9. per_timepoint_fisher_aa ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_aa.csv" --family fisher_aa

# ============================================================
# 9b. per_timepoint_fisher_aa -- cross-timepoint summary tables
# ============================================================
echo "---- 9b. per_timepoint_fisher_aa cross-timepoint summary ----"
python3 scripts/cross_tp_summary_checker.py "results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_aa.csv"

# ============================================================
# 10. per_timepoint_fisher_codon  (family: fisher_codon)
# ============================================================
echo "---- 10. per_timepoint_fisher_codon ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_codon.csv" --family fisher_codon

# ============================================================
# 10b. per_timepoint_fisher_codon -- cross-timepoint summary tables
# ============================================================
echo "---- 10b. per_timepoint_fisher_codon cross-timepoint summary ----"
python3 scripts/cross_tp_summary_checker.py "results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_codon.csv"

# ============================================================
# 11. timepoint_fisher_within_condition_d10_vs_d0_aa  (family: tfwc)
# ============================================================
echo "---- 11. timepoint_fisher_within_condition_d10_vs_d0_aa ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_aa.csv" --family tfwc

# ============================================================
# 12. timepoint_fisher_within_condition_d10_vs_d0_codon  (family: tfwc)
# ============================================================
echo "---- 12. timepoint_fisher_within_condition_d10_vs_d0_codon ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_codon.csv" --family tfwc

# ============================================================
# 13. timepoint_fisher_within_condition_d10_vs_d5_aa  (family: tfwc)
# ============================================================
echo "---- 13. timepoint_fisher_within_condition_d10_vs_d5_aa ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d5_aa.csv" --family tfwc

# ============================================================
# 14. timepoint_fisher_within_condition_d10_vs_d5_codon  (family: tfwc)
# ============================================================
echo "---- 14. timepoint_fisher_within_condition_d10_vs_d5_codon ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d5_codon.csv" --family tfwc

# ============================================================
# 15. timepoint_fisher_within_condition_d5_vs_d0_aa  (family: tfwc)
# ============================================================
echo "---- 15. timepoint_fisher_within_condition_d5_vs_d0_aa ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_aa.csv" --family tfwc

# ============================================================
# 16. timepoint_fisher_within_condition_d5_vs_d0_codon  (family: tfwc)
# ============================================================
echo "---- 16. timepoint_fisher_within_condition_d5_vs_d0_codon ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_codon.csv" --family tfwc

# ============================================================
# 17. within_condition_binomial_aa  (family: binom_aa  -- rule TBD)
# ============================================================
echo "---- 17. within_condition_binomial_aa ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/within_condition_binomial_aa.csv" --family binom_aa

# ============================================================
# 18. within_condition_binomial_codon  (family: binom_codon)
# ============================================================
echo "---- 18. within_condition_binomial_codon ----"
python3 scripts/dylan_table_checker.py "results/stall_sites/enrichment/analysis_stats/within_condition_binomial_codon.csv" --family binom_codon
