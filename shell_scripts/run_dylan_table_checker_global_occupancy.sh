#!/bin/bash
#----------------------------------------------------
# GLOBAL OCCUPANCY variant of run_dylan_table_checker.sh.
#
# Runs the per-family table scripts on each global occupancy
# `analysis_corrected` CSV. The global occupancy CSVs are the same 18
# analyses as the stall_sites `analysis_stats` set (identical column schemas,
# same test code) -- only the underlying data differs (whole-transcriptome
# codon occupancy vs called stall sites) and the filename convention is
# reversed (level token FIRST: `aa_<analysis>` / `codon_<analysis>`, rather
# than stall_sites' `<analysis>_aa` / `<analysis>_codon`). The checker scripts
# key off COLUMNS, not filenames -- the aa/codon split is passed via
# `--family fisher_aa`/`fisher_codon` -- so they run on these CSVs unchanged.
#
# Blocks 1-10 run `dylan_table_checker.py` (audit/checker). Blocks 11-16 (the
# within_condition / tfwc family) run `within_condition_sig_split.py`, which
# GENERATES the three-section paired Top-hits tables (both / BWM-only /
# control-only). Blocks 17-18 (the within_condition_binomial family) run
# `cross_group_concordance_tables.py`, which GENERATES the three cross-group
# concordance tables (concordant enrichment / concordant depletion /
# discordant) pasted into the matching .qmd `## Top hits` section.
#
# WORKFLOW
#   - Full run    : `bash shell_scripts/run_dylan_table_checker_global_occupancy.sh`
#   - Single one  : copy the two-line block under any numbered banner below
#                   and paste it into your shell (bash OR PowerShell).
#                   Each block uses a literal relative path, so as long as
#                   your CWD is the repo root, it just works.
#
# Output prints to the terminal; nothing is written to disk. Compare each
# block's printed picks against the corresponding Olive report's Top-hits
# sub-tables to verify Dylan's selection rule was applied correctly.
#
# Numbering matches the global occupancy `_MANUAL_REVIEW.md` / `_OLIVE_PLAN.md`
# (results/global_occupancy/olive_reports/). The family<->plot-dir mapping
# (the bridge across the reversed filename convention) lives in that
# `_OLIVE_PLAN.md` Context block.
#
# Wilcoxon blocks (1-8) also print a `low-count audit` section before the
# Top hits. A row is flagged `low-count` when min(median_arm_A,
# median_arm_B) < 0.005. Override per block with `--low-count-threshold`.
#
# Blocks 9b and 10b run `cross_tp_summary_checker.py` on the per_timepoint
# fisher CSVs (#9 + #10) to reproduce the two cross-timepoint summary tables
# (direction concordance + direction-flip cells across timepoints).
#
# tfwc blocks (11-16) GENERATE the three-section paired Top-hits tables via
# `scripts/within_condition_sig_split.py` (NOT the checker). codon blocks pass
# `--rare-bwm-threshold 50 --rare-control-threshold 50` to match the codon
# count convention. Loosen the FDR cutoff per block with `--sig-threshold 0.10`.
#
# 17/18 blocks GENERATE the three cross-group concordance tables via
# `scripts/cross_group_concordance_tables.py`. Each table shows every cell with
# `#sig` >= `--min-sig` (groups with p_adj<0.05, of 6). Tune with `--min-sig`,
# `--sig-threshold`, `--bg-tight-threshold`, `--rare-k-threshold`.
#----------------------------------------------------

# When running the whole script via bash, cd to repo root so the relative
# paths below resolve. (Bash-only -- irrelevant for copy-paste.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ============================================================
# 1. aa_wilcoxon_condition  (family: between_condition_wilcoxon)
# ============================================================
echo "---- 1. aa_wilcoxon_condition ----"
python3 scripts/dylan_table_checker.py "results/global_occupancy/analysis_corrected/aa_wilcoxon_condition.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 2. codon_wilcoxon_condition  (family: between_condition_wilcoxon)
# ============================================================
echo "---- 2. codon_wilcoxon_condition ----"
python3 scripts/dylan_table_checker.py "results/global_occupancy/analysis_corrected/codon_wilcoxon_condition.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 3. aa_wilcoxon_timepoint_d10_vs_d0  (family: between_timepoint_wilcoxon)
# ============================================================
echo "---- 3. aa_wilcoxon_timepoint_d10_vs_d0 ----"
python3 scripts/dylan_table_checker.py "results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d10_vs_d0.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 4. codon_wilcoxon_timepoint_d10_vs_d0  (family: between_timepoint_wilcoxon)
# ============================================================
echo "---- 4. codon_wilcoxon_timepoint_d10_vs_d0 ----"
python3 scripts/dylan_table_checker.py "results/global_occupancy/analysis_corrected/codon_wilcoxon_timepoint_d10_vs_d0.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 5. aa_wilcoxon_timepoint_d10_vs_d5  (family: between_timepoint_wilcoxon)
# ============================================================
echo "---- 5. aa_wilcoxon_timepoint_d10_vs_d5 ----"
python3 scripts/dylan_table_checker.py "results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d10_vs_d5.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 6. codon_wilcoxon_timepoint_d10_vs_d5  (family: between_timepoint_wilcoxon)
# ============================================================
echo "---- 6. codon_wilcoxon_timepoint_d10_vs_d5 ----"
python3 scripts/dylan_table_checker.py "results/global_occupancy/analysis_corrected/codon_wilcoxon_timepoint_d10_vs_d5.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 7. aa_wilcoxon_timepoint_d5_vs_d0  (family: between_timepoint_wilcoxon)
# ============================================================
echo "---- 7. aa_wilcoxon_timepoint_d5_vs_d0 ----"
python3 scripts/dylan_table_checker.py "results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d5_vs_d0.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 8. codon_wilcoxon_timepoint_d5_vs_d0  (family: between_timepoint_wilcoxon)
# ============================================================
echo "---- 8. codon_wilcoxon_timepoint_d5_vs_d0 ----"
python3 scripts/dylan_table_checker.py "results/global_occupancy/analysis_corrected/codon_wilcoxon_timepoint_d5_vs_d0.csv" --family wilcoxon --low-count-threshold 0.005

# ============================================================
# 9. aa_per_timepoint_fisher  (family: per_timepoint_fisher)
# ============================================================
echo "---- 9. aa_per_timepoint_fisher ----"
python3 scripts/dylan_table_checker.py "results/global_occupancy/analysis_corrected/aa_per_timepoint_fisher.csv" --family fisher_aa

# ============================================================
# 9b. aa_per_timepoint_fisher -- cross-timepoint summary tables
# ============================================================
echo "---- 9b. aa_per_timepoint_fisher cross-timepoint summary ----"
python3 scripts/cross_tp_summary_checker.py "results/global_occupancy/analysis_corrected/aa_per_timepoint_fisher.csv"

# ============================================================
# 10. codon_per_timepoint_fisher  (family: per_timepoint_fisher)
# ============================================================
echo "---- 10. codon_per_timepoint_fisher ----"
python3 scripts/dylan_table_checker.py "results/global_occupancy/analysis_corrected/codon_per_timepoint_fisher.csv" --family fisher_codon

# ============================================================
# 10b. codon_per_timepoint_fisher -- cross-timepoint summary tables
# ============================================================
echo "---- 10b. codon_per_timepoint_fisher cross-timepoint summary ----"
python3 scripts/cross_tp_summary_checker.py "results/global_occupancy/analysis_corrected/codon_per_timepoint_fisher.csv"

# ============================================================
# 11. aa_timepoint_fisher_within_condition_d10_vs_d0  (family: tfwc)
# ============================================================
echo "---- 11. aa_timepoint_fisher_within_condition_d10_vs_d0 ----"
python3 scripts/within_condition_sig_split.py "results/global_occupancy/analysis_corrected/aa_timepoint_fisher_within_condition_d10_vs_d0.csv"

# ============================================================
# 12. codon_timepoint_fisher_within_condition_d10_vs_d0  (family: tfwc)
# ============================================================
echo "---- 12. codon_timepoint_fisher_within_condition_d10_vs_d0 ----"
python3 scripts/within_condition_sig_split.py "results/global_occupancy/analysis_corrected/codon_timepoint_fisher_within_condition_d10_vs_d0.csv" --rare-bwm-threshold 50 --rare-control-threshold 50

# ============================================================
# 13. aa_timepoint_fisher_within_condition_d10_vs_d5  (family: tfwc)
# ============================================================
echo "---- 13. aa_timepoint_fisher_within_condition_d10_vs_d5 ----"
python3 scripts/within_condition_sig_split.py "results/global_occupancy/analysis_corrected/aa_timepoint_fisher_within_condition_d10_vs_d5.csv"

# ============================================================
# 14. codon_timepoint_fisher_within_condition_d10_vs_d5  (family: tfwc)
# ============================================================
echo "---- 14. codon_timepoint_fisher_within_condition_d10_vs_d5 ----"
python3 scripts/within_condition_sig_split.py "results/global_occupancy/analysis_corrected/codon_timepoint_fisher_within_condition_d10_vs_d5.csv" --rare-bwm-threshold 50 --rare-control-threshold 50

# ============================================================
# 15. aa_timepoint_fisher_within_condition_d5_vs_d0  (family: tfwc)
# ============================================================
echo "---- 15. aa_timepoint_fisher_within_condition_d5_vs_d0 ----"
python3 scripts/within_condition_sig_split.py "results/global_occupancy/analysis_corrected/aa_timepoint_fisher_within_condition_d5_vs_d0.csv"

# ============================================================
# 16. codon_timepoint_fisher_within_condition_d5_vs_d0  (family: tfwc)
# ============================================================
echo "---- 16. codon_timepoint_fisher_within_condition_d5_vs_d0 ----"
python3 scripts/within_condition_sig_split.py "results/global_occupancy/analysis_corrected/codon_timepoint_fisher_within_condition_d5_vs_d0.csv" --rare-bwm-threshold 50 --rare-control-threshold 50

# ============================================================
# 17. aa_within_condition_binomial  (family: within_condition_binomial)
# ============================================================
echo "---- 17. aa_within_condition_binomial ----"
python3 scripts/cross_group_concordance_tables.py "results/global_occupancy/analysis_corrected/aa_within_condition_binomial.csv" --min-sig 2

# ============================================================
# 18. codon_within_condition_binomial  (family: within_condition_binomial)
# ============================================================
echo "---- 18. codon_within_condition_binomial ----"
python3 scripts/cross_group_concordance_tables.py "results/global_occupancy/analysis_corrected/codon_within_condition_binomial.csv" --min-sig 2
