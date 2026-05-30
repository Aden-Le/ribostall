#!/bin/bash
#----------------------------------------------------
# Run the per-family table scripts on each analysis_stats CSV.
#
# Blocks 1-10 run `dylan_table_checker.py` (audit/checker). Blocks 11-16 (the
# within_condition / tfwc family) run `within_condition_sig_split.py`, which
# GENERATES the three-section paired Top-hits tables (both / BWM-only /
# control-only) -- see the tfwc note below. Blocks 17-18 (the
# within_condition_binomial family) run `cross_group_concordance_tables.py`,
# which GENERATES the three cross-group Top-10 tables (concordant enrichment /
# concordant depletion / discordant) pasted into the matching .qmd
# `## Top hits` section -- see the binomial note below.
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
# tfwc blocks (11-16) GENERATE the three-section paired Top-hits tables for
# the within_condition family via `scripts/within_condition_sig_split.py`
# (NOT the checker). Each run prints three markdown tables -- "Significant in
# both", "Significant in BWM only", "Significant in control only" -- one row
# per (site, feature) cell that clears FDR<0.05 in at least one condition,
# pairing the BWM and control `log2_OR` plus an `Effect change` column
# (BWM log2_OR - control log2_OR, the BWM-vs-control divergence) and a
# `low-count` flag. Rows are grouped by site A/P/E, then sorted by
# `Effect change` descending; cells significant in neither condition are
# dropped. Paste the output into BOTH the Olive .qmd Top-hits section and
# Dylan's interpretation .md -- Olive expands the AA one-letter codes to full
# names and lowercases the headers, Dylan keeps the three-letter abbreviation.
# The `low-count` flag names which arm dips below the per-condition count
# threshold (`C` = control): aa blocks use the < 100 BWM / < 200 control
# defaults; codon blocks pass `--rare-bwm-threshold 50 --rare-control-threshold 50`
# to match the codon count convention. Loosen the FDR cutoff per block with
# `--sig-threshold 0.10`.
#
# 17/18 blocks GENERATE the three cross-group Top-10 tables for the
# within_condition_binomial family via
# `scripts/cross_group_concordance_tables.py`. Each run prints three markdown
# tables -- "Concordant enrichment", "Concordant depletion", "Discordant" --
# one row per (site, feature) cell, pooling the six per-group
# `log2_enrichment` values (BWM and control x d0/d5/d10). A cell is concordant
# when all 6 groups share one sign, discordant otherwise. Each table is the
# top 10 by `#sig` (groups with p_adj<0.05, of 6) desc, then `Max Change`
# (max-min log2_enrichment across the 6 groups) desc; the kept rows are
# displayed sorted by site A/P/E, then `#sig` desc, then `Max Change` desc.
# The `log2_enrichment` cell lists the six values on two `\newline`-separated
# lines; the `flag` column aggregates iid-amp / bg-tight / rare-aa|rare-codon
# across the 6 groups. Paste each block into the matching Olive .qmd
# `## Top hits` section (replacing the old cross-group concordance pair). Tune
# with `--top-n`, `--sig-threshold`, `--bg-tight-threshold`,
# `--rare-k-threshold`.
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
python3 scripts/within_condition_sig_split.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_aa.csv"

# ============================================================
# 12. timepoint_fisher_within_condition_d10_vs_d0_codon  (family: tfwc)
# ============================================================
echo "---- 12. timepoint_fisher_within_condition_d10_vs_d0_codon ----"
python3 scripts/within_condition_sig_split.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_codon.csv" --rare-bwm-threshold 50 --rare-control-threshold 50

# ============================================================
# 13. timepoint_fisher_within_condition_d10_vs_d5_aa  (family: tfwc)
# ============================================================
echo "---- 13. timepoint_fisher_within_condition_d10_vs_d5_aa ----"
python3 scripts/within_condition_sig_split.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d5_aa.csv"

# ============================================================
# 14. timepoint_fisher_within_condition_d10_vs_d5_codon  (family: tfwc)
# ============================================================
echo "---- 14. timepoint_fisher_within_condition_d10_vs_d5_codon ----"
python3 scripts/within_condition_sig_split.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d5_codon.csv" --rare-bwm-threshold 50 --rare-control-threshold 50

# ============================================================
# 15. timepoint_fisher_within_condition_d5_vs_d0_aa  (family: tfwc)
# ============================================================
echo "---- 15. timepoint_fisher_within_condition_d5_vs_d0_aa ----"
python3 scripts/within_condition_sig_split.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_aa.csv"

# ============================================================
# 16. timepoint_fisher_within_condition_d5_vs_d0_codon  (family: tfwc)
# ============================================================
echo "---- 16. timepoint_fisher_within_condition_d5_vs_d0_codon ----"
python3 scripts/within_condition_sig_split.py "results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_codon.csv" --rare-bwm-threshold 50 --rare-control-threshold 50

# ============================================================
# 17. within_condition_binomial_aa  (family: binom_aa)
# ============================================================
echo "---- 17. within_condition_binomial_aa ----"
python3 scripts/cross_group_concordance_tables.py "results/stall_sites/enrichment/analysis_stats/within_condition_binomial_aa.csv" --top-n 15

# ============================================================
# 18. within_condition_binomial_codon  (family: binom_codon)
# ============================================================
echo "---- 18. within_condition_binomial_codon ----"
python3 scripts/cross_group_concordance_tables.py "results/stall_sites/enrichment/analysis_stats/within_condition_binomial_codon.csv" --top-n 15
