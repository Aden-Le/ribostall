#!/bin/bash
#----------------------------------------------------
# Re-render Olive Quarto reports.
#
# WORKFLOW
#   - Full run    : `bash shell_scripts/misc/render_olive_reports.sh`  (from repo root)
#   - Single one  : copy the two-line block under any numbered banner below
#                   and paste it into your shell (bash OR PowerShell).
#                   Each block uses a literal relative path, so as long as
#                   your CWD is the repo root, it just works.
#
# Numbering matches `_MANUAL_REVIEW.md` / `_OLIVE_PLAN.md` in
# `results/stall_sites/enrichment/olive_reports/`.
# All 18 blocks are active; their .qmd files have been generated.
#----------------------------------------------------


# cd to repo root (two levels up) so the relative paths below resolve. (This block is bash-only and is skipped by anyone
# copy-pasting an individual numbered block.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# ============================================================
# 1. between_condition_wilcoxon_aa
# ============================================================
echo "---- 1. between_condition_wilcoxon_aa ----"
quarto render "results/stall_sites/enrichment/olive_reports/between_condition_wilcoxon_aa.qmd"

# ============================================================
# 2. between_condition_wilcoxon_codon
# ============================================================
echo "---- 2. between_condition_wilcoxon_codon ----"
quarto render "results/stall_sites/enrichment/olive_reports/between_condition_wilcoxon_codon.qmd"

# ============================================================
# 3. between_timepoint_wilcoxon_d10_vs_d0_aa
# ============================================================
echo "---- 3. between_timepoint_wilcoxon_d10_vs_d0_aa ----"
quarto render "results/stall_sites/enrichment/olive_reports/between_timepoint_wilcoxon_d10_vs_d0_aa.qmd"

# ============================================================
# 4. between_timepoint_wilcoxon_d10_vs_d0_codon
# ============================================================
echo "---- 4. between_timepoint_wilcoxon_d10_vs_d0_codon ----"
quarto render "results/stall_sites/enrichment/olive_reports/between_timepoint_wilcoxon_d10_vs_d0_codon.qmd"

# ============================================================
# 5. between_timepoint_wilcoxon_d10_vs_d5_aa
# ============================================================
echo "---- 5. between_timepoint_wilcoxon_d10_vs_d5_aa ----"
quarto render "results/stall_sites/enrichment/olive_reports/between_timepoint_wilcoxon_d10_vs_d5_aa.qmd"

# ============================================================
# 6. between_timepoint_wilcoxon_d10_vs_d5_codon
# ============================================================
echo "---- 6. between_timepoint_wilcoxon_d10_vs_d5_codon ----"
quarto render "results/stall_sites/enrichment/olive_reports/between_timepoint_wilcoxon_d10_vs_d5_codon.qmd"

# ============================================================
# 7. between_timepoint_wilcoxon_d5_vs_d0_aa
# ============================================================
echo "---- 7. between_timepoint_wilcoxon_d5_vs_d0_aa ----"
quarto render "results/stall_sites/enrichment/olive_reports/between_timepoint_wilcoxon_d5_vs_d0_aa.qmd"

# ============================================================
# 8. between_timepoint_wilcoxon_d5_vs_d0_codon
# ============================================================
echo "---- 8. between_timepoint_wilcoxon_d5_vs_d0_codon ----"
quarto render "results/stall_sites/enrichment/olive_reports/between_timepoint_wilcoxon_d5_vs_d0_codon.qmd"

# ============================================================
# 9. per_timepoint_fisher_aa
# ============================================================
echo "---- 9. per_timepoint_fisher_aa ----"
quarto render "results/stall_sites/enrichment/olive_reports/per_timepoint_fisher_aa.qmd"

# ============================================================
# 10. per_timepoint_fisher_codon
# ============================================================
echo "---- 10. per_timepoint_fisher_codon ----"
quarto render "results/stall_sites/enrichment/olive_reports/per_timepoint_fisher_codon.qmd"

# ============================================================
# 11. timepoint_fisher_within_condition_d10_vs_d0_aa
# ============================================================
echo "---- 11. timepoint_fisher_within_condition_d10_vs_d0_aa ----"
quarto render "results/stall_sites/enrichment/olive_reports/timepoint_fisher_within_condition_d10_vs_d0_aa.qmd"

# ============================================================
# 12. timepoint_fisher_within_condition_d10_vs_d0_codon
# ============================================================
echo "---- 12. timepoint_fisher_within_condition_d10_vs_d0_codon ----"
quarto render "results/stall_sites/enrichment/olive_reports/timepoint_fisher_within_condition_d10_vs_d0_codon.qmd"

# ============================================================
# 13. timepoint_fisher_within_condition_d10_vs_d5_aa
# ============================================================
echo "---- 13. timepoint_fisher_within_condition_d10_vs_d5_aa ----"
quarto render "results/stall_sites/enrichment/olive_reports/timepoint_fisher_within_condition_d10_vs_d5_aa.qmd"

# ============================================================
# 14. timepoint_fisher_within_condition_d10_vs_d5_codon
# ============================================================
echo "---- 14. timepoint_fisher_within_condition_d10_vs_d5_codon ----"
quarto render "results/stall_sites/enrichment/olive_reports/timepoint_fisher_within_condition_d10_vs_d5_codon.qmd"

# ============================================================
# 15. timepoint_fisher_within_condition_d5_vs_d0_aa
# ============================================================
echo "---- 15. timepoint_fisher_within_condition_d5_vs_d0_aa ----"
quarto render "results/stall_sites/enrichment/olive_reports/timepoint_fisher_within_condition_d5_vs_d0_aa.qmd"

# ============================================================
# 16. timepoint_fisher_within_condition_d5_vs_d0_codon
# ============================================================
echo "---- 16. timepoint_fisher_within_condition_d5_vs_d0_codon ----"
quarto render "results/stall_sites/enrichment/olive_reports/timepoint_fisher_within_condition_d5_vs_d0_codon.qmd"

# ============================================================
# 17. within_condition_binomial_aa
# ============================================================
echo "---- 17. within_condition_binomial_aa ----"
quarto render "results/stall_sites/enrichment/olive_reports/within_condition_binomial_aa.qmd"

# ============================================================
# 18. within_condition_binomial_codon
# ============================================================
echo "---- 18. within_condition_binomial_codon ----"
quarto render "results/stall_sites/enrichment/olive_reports/within_condition_binomial_codon.qmd"
