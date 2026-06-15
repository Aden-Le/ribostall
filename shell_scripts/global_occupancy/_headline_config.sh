#!/bin/bash
#----------------------------------------------------
# Shared headline / direction config — GLOBAL OCCUPANCY pipeline.
#
# Sourced by BOTH the stats runner (run_global_codon_occ_stats.sh, which passes
# the numerator to --headline-condition) AND the analyze_*.sh plot launchers
# (which derive the between-condition plot labels from these values). Because the
# stats direction and the plot labels come from this ONE file, they cannot drift.
#
# NOTE: global occupancy has no background-aware-diff analysis (it normalizes
# every condition to a single shared transcriptome background, which would make
# such a test collapse into the per-timepoint Fisher), so there is no enrichment-
# ratio x-label here — only the Wilcoxon (Analysis 2) and per-timepoint Fisher
# (Analysis 4) directions are headline-driven.
#
# Edit ONLY the two condition names below. The "Derived" block follows them.
#----------------------------------------------------

# The headline condition = numerator / direction reference. A positive effect
# (log2_FC / log2 odds ratio) means "enriched in this condition". Must be one of
# the two condition labels in the runner's EXP_GROUPS (part before the first
# underscore, e.g. BWM or control).
HEADLINE_CONDITION="BWM"
# The other condition (denominator).
OTHER_CONDITION="control"

# --- Derived (do not edit): labels follow the headline so they cannot drift ---
# Used as the per-timepoint Fisher --comparison-label and in titles.
COMPARISON_LABEL="${HEADLINE_CONDITION} vs ${OTHER_CONDITION}"
# Used as the between-condition Wilcoxon --comparison value (a plot prefix/label).
COMPARISON_TAG="${HEADLINE_CONDITION}_vs_${OTHER_CONDITION}"
