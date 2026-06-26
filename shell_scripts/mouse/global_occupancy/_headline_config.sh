#!/bin/bash
#----------------------------------------------------
# Shared headline / direction config — MOUSE GLOBAL OCCUPANCY pipeline.
#
# Sourced by BOTH the stats runner (run_global_codon_occ_stats.sh, which passes
# the numerator to --headline-condition) AND any analyze_*.sh plot launchers
# (which derive the between-condition plot labels from these values). Because the
# stats direction and the plot labels come from this ONE file, they cannot drift.
#
# NOTE: global occupancy normalizes every condition to a single shared
# transcriptome background, so there is no enrichment-ratio x-label here — only
# the Wilcoxon (Analysis 2) and per-timepoint Fisher (Analysis 3) directions are
# headline-driven.
#
# Edit ONLY the two condition names below. The "Derived" block follows them.
#----------------------------------------------------

# The headline condition = numerator / direction reference. A positive effect
# (log2_FC / log2 odds ratio) means "enriched in this condition". Must be one of
# the two condition labels in the runner's EXP_GROUPS (the part before the first
# underscore; for the flat mouse design the whole group label, e.g. treatment or
# control).
HEADLINE_CONDITION="treatment"
# The other condition (denominator).
OTHER_CONDITION="control"

# --- Derived (do not edit): labels follow the headline so they cannot drift ---
# Used as the per-timepoint Fisher --comparison-label and in titles.
COMPARISON_LABEL="${HEADLINE_CONDITION} vs ${OTHER_CONDITION}"
# Used as the between-condition Wilcoxon --comparison value (a plot prefix/label).
COMPARISON_TAG="${HEADLINE_CONDITION}_vs_${OTHER_CONDITION}"
