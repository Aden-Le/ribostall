#!/bin/bash
#----------------------------------------------------
# Shared headline / direction config — CONSENSUS UNION stall-sites pipeline.
#
# Sourced by BOTH the stats runner (run_stall_sites_consensus_union_stats.sh,
# which passes the numerator to --headline-condition) AND the analyze_*.sh plot
# launchers (which derive every between-condition plot label from these values).
# Because the stats direction and the plot labels come from this ONE file, they
# cannot drift out of sync: change the headline here and everything downstream
# follows.
#
# Edit ONLY the two condition names below. The "Derived" block follows them.
#----------------------------------------------------

# The headline condition = numerator / direction reference. A positive effect
# (delta_log2_enrichment) means "enriched vs background in this condition".
# Must be one of the two condition labels in the runner's EXP_GROUPS (the part
# before the first underscore, e.g. BWM or control).
HEADLINE_CONDITION="BWM"
# The other condition (denominator).
OTHER_CONDITION="control"

# --- Derived (do not edit): labels follow the headline so they cannot drift ---
# Used as background-diff --comparison-label and in titles.
COMPARISON_LABEL="${HEADLINE_CONDITION} vs ${OTHER_CONDITION}"
# Injected grouping-column value / plot file prefix (consensus CSVs have no
# timepoint/condition column, so the launchers add a constant one).
COMPARISON_TAG="${HEADLINE_CONDITION}_vs_${OTHER_CONDITION}"
# Used as the background-diff x-axis label (enrichment ratio, already log2).
X_LABEL_RATIO="Log2 Enrichment Ratio (${HEADLINE_CONDITION} / ${OTHER_CONDITION})"
