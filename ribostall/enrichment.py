"""
Statistical enrichment tests for amino acid composition at ribosome stall sites.

Four analyses:
  1. Within-condition enrichment (binomial test vs genome background)
  2. Between-condition overall (Wilcoxon rank-sum, n=6 vs n=6)
  3. Between-timepoint (Wilcoxon pooled across conditions + Fisher's within condition)
  4. Between-condition per-timepoint (Fisher's exact test)

Each E/P/A site is tested INDEPENDENTLY — never accumulated across sites.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from ribostall.stats_core import (
    bh_fdr,
    apply_bh_fdr,
    binom_row,
    wilcoxon_row,
    fisher_row,
    background_diff_row,
)

# bh_fdr is re-exported from stats_core so existing
# `from ribostall.enrichment import bh_fdr` callers keep working.
__all__ = [
    "bh_fdr",
    "within_condition_enrichment",
    "between_condition_wilcoxon",
    "between_timepoint_wilcoxon",
    "between_timepoint_fisher_within_condition",
    "per_timepoint_fisher",
    "between_condition_fisher",
    "between_condition_background_diff",
    "between_timepoint_background_diff",
    "plot_coverage_density",
]


# ---------------------------------------------------------------------------
# Analysis 1: Within-condition enrichment (Binomial test)
# ---------------------------------------------------------------------------
def within_condition_enrichment(
    replicate_counts: dict,
    bg_freq_per_group: dict,
    rep_to_condition: dict,
    rep_to_group: dict,
    *,
    feature_col: str = "amino_acid",
) -> pd.DataFrame:
    """
    For each group, test whether each amino acid at each E/P/A site is
    enriched or depleted vs that group's background AA frequencies.

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series(AA->count), "P": Series(AA->count), "A": Series(AA->count)}}
    bg_freq_per_group : dict
        {group: pd.Series} — background AA frequencies per group.
    rep_to_condition : dict
        {replicate: "control" or "BWM"}
    rep_to_group : dict
        {replicate: group_name}

    Returns
    -------
    pd.DataFrame with columns:
        site, group, condition, timepoint, <feature_col>, observed_count, total_n,
        observed_freq, bg_freq, log2_enrichment, weighted_log2_enrichment, p_value, p_adj
    """
    # Get all groups present in the replicate counts: Ex: "control_day_0", "BWM_day_5", etc.
    groups = sorted(set(rep_to_group[r] for r in replicate_counts if r in rep_to_group))

    rows = []

    for group in groups:
        # Background frequencies for this group
        bg_freq = bg_freq_per_group[group]
        # Replicates in this group | Ex: "control_day_0" -> ["control_day_0_rep1", "control_day_0_rep2"]
        group_reps = [r for r in replicate_counts if rep_to_group.get(r) == group]
        # Condition for this group (e.g. "control" or "BWM")
        condition = rep_to_condition.get(group_reps[0], "") if group_reps else ""
        parts = group.split("_", 1)
        timepoint = parts[1] if len(parts) > 1 else group

        for site in ("E", "P", "A"):
            # Pool counts across replicates in this group
            pooled = None
            for rep in group_reps:
                # counts is a Series of AA->count for this replicate and site
                counts = replicate_counts[rep][site]
                if pooled is None:
                    pooled = counts.copy()
                else:
                    pooled = pooled.add(counts, fill_value=0)

            if pooled is None:
                continue
            
            # Total n is the total count of all amino acids/codons at this site in this group (not background)
            total_n = int(pooled.sum())
            if total_n == 0:
                continue

            for aa in pooled.index:
                # k is the sum of target aa/codon for both replicates at this site
                k = int(pooled[aa])
                # p_bg is the background frequency of this aa in this group (with small pseudocount to avoid zero)
                p_bg = float(bg_freq.get(aa, 1e-6))

                # For output, we want the raw p-value. Multiple testing correction will be done later per (group, site).
                res = binom_row(k, total_n, p_bg)
                rows.append({
                    "site": site,
                    "group": group,
                    "condition": condition,
                    "timepoint": timepoint,
                    feature_col: aa,
                    "observed_count": res["observed_count"],
                    "total_n": res["total_n"],
                    "observed_freq": res["observed_freq"],
                    "bg_freq": res["bg_freq"],
                    "log2_enrichment": res["log2_enrichment"],
                    "weighted_log2_enrichment": res["weighted_log2_enrichment"],
                    "p_value": res["p_value"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per (group, site): each E/P/A site is treated as its own
    # family of hypotheses ("which AAs stand out at this position in this group?").
    df = apply_bh_fdr(df, ["group", "site"])
    return df.sort_values(["group", "site", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 2: Between-condition overall (Wilcoxon rank-sum)
# ---------------------------------------------------------------------------
def between_condition_wilcoxon(
    replicate_counts: dict,
    rep_to_condition: dict,
    *,
    feature_col: str = "amino_acid",
    headline_condition: Optional[str] = None,
) -> pd.DataFrame:
    """
    For each feature (amino acid or codon) at each E/P/A site, compare
    per-replicate stall frequencies between conditions using Wilcoxon
    rank-sum (Mann-Whitney U).

    Direction: ``log2_FC`` is ``log2(median_cond_a / median_cond_b)``, so a
    positive value means the feature has a higher per-replicate stall frequency
    in ``cond_a``. ``cond_a`` is the *headline* condition.

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series, "P": Series, "A": Series}}
    rep_to_condition : dict
        {replicate: "control" or "BWM"}
    feature_col : str
        Output column name for the feature level (e.g. "amino_acid" or "codon").
    headline_condition : str or None
        Which of the two conditions is the headline (``cond_a``, the log2_FC
        numerator), so a positive ``log2_FC`` means higher in it. Must equal one
        of the two condition labels. If ``None`` (default), the conditions are
        ordered alphabetically — backward-compatible behaviour.

    Returns
    -------
    pd.DataFrame with columns:
        site, <feature_col>, median_<cond_a>, median_<cond_b>, log2_FC, U_stat, p_value, p_adj
    """
    conditions = sorted(set(rep_to_condition.values()))
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, got {conditions}")
    if headline_condition is not None:
        if headline_condition not in conditions:
            raise ValueError(
                f"headline_condition {headline_condition!r} is not one of the "
                f"two conditions {conditions}"
            )
        # Headline becomes cond_a (log2_FC numerator): positive log2_FC means
        # higher per-replicate frequency in headline_condition.
        cond_a = headline_condition
        cond_b = next(c for c in conditions if c != headline_condition)
    else:
        # Default: alphabetical order for consistent output column naming
        # (e.g. median_BWM, median_control).
        cond_a, cond_b = conditions  # alphabetical: BWM, control

    # Compute per-replicate frequencies
    # Example: rep_freqs = {
    #   "control_day_0_rep1": {"E": Series(unit->freq), "P": Series(unit->freq), "A": Series(unit->freq)},
    #   "control_day_0_rep2": {"E": Series(unit->freq), "P": Series(unit->freq), "A": Series(unit->freq)},
    #   ...
    rep_freqs = {}
    for rep, site_counts in replicate_counts.items():
        rep_freqs[rep] = {}
        for site in ("E", "P", "A"):
            # Gets each unit's counts for this site
            counts = site_counts[site]
            # Gets the total counts of all units at this site in this replicate
            total = counts.sum()
            # Converts them into frequencies for each unit at this site in this replicate
            rep_freqs[rep][site] = counts / total if total > 0 else counts * 0.0

    # Get unit list from first replicate
    first_rep = next(iter(replicate_counts))
    unit_list = list(replicate_counts[first_rep]["E"].index)

    rows = []
    for site in ("E", "P", "A"):
        for unit in unit_list:
            # Get frequencies for this unit at this site across replicates in each condition
            freqs_a = [rep_freqs[r][site][unit] for r in rep_freqs if rep_to_condition.get(r) == cond_a]
            freqs_b = [rep_freqs[r][site][unit] for r in rep_freqs if rep_to_condition.get(r) == cond_b]

            res = wilcoxon_row(freqs_a, freqs_b)
            rows.append({
                "site": site,
                feature_col: unit,
                f"median_{cond_a}": res["median_a"],
                f"median_{cond_b}": res["median_b"],
                "log2_FC": res["log2_FC"],
                "U_stat": res["U_stat"],
                "p_value": res["p_value"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # BH-FDR correction per site: each E/P/A site is treated as its own family.
    df = apply_bh_fdr(df, ["site"])
    return df.sort_values(["site", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 3a: Between-timepoint Wilcoxon (pooled across conditions)
# ---------------------------------------------------------------------------
def between_timepoint_wilcoxon(
    replicate_counts: dict,
    rep_to_timepoint: dict,
    *,
    feature_col: str = "amino_acid",
    time_a: str = "day_10",
    time_b: str = "day_0",
) -> pd.DataFrame:
    """
    For each (site, feature), compare per-replicate stall frequencies between
    two timepoints, pooling across conditions. Day_a reps (n=4) vs Day_b reps
    (n=4) when both BWM and control replicates exist at each timepoint.

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series, "P": Series, "A": Series}}
    rep_to_timepoint : dict
        {replicate: "day_0", "day_5", or "day_10"}
    feature_col : str
        Output column name for the feature level (e.g. "amino_acid" or "codon").
    time_a, time_b : str
        The two timepoints to compare.

    Returns
    -------
    pd.DataFrame with columns:
        site, <feature_col>, median_<time_a>, median_<time_b>, log2_FC,
        U_stat, p_value, p_adj
    """
    # Compute per-replicate frequencies (same logic as between_condition_wilcoxon)
    # Example: rep_freqs = {
    #   "BWM_day_0_rep1": {"E": Series(unit->freq), "P": Series(unit->freq), "A": Series(unit->freq)},
    #   ...
    # }

    # Turns raw counts into frequencies for each replicate and site
    rep_freqs = {}
    for rep, site_counts in replicate_counts.items():
        rep_freqs[rep] = {}
        for site in ("E", "P", "A"):
            counts = site_counts[site]
            total = counts.sum()
            rep_freqs[rep][site] = counts / total if total > 0 else counts * 0.0

    # Get unit list from first replicate
    first_rep = next(iter(replicate_counts))
    unit_list = list(replicate_counts[first_rep]["E"].index)

    rows = []
    for site in ("E", "P", "A"):
        for unit in unit_list:
            # All reps with timepoint == time_a, pooled across conditions (n=4 typically)
            freqs_a = [rep_freqs[r][site][unit] for r in rep_freqs if rep_to_timepoint.get(r) == time_a]
            # All reps with timepoint == time_b, pooled across conditions (n=4 typically)
            freqs_b = [rep_freqs[r][site][unit] for r in rep_freqs if rep_to_timepoint.get(r) == time_b]

            res = wilcoxon_row(freqs_a, freqs_b)
            rows.append({
                "site": site,
                feature_col: unit,
                f"median_{time_a}": res["median_a"],
                f"median_{time_b}": res["median_b"],
                "log2_FC": res["log2_FC"],
                "U_stat": res["U_stat"],
                "p_value": res["p_value"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # BH-FDR correction per site: each E/P/A site is treated as its own family.
    df = apply_bh_fdr(df, ["site"])
    return df.sort_values(["site", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 3b: Between-timepoint Fisher's (within each condition)
# ---------------------------------------------------------------------------
def between_timepoint_fisher_within_condition(
    replicate_counts: dict,
    rep_to_condition: dict,
    rep_to_timepoint: dict,
    *,
    feature_col: str = "amino_acid",
    time_a: str = "day_10",
    time_b: str = "day_0",
) -> pd.DataFrame:
    """
    Within each condition × site, pool replicates at time_a and time_b and
    run Fisher's exact test on each unit (amino acid or codon).

    NOTE: pooling biological replicates is pseudoreplication. P-values should
    be interpreted cautiously.

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series, "P": Series, "A": Series}}
    rep_to_condition : dict
        {replicate: "control" or "BWM"}
    rep_to_timepoint : dict
        {replicate: "day_0", "day_5", or "day_10"}
    feature_col : str
        Output column name for the feature level (e.g. "amino_acid" or "codon").
    time_a, time_b : str
        The two timepoints to compare.

    Returns
    -------
    pd.DataFrame with columns:
        condition, site, <feature_col>, <time_a>_count, <time_a>_total,
        <time_b>_count, <time_b>_total, odds_ratio, p_value, p_adj
    """
    # All conditions present, sorted alphabetically (e.g. ["BWM", "control"])
    conditions = sorted(set(rep_to_condition.values()))

    # Unit list from first replicate
    first_rep = next(iter(replicate_counts))
    unit_list = list(replicate_counts[first_rep]["E"].index)

    rows = []
    # For each condition (BWM, control)
    for cond in conditions:
        for site in ("E", "P", "A"):
            # Pool counts across reps at this (condition, site) for each timepoint
            pooled_by_tp = {}
            for tp in (time_a, time_b):
                pooled = None
                for rep in replicate_counts:
                    if rep_to_condition.get(rep) != cond or rep_to_timepoint.get(rep) != tp:
                        continue
                    counts = replicate_counts[rep][site]
                    if pooled is None:
                        pooled = counts.copy()
                    else:
                        pooled = pooled.add(counts, fill_value=0)
                pooled_by_tp[tp] = pooled

            # Skip if either timepoint has no replicates for this condition
            if any(v is None for v in pooled_by_tp.values()):
                continue

            total_a = int(pooled_by_tp[time_a].sum())
            total_b = int(pooled_by_tp[time_b].sum())
            if total_a == 0 or total_b == 0:
                continue

            for unit in unit_list:
                # 2x2 table: [[time_a_unit, time_a_notUnit], [time_b_unit, time_b_notUnit]]
                count_a = int(pooled_by_tp[time_a].get(unit, 0))
                count_b = int(pooled_by_tp[time_b].get(unit, 0))

                res = fisher_row(count_a, total_a, count_b, total_b)
                rows.append({
                    "site": site,
                    "condition": cond,
                    feature_col: unit,
                    f"{time_a}_count": count_a,
                    f"{time_a}_total": total_a,
                    f"{time_b}_count": count_b,
                    f"{time_b}_total": total_b,
                    "odds_ratio": res["odds_ratio"],
                    "p_value": res["p_value"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per (condition, site): each E/P/A site within a condition
    # is treated as its own family of hypotheses.
    df = apply_bh_fdr(df, ["condition", "site"])
    return df.sort_values(["condition", "site", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 4: Per-timepoint between-condition (Fisher's exact test)
# ---------------------------------------------------------------------------
def per_timepoint_fisher(
    replicate_counts: dict,
    rep_to_condition: dict,
    rep_to_timepoint: dict,
    *,
    feature_col: str = "amino_acid",
    headline_condition: Optional[str] = None,
    timepoints: list,
) -> pd.DataFrame:
    """
    For each timepoint, pool 2 reps per condition and run Fisher's exact test
    on each unit (amino acid or codon) at each E/P/A site.

    NOTE: pooling 2 biological replicates is pseudoreplication. P-values should
    be interpreted cautiously.

    Direction: the 2x2 odds ratio is computed as (odds in ``cond_a``) /
    (odds in ``cond_b``), so a positive log2(odds_ratio) means the unit is
    enriched in ``cond_a``. ``cond_a`` is the *headline* condition.

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series, "P": Series, "A": Series}}
    rep_to_condition : dict
        {replicate: "control" or "BWM"}
    rep_to_timepoint : dict
        {replicate: "day_0", "day_5", or "day_10"}
    feature_col : str
        Output column name for the feature level (e.g. "amino_acid" or "codon").
    headline_condition : str or None
        Which of the two conditions is the headline (``cond_a``, the odds-ratio
        numerator), so a positive log2(odds_ratio) means enriched in it. Must
        equal one of the two condition labels. If ``None`` (default), the
        conditions are ordered alphabetically — backward-compatible behaviour.
    timepoints : list
        Timepoint labels in the desired (chronological) order, e.g.
        ``["day_0", "day_5", "day_10"]``. Sets both the iteration order and the
        order of the timepoint blocks in the output. Timepoints are NOT sorted
        automatically — a string sort places "day_10" before "day_5".

    Returns
    -------
    pd.DataFrame with columns:
        timepoint, site, <feature_col>, <cond>_count, <cond>_total,
        odds_ratio, p_value, p_adj
    """
    # Conditions are sorted alphabetically (the headline/direction default).
    # Timepoints keep the caller-declared order (no sorting).
    conditions = sorted(set(rep_to_condition.values())) # e.g. ["BWM", "control"]

    # Direction: cond_a is the odds-ratio numerator (positive log2(OR) = enriched
    # in cond_a). headline_condition makes that the headline; default alphabetical.
    if headline_condition is not None:
        if headline_condition not in conditions:
            raise ValueError(
                f"headline_condition {headline_condition!r} is not one of the "
                f"conditions {conditions}"
            )
        cond_a = headline_condition
        cond_b = next(c for c in conditions if c != headline_condition)
    else:
        cond_a, cond_b = conditions[0], conditions[1]

    # Gets the unit list using the first replicate (assumes all replicates have the same index)
    first_rep = next(iter(replicate_counts))
    unit_list = list(replicate_counts[first_rep]["E"].index)

    rows = []
    # For each time point, day0, day5, day10
    for tp in timepoints:
        for site in ("E", "P", "A"):
            # Pool counts per condition at this timepoint
            pooled_by_cond = {}
            # BWM or Control
            for cond in conditions:
                pooled = None
                for rep in replicate_counts:
                    # if this replicate doesn't match the current condition and timepoint, skip it
                    if rep_to_condition.get(rep) != cond or rep_to_timepoint.get(rep) != tp:
                        continue
                    # The counts of all units at this site in this replicate
                    counts = replicate_counts[rep][site]
                    if pooled is None:
                        pooled = counts.copy()
                    else:
                        pooled = pooled.add(counts, fill_value=0)
                # Pooled replicate counts for each condition at this timepoint and site
                pooled_by_cond[cond] = pooled

            # Check both conditions have data
            if any(v is None for v in pooled_by_cond.values()):
                continue

            # For each unit, build the 2x2 contingency table and run Fisher's exact test
            for unit in unit_list:
                # 2x2 table: [[cond_a_unit, cond_a_notUnit], [cond_b_unit, cond_b_notUnit]]
                # — row 0 = cond_a (headline / odds-ratio numerator), row 1 = cond_b.
                counts = {cond: int(pooled_by_cond[cond].get(unit, 0)) for cond in conditions}
                totals = {cond: int(pooled_by_cond[cond].sum()) for cond in conditions}
                res = fisher_row(counts[cond_a], totals[cond_a], counts[cond_b], totals[cond_b])

                row = {
                    "site": site,
                    "timepoint": tp,
                    feature_col: unit,
                    "odds_ratio": res["odds_ratio"],
                    "p_value": res["p_value"],
                }
                for cond in conditions:
                    row[f"{cond}_count"] = counts[cond]
                    row[f"{cond}_total"] = totals[cond]
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per (timepoint, site): each E/P/A site within a timepoint
    # is treated as its own family of hypotheses.
    df = apply_bh_fdr(df, ["timepoint", "site"])
    # Order the timepoint blocks as declared (not lexicographic, where "day_10"
    # sorts before "day_5"); site/p_adj ordering within a timepoint is unchanged.
    tp_rank = {tp: i for i, tp in enumerate(timepoints)}
    df["_tp_order"] = df["timepoint"].map(tp_rank)
    return df.sort_values(["_tp_order", "site", "p_adj"]).drop(columns="_tp_order")


# ---------------------------------------------------------------------------
# Analysis 5: Between-condition (Fisher's exact test, timepoint-free)
# ---------------------------------------------------------------------------
def between_condition_fisher(
    replicate_counts: dict,
    rep_to_condition: dict,
    *,
    feature_col: str = "amino_acid",
    headline_condition: Optional[str] = None,
) -> pd.DataFrame:
    """
    For each unit (amino acid or codon) at each E/P/A site, pool replicate
    counts per condition and run Fisher's exact test comparing the two
    conditions. This is the timepoint-free counterpart of
    ``per_timepoint_fisher`` — there is only one comparison per site.

    NOTE: pooling replicates is pseudoreplication. P-values should be
    interpreted cautiously.

    Direction: the 2x2 odds ratio is computed as (odds in ``cond_a``) /
    (odds in ``cond_b``), so a positive log2(odds_ratio) means the unit is
    enriched in ``cond_a``. ``cond_a`` is the *headline* condition.

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series, "P": Series, "A": Series}}
    rep_to_condition : dict
        {replicate: "control" or "treatment"}
    feature_col : str
        Output column name for the feature level (e.g. "amino_acid" or "codon").
    headline_condition : str or None
        Which of the two conditions is the headline (``cond_a``, the odds-ratio
        numerator), so a positive log2(odds_ratio) means enriched in it. Must
        equal one of the two condition labels. If ``None`` (default), the
        conditions are ordered alphabetically — backward-compatible behaviour.

    Returns
    -------
    pd.DataFrame with columns:
        site, <feature_col>, <cond_a>_count, <cond_a>_total, <cond_b>_count,
        <cond_b>_total, odds_ratio, p_value, p_adj
    """
    conditions = sorted(set(rep_to_condition.values()))
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, got {conditions}")
    if headline_condition is not None:
        if headline_condition not in conditions:
            raise ValueError(
                f"headline_condition {headline_condition!r} is not one of the "
                f"two conditions {conditions}"
            )
        # Headline becomes cond_a (odds-ratio numerator): positive log2(OR)
        # means enriched in headline_condition.
        cond_a = headline_condition
        cond_b = next(c for c in conditions if c != headline_condition)
    else:
        # Default: alphabetical ordering (consistent column naming).
        cond_a, cond_b = conditions

    # Unit list from first replicate (assumes all replicates share the same index).
    # first rep would be the first key in the replicate_counts dictionary
    first_rep = next(iter(replicate_counts))
    # Unit List would be the index of the first replicate's E column (basically the AAs)
    unit_list = list(replicate_counts[first_rep]["E"].index)

    rows = []
    for site in ("E", "P", "A"):
        # Pool counts per condition across replicates at this site.
        pooled_by_cond = {}
        for cond in conditions:
            pooled = None
            for rep in replicate_counts:
                if rep_to_condition.get(rep) != cond:
                    continue
                counts = replicate_counts[rep][site]
                if pooled is None:
                    pooled = counts.copy()
                else:
                    pooled = pooled.add(counts, fill_value=0)
            pooled_by_cond[cond] = pooled

        # Skip the site if either condition has no replicates.
        if any(v is None for v in pooled_by_cond.values()):
            continue

        total_a = int(pooled_by_cond[cond_a].sum())
        total_b = int(pooled_by_cond[cond_b].sum())
        if total_a == 0 or total_b == 0:
            continue

        for unit in unit_list:
            # 2x2 table: [[cond_a_unit, cond_a_notUnit], [cond_b_unit, cond_b_notUnit]]
            count_a = int(pooled_by_cond[cond_a].get(unit, 0))
            count_b = int(pooled_by_cond[cond_b].get(unit, 0))

            res = fisher_row(count_a, total_a, count_b, total_b)
            rows.append({
                "site": site,
                feature_col: unit,
                f"{cond_a}_count": count_a,
                f"{cond_a}_total": total_a,
                f"{cond_b}_count": count_b,
                f"{cond_b}_total": total_b,
                "odds_ratio": res["odds_ratio"],
                "p_value": res["p_value"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per site: each E/P/A site is its own family of hypotheses.
    df = apply_bh_fdr(df, ["site"])
    return df.sort_values(["site", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 6: Between-condition, background-aware (binomial-vs-background offset)
# ---------------------------------------------------------------------------
def between_condition_background_diff(
    replicate_counts: dict,
    rep_to_condition: dict,
    bg_freq_per_cond: dict,
    *,
    feature_col: str = "amino_acid",
    headline_condition: Optional[str] = None,
) -> pd.DataFrame:
    """
    Background-aware counterpart of ``between_condition_fisher``. For each unit
    (amino acid or codon) at each E/P/A site, it asks whether the unit's
    enrichment OVER ITS OWN BACKGROUND differs between the two conditions,
    rather than whether the raw stall-site share differs.

    Each condition's enrichment is ``stall_count / (stall_total * bg_freq)`` —
    exactly Analysis 1's quantity. The effect size ``delta_log2_enrichment`` is
    the difference of the two within-condition log2 enrichments, and the test is
    the exact conditional binomial for two equal Poisson rates with background
    as the offset (see ``stats_core.background_diff_row``).

    Why this exists: Fisher compares raw shares, so a shift in the expressed /
    translated transcriptome between conditions can masquerade as differential
    stalling. Normalizing each condition to its own background removes that
    confound; the two tests agree when the backgrounds match and diverge only
    when they differ (which is itself the diagnostic).

    NOTE: pooling replicates is pseudoreplication. As with ``between_condition_
    fisher``, this does not model biological-replicate variability, so p-values
    should be interpreted cautiously.

    Direction: ``delta_log2_enrichment`` > 0 means the unit is more enriched
    (vs background) in the headline condition (``cond_headline``). If
    ``headline_condition`` is ``None`` the conditions are ordered alphabetically.

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series, "P": Series, "A": Series}}
    rep_to_condition : dict
        {replicate: "control" or "treatment"}
    bg_freq_per_cond : dict
        {condition: pd.Series} — background frequencies per condition (indexed by
        the same alphabet as ``replicate_counts``). In the consensus flat design
        group == condition, so the per-group backgrounds key directly here.
    feature_col : str
        Output column name for the feature level (e.g. "amino_acid" or "codon").
    headline_condition : str or None
        Which condition is the headline (numerator of the enrichment ratio).
        Must equal one of the two condition labels. Default: alphabetical.

    Returns
    -------
    pd.DataFrame with columns:
        site, <feature_col>, <cond_headline>_count, <cond_headline>_total,
        <cond_headline>_bg_freq, <cond_other>_count, <cond_other>_total,
        <cond_other>_bg_freq, log2_enrich_<cond_headline>,
        log2_enrich_<cond_other>, delta_log2_enrichment, enrichment_ratio,
        p_value, p_adj
    """
    conditions = sorted(set(rep_to_condition.values()))
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, got {conditions}")
    if headline_condition is not None:
        if headline_condition not in conditions:
            raise ValueError(
                f"headline_condition {headline_condition!r} is not one of the "
                f"two conditions {conditions}"
            )
        cond_headline = headline_condition
        cond_other = next(c for c in conditions if c != headline_condition)
    else:
        cond_headline, cond_other = conditions

    # This can be the amino acid list or the codon list
    # Unit list from first replicate (assumes all replicates share the same index).
    first_rep = next(iter(replicate_counts))
    unit_list = list(replicate_counts[first_rep]["E"].index)

    rows = []
    for site in ("E", "P", "A"):
        # Pool counts per condition across replicates at this site (same as Fisher).
        pooled_by_cond = {}
        for cond in conditions:
            pooled = None
            for rep in replicate_counts:
                if rep_to_condition.get(rep) != cond:
                    continue
                counts = replicate_counts[rep][site]
                if pooled is None:
                    pooled = counts.copy()
                else:
                    # Fill value is needed because if value didn't exist 5 + NaN wold be NaN
                    pooled = pooled.add(counts, fill_value=0)
            pooled_by_cond[cond] = pooled

        # Skip the site if either condition has no replicates.
        if any(v is None for v in pooled_by_cond.values()):
            continue
        
        # The total fo the headline
        stall_total_headline = int(pooled_by_cond[cond_headline].sum())
        # The total of the other condition (control)
        stall_total_other = int(pooled_by_cond[cond_other].sum())
        if stall_total_headline == 0 or stall_total_other == 0:
            continue
        
        # For each amino acid or codon
        for unit in unit_list:
            # Gets the count for that unit in each condition
            stall_count_headline = int(pooled_by_cond[cond_headline].get(unit, 0))
            stall_count_other = int(pooled_by_cond[cond_other].get(unit, 0))

            # Small pseudocount mirrors within_condition_enrichment so a unit
            # absent from a condition's background does not divide by zero.
            bg_freq_headline = float(bg_freq_per_cond[cond_headline].get(unit, 1e-6))
            bg_freq_other = float(bg_freq_per_cond[cond_other].get(unit, 1e-6))

            res = background_diff_row(
                stall_count_headline, stall_total_headline, bg_freq_headline,
                stall_count_other, stall_total_other, bg_freq_other,
            )
            
            rows.append({
                "site": site,
                feature_col: unit,
                f"{cond_headline}_count": stall_count_headline,
                f"{cond_headline}_total": stall_total_headline,
                f"{cond_headline}_bg_freq": bg_freq_headline,
                f"{cond_other}_count": stall_count_other,
                f"{cond_other}_total": stall_total_other,
                f"{cond_other}_bg_freq": bg_freq_other,
                f"log2_enrich_{cond_headline}": res["log2_enrich_headline"],
                f"log2_enrich_{cond_other}": res["log2_enrich_other"],
                "delta_log2_enrichment": res["delta_log2_enrichment"],
                "enrichment_ratio": res["enrichment_ratio"],
                "p_value": res["p_value"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per site: each E/P/A site is its own family of hypotheses.
    df = apply_bh_fdr(df, ["site"])
    return df.sort_values(["site", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 7: Between-timepoint, background-aware (pooled across conditions)
# ---------------------------------------------------------------------------
def between_timepoint_background_diff(
    replicate_counts: dict,
    rep_to_timepoint: dict,
    bg_freq_per_timepoint: dict,
    *,
    feature_col: str = "amino_acid",
    time_a: str = "day_10",
    time_b: str = "day_0",
) -> pd.DataFrame:
    """
    Between-timepoint counterpart of ``between_condition_background_diff``,
    pooling replicates ACROSS CONDITIONS within each timepoint. For each unit
    (amino acid or codon) at each E/P/A site it asks whether the unit's
    enrichment OVER ITS OWN BACKGROUND differs between two timepoints.

    Unlike the between-timepoint Fisher (``between_timepoint_fisher_within_
    condition``) — which (a) stays within each condition and (b) compares raw
    stall-site shares — this test pools both conditions' replicates at each
    timepoint (like ``between_timepoint_wilcoxon``) and normalizes each timepoint
    to its OWN pooled background, so a shift in the expressed transcriptome across
    the time course cannot masquerade as differential stalling.

    Each timepoint's enrichment is ``stall_count / (stall_total * bg_freq)`` —
    exactly Analysis 1's quantity. The effect size ``delta_log2_enrichment`` is
    the difference of the two within-timepoint log2 enrichments, and the test is
    the exact conditional binomial for two equal Poisson rates with background as
    the offset (see ``stats_core.background_diff_row``). The two agree with the
    Fisher counterpart when the two timepoints' backgrounds match and diverge only
    when they differ (which is itself the diagnostic).

    NOTE: pooling replicates is pseudoreplication. As with the other pooled
    tests, this does not model biological-replicate variability, so p-values
    should be interpreted cautiously.

    Direction: ``delta_log2_enrichment`` > 0 means the unit is more enriched (vs
    background) at ``time_a``. The two timepoints are compared in the fixed
    ``time_a`` vs ``time_b`` order, independent of any headline condition (the
    comparison is between timepoints, not conditions) — mirroring the other
    ``between_timepoint_*`` tests.

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series, "P": Series, "A": Series}}
    rep_to_timepoint : dict
        {replicate: "day_0", "day_5", or "day_10"}
    bg_freq_per_timepoint : dict
        {timepoint: pd.Series} — pooled-across-conditions background frequencies
        per timepoint, indexed by the same alphabet as ``replicate_counts``. The
        caller builds these by count-weighted pooling of the per-group
        backgrounds at each timepoint (sum bg_count across conditions, then
        renormalize).
    feature_col : str
        Output column name for the feature level (e.g. "amino_acid" or "codon").
    time_a, time_b : str
        The two timepoints to compare. ``time_a`` is the enrichment-ratio
        numerator (positive ``delta_log2_enrichment`` = more enriched at it).

    Returns
    -------
    pd.DataFrame with columns:
        site, time_a, time_b, <feature_col>, count_time_a, total_time_a,
        bg_freq_time_a, count_time_b, total_time_b, bg_freq_time_b,
        log2_enrich_time_a, log2_enrich_time_b, delta_log2_enrichment,
        enrichment_ratio, p_value, p_adj

    The ``time_a`` / ``time_b`` columns carry the timepoint *labels* (e.g.
    "day_10", "day_0") rather than baking them into the count/total column names,
    so several day-pairs can be concatenated into one combined CSV with identical
    columns.
    """
    # Unit list from first replicate (assumes all replicates share the same index).
    first_rep = next(iter(replicate_counts))
    unit_list = list(replicate_counts[first_rep]["E"].index)

    rows = []
    for site in ("E", "P", "A"):
        # Pool counts across ALL replicates at each timepoint (both conditions),
        # mirroring between_timepoint_wilcoxon's pooling.
        pooled_by_tp = {}
        for tp in (time_a, time_b):
            pooled = None
            for rep in replicate_counts:
                if rep_to_timepoint.get(rep) != tp:
                    continue
                counts = replicate_counts[rep][site]
                if pooled is None:
                    pooled = counts.copy()
                else:
                    pooled = pooled.add(counts, fill_value=0)
            pooled_by_tp[tp] = pooled

        # Skip the site if either timepoint has no replicates.
        if any(v is None for v in pooled_by_tp.values()):
            continue

        stall_total_a = int(pooled_by_tp[time_a].sum())
        stall_total_b = int(pooled_by_tp[time_b].sum())
        if stall_total_a == 0 or stall_total_b == 0:
            continue

        for unit in unit_list:
            stall_count_a = int(pooled_by_tp[time_a].get(unit, 0))
            stall_count_b = int(pooled_by_tp[time_b].get(unit, 0))

            # Small pseudocount mirrors within_condition_enrichment / the
            # between-condition diff so a unit absent from a timepoint's
            # background does not divide by zero.
            bg_freq_a = float(bg_freq_per_timepoint[time_a].get(unit, 1e-6))
            bg_freq_b = float(bg_freq_per_timepoint[time_b].get(unit, 1e-6))

            res = background_diff_row(
                stall_count_a, stall_total_a, bg_freq_a,
                stall_count_b, stall_total_b, bg_freq_b,
            )

            rows.append({
                "site": site,
                "time_a": time_a,
                "time_b": time_b,
                feature_col: unit,
                "count_time_a": stall_count_a,
                "total_time_a": stall_total_a,
                "bg_freq_time_a": bg_freq_a,
                "count_time_b": stall_count_b,
                "total_time_b": stall_total_b,
                "bg_freq_time_b": bg_freq_b,
                "log2_enrich_time_a": res["log2_enrich_headline"],
                "log2_enrich_time_b": res["log2_enrich_other"],
                "delta_log2_enrichment": res["delta_log2_enrichment"],
                "enrichment_ratio": res["enrichment_ratio"],
                "p_value": res["p_value"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per site: each E/P/A site is its own family of hypotheses
    # (mirrors between_timepoint_wilcoxon and between_condition_background_diff).
    df = apply_bh_fdr(df, ["site"])
    return df.sort_values(["site", "p_adj"])


# ---------------------------------------------------------------------------
# Coverage density plot
# ---------------------------------------------------------------------------
def plot_coverage_density(cov, groups, out_dir, trim_start: int = 0, trim_stop: int = 0):
    """
    Plot KDE density curves of per-transcript average coverage (reads/nt)
    for every replicate on a single composite figure.

    The per-transcript mean is computed on the elongation body — the CDS with
    the first ``trim_start`` and last ``trim_stop`` codons removed — to mirror
    the window used by ``filter_tx`` and ``call_stalls``.

    Parameters
    ----------
    cov : dict
        {replicate: {transcript: np.ndarray of per-nt counts, CDS-only}}
    groups : dict
        {group_name: [rep1, rep2, ...]}
    out_dir : str
        Directory to save the output figure (coverage_density.png).
    trim_start, trim_stop : int
        Number of codons to drop from the start / end of each CDS before
        computing the per-transcript mean.
    """
    trim_start_nt = trim_start * 3
    trim_stop_nt = trim_stop * 3

    def _body_mean(arr):
        arr = np.asarray(arr, float)
        if len(arr) <= trim_start_nt + trim_stop_nt:
            return 0.0
        body = arr[trim_start_nt : len(arr) - trim_stop_nt]
        return body.mean() if body.size else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))

    group_list = list(groups.keys())
    cmap = plt.cm.get_cmap("tab10", max(len(group_list), 1))
    group_colors = {grp: cmap(i) for i, grp in enumerate(group_list)}
    linestyles = ["-", "--", ":", "-."]

    for group, reps in groups.items():
        color = group_colors[group]
        for j, rep in enumerate(reps):
            if rep not in cov:
                continue
            tx_dict = cov[rep]
            means = np.array([_body_mean(v) for v in tx_dict.values()])
            means = means[means > 0]
            if len(means) < 2:
                continue

            log_means = np.log10(means)
            kde = stats.gaussian_kde(log_means)
            x = np.linspace(log_means.min() - 0.5, log_means.max() + 0.5, 500)
            ax.plot(
                x, kde(x),
                label=f"{rep} ({group})",
                color=color,
                linestyle=linestyles[j % len(linestyles)],
                linewidth=1.5,
            )

    ax.set_xlabel("log10(Mean coverage per transcript, reads/nt)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Per-transcript average coverage distribution\n"
        f"(elongation body: first {trim_start} and last {trim_stop} codons removed)"
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "coverage_density.png"), dpi=150)
    plt.close(fig)
