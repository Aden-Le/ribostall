"""
Statistical tests for global codon and amino acid occupancy analysis.

Four analyses:
  1. Within-condition enrichment (binomial test vs transcriptome background)
  2. Between-condition overall (Wilcoxon rank-sum, BWM vs Control, n=6 vs n=6)
  3. Between-timepoint (Wilcoxon pooled across conditions + Fisher's within condition)
  4. Per-timepoint between-condition (Fisher's exact test)

Each test is run independently for codon-level and amino acid-level occupancy.
"""

import numpy as np
import pandas as pd
from scipy import stats

from functions_folder.functions_enrichment import bh_fdr


# ---------------------------------------------------------------------------
# Analysis 1: Within-condition enrichment (Binomial test)
# ---------------------------------------------------------------------------
def within_condition_binomial_occupancy(
    raw_counts_by_exp: dict,
    transcriptome_counts: dict,
    groups: dict,
    rep_to_group: dict,
) -> pd.DataFrame:
    """
    For each group, test whether each codon/AA's share of ribosome reads
    differs from its share of the transcriptome (background frequency).

    Parameters
    ----------
    raw_counts_by_exp : dict
        {experiment: {unit: raw_read_sum}} where unit is codon or AA.
    transcriptome_counts : dict
        {unit: count_in_transcriptome} — background counts.
    groups : dict
        {group_name: [rep1, rep2, ...]}
    rep_to_group : dict
        {replicate: group_name}

    Returns
    -------
    pd.DataFrame with columns:
        group, condition, timepoint, unit, observed_count, total_n,
        observed_freq, bg_freq, log2_enrichment, weighted_log2,
        p_value, p_adj
    """
    # Background frequencies
    total_bg = sum(transcriptome_counts.values())
    bg_freq = {u: c / total_bg for u, c in transcriptome_counts.items()} if total_bg > 0 else {}

    all_units = sorted(transcriptome_counts.keys())
    rows = []

    for grp, reps in sorted(groups.items()):
        parts = grp.split("_", 1)
        condition = parts[0]
        timepoint = parts[1] if len(parts) > 1 else grp

        # Pool raw counts across replicates in this group
        pooled = {}
        for unit in all_units:
            pooled[unit] = sum(raw_counts_by_exp.get(rep, {}).get(unit, 0.0) for rep in reps)

        total_n = sum(pooled.values())
        if total_n == 0:
            continue

        for unit in all_units:
            k = int(round(pooled[unit]))
            p_bg = bg_freq.get(unit, 1e-6)
            freq = k / total_n if total_n > 0 else 0.0

            log2_enrich = np.log2(freq / p_bg) if freq > 0 and p_bg > 0 else 0.0
            weighted_log2 = freq * log2_enrich

            result = stats.binomtest(k, int(round(total_n)), p_bg, alternative="two-sided")

            rows.append({
                "group": grp,
                "condition": condition,
                "timepoint": timepoint,
                "unit": unit,
                "observed_count": k,
                "total_n": int(round(total_n)),
                "observed_freq": freq,
                "bg_freq": p_bg,
                "log2_enrichment": log2_enrich,
                "weighted_log2": weighted_log2,
                "p_value": result.pvalue,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per group
    dfs = []
    for grp in sorted(groups.keys()):
        mask = df["group"] == grp
        sub = df.loc[mask].copy()
        sub["p_adj"] = bh_fdr(sub["p_value"].values)
        dfs.append(sub)
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["group", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 2: Between-condition overall (Wilcoxon rank-sum)
# ---------------------------------------------------------------------------
def between_condition_wilcoxon_occupancy(
    rates_by_exp: dict,
    rep_to_condition: dict,
) -> pd.DataFrame:
    """
    Compare per-replicate normalized occupancy rates between conditions
    using Wilcoxon rank-sum (Mann-Whitney U). BWM vs Control, n=6 vs n=6.

    Parameters
    ----------
    rates_by_exp : dict
        {experiment: {unit: normalized_rate}}
    rep_to_condition : dict
        {replicate: "control" or "BWM"}

    Returns
    -------
    pd.DataFrame with columns:
        unit, median_{cond_a}, median_{cond_b}, log2_FC, U_stat, p_value, p_adj
    """
    conditions = sorted(set(rep_to_condition.values()))
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, got {conditions}")
    cond_a, cond_b = conditions  # alphabetical: BWM, control

    # Get all units from first replicate
    first_rep = next(iter(rates_by_exp))
    all_units = sorted(rates_by_exp[first_rep].keys())

    rows = []
    for unit in all_units:
        rates_a = np.array([rates_by_exp[r][unit] for r in rates_by_exp
                            if rep_to_condition.get(r) == cond_a], dtype=float)
        rates_b = np.array([rates_by_exp[r][unit] for r in rates_by_exp
                            if rep_to_condition.get(r) == cond_b], dtype=float)

        med_a = float(np.median(rates_a)) if len(rates_a) > 0 else 0.0
        med_b = float(np.median(rates_b)) if len(rates_b) > 0 else 0.0

        # log2 FC: cond_a / cond_b (BWM / control if alphabetical)
        if med_a > 0 and med_b > 0:
            log2_fc = np.log2(med_a / med_b)
        else:
            log2_fc = 0.0

        if len(rates_a) >= 2 and len(rates_b) >= 2:
            try:
                u_stat, p_val = stats.mannwhitneyu(rates_a, rates_b, alternative="two-sided")
            except ValueError:
                u_stat, p_val = np.nan, 1.0
        else:
            u_stat, p_val = np.nan, 1.0

        rows.append({
            "unit": unit,
            f"median_{cond_a}": med_a,
            f"median_{cond_b}": med_b,
            "log2_FC": log2_fc,
            "U_stat": u_stat,
            "p_value": p_val,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_adj"] = bh_fdr(df["p_value"].values)
    return df.sort_values("p_adj")


# ---------------------------------------------------------------------------
# Analysis 3a: Between-timepoint Wilcoxon (pooled across conditions)
# ---------------------------------------------------------------------------
def between_timepoint_wilcoxon_occupancy(
    rates_by_exp: dict,
    rep_to_timepoint: dict,
    time_a: str = "day_0",
    time_b: str = "day_10",
) -> pd.DataFrame:
    """
    Compare occupancy rates between two timepoints, pooling across conditions.
    Day 0 reps (n=4) vs Day 10 reps (n=4).

    Parameters
    ----------
    rates_by_exp : dict
        {experiment: {unit: normalized_rate}}
    rep_to_timepoint : dict
        {replicate: "day_0", "day_5", or "day_10"}
    time_a, time_b : str
        The two timepoints to compare.

    Returns
    -------
    pd.DataFrame with columns:
        unit, median_{time_a}, median_{time_b}, log2_FC, U_stat, p_value, p_adj
    """
    first_rep = next(iter(rates_by_exp))
    all_units = sorted(rates_by_exp[first_rep].keys())

    rows = []
    for unit in all_units:
        rates_a = np.array([rates_by_exp[r][unit] for r in rates_by_exp
                            if rep_to_timepoint.get(r) == time_a], dtype=float)
        rates_b = np.array([rates_by_exp[r][unit] for r in rates_by_exp
                            if rep_to_timepoint.get(r) == time_b], dtype=float)

        med_a = float(np.median(rates_a)) if len(rates_a) > 0 else 0.0
        med_b = float(np.median(rates_b)) if len(rates_b) > 0 else 0.0

        if med_a > 0 and med_b > 0:
            log2_fc = np.log2(med_a / med_b)
        else:
            log2_fc = 0.0

        if len(rates_a) >= 2 and len(rates_b) >= 2:
            try:
                u_stat, p_val = stats.mannwhitneyu(rates_a, rates_b, alternative="two-sided")
            except ValueError:
                u_stat, p_val = np.nan, 1.0
        else:
            u_stat, p_val = np.nan, 1.0

        rows.append({
            "unit": unit,
            f"median_{time_a}": med_a,
            f"median_{time_b}": med_b,
            "log2_FC": log2_fc,
            "U_stat": u_stat,
            "p_value": p_val,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_adj"] = bh_fdr(df["p_value"].values)
    return df.sort_values("p_adj")


# ---------------------------------------------------------------------------
# Analysis 3b: Between-timepoint Fisher's (within each condition)
# ---------------------------------------------------------------------------
def between_timepoint_fisher_within_condition(
    raw_counts_by_exp: dict,
    groups: dict,
    rep_to_condition: dict,
    rep_to_timepoint: dict,
    time_a: str = "day_0",
    time_b: str = "day_10",
) -> pd.DataFrame:
    """
    Within each condition, pool Day 0 and Day 10 reps into counts and run
    Fisher's exact test on a 2x2 contingency table.

    NOTE: pooling 2 biological replicates is pseudoreplication. P-values
    should be interpreted cautiously.

    Parameters
    ----------
    raw_counts_by_exp : dict
        {experiment: {unit: raw_read_sum}}
    groups : dict
        {group_name: [rep1, rep2, ...]}
    rep_to_condition : dict
        {replicate: "control" or "BWM"}
    rep_to_timepoint : dict
        {replicate: "day_0", "day_5", or "day_10"}
    time_a, time_b : str
        The two timepoints to compare.

    Returns
    -------
    pd.DataFrame with columns:
        condition, unit, {time_a}_count, {time_a}_total,
        {time_b}_count, {time_b}_total, odds_ratio, p_value, p_adj
    """
    conditions = sorted(set(rep_to_condition.values()))
    first_rep = next(iter(raw_counts_by_exp))
    all_units = sorted(raw_counts_by_exp[first_rep].keys())

    rows = []
    for cond in conditions:
        # Pool reps for each timepoint within this condition
        pooled = {}
        for tp in (time_a, time_b):
            pooled[tp] = {}
            for unit in all_units:
                pooled[tp][unit] = sum(
                    raw_counts_by_exp.get(rep, {}).get(unit, 0.0)
                    for rep in raw_counts_by_exp
                    if rep_to_condition.get(rep) == cond and rep_to_timepoint.get(rep) == tp
                )

        total_a = sum(pooled[time_a].values())
        total_b = sum(pooled[time_b].values())
        if total_a == 0 or total_b == 0:
            continue

        for unit in all_units:
            count_a = int(round(pooled[time_a][unit]))
            count_b = int(round(pooled[time_b][unit]))
            not_a = int(round(total_a)) - count_a
            not_b = int(round(total_b)) - count_b

            table = np.array([[count_a, not_a], [count_b, not_b]])
            try:
                odds_ratio, p_val = stats.fisher_exact(table, alternative="two-sided")
            except ValueError:
                odds_ratio, p_val = np.nan, 1.0

            rows.append({
                "condition": cond,
                "unit": unit,
                f"{time_a}_count": count_a,
                f"{time_a}_total": int(round(total_a)),
                f"{time_b}_count": count_b,
                f"{time_b}_total": int(round(total_b)),
                "odds_ratio": odds_ratio,
                "p_value": p_val,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per condition
    dfs = []
    for cond in conditions:
        mask = df["condition"] == cond
        sub = df.loc[mask].copy()
        sub["p_adj"] = bh_fdr(sub["p_value"].values)
        dfs.append(sub)
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["condition", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 4: Per-timepoint between-condition (Fisher's exact test)
# ---------------------------------------------------------------------------
def per_timepoint_fisher_occupancy(
    raw_counts_by_exp: dict,
    rep_to_condition: dict,
    rep_to_timepoint: dict,
) -> pd.DataFrame:
    """
    For each timepoint, pool reps per condition and run Fisher's exact test
    on each codon/AA.

    NOTE: pooling 2 biological replicates is pseudoreplication. P-values
    should be interpreted cautiously.

    Parameters
    ----------
    raw_counts_by_exp : dict
        {experiment: {unit: raw_read_sum}}
    rep_to_condition : dict
        {replicate: "control" or "BWM"}
    rep_to_timepoint : dict
        {replicate: "day_0", "day_5", or "day_10"}

    Returns
    -------
    pd.DataFrame with columns:
        timepoint, unit, {cond_a}_count, {cond_a}_total,
        {cond_b}_count, {cond_b}_total, odds_ratio, p_value, p_adj
    """
    conditions = sorted(set(rep_to_condition.values()))
    timepoints = sorted(set(rep_to_timepoint.values()))
    first_rep = next(iter(raw_counts_by_exp))
    all_units = sorted(raw_counts_by_exp[first_rep].keys())

    rows = []
    for tp in timepoints:
        # Pool reps per condition at this timepoint
        pooled_by_cond = {}
        for cond in conditions:
            pooled_by_cond[cond] = {}
            for unit in all_units:
                pooled_by_cond[cond][unit] = sum(
                    raw_counts_by_exp.get(rep, {}).get(unit, 0.0)
                    for rep in raw_counts_by_exp
                    if rep_to_condition.get(rep) == cond and rep_to_timepoint.get(rep) == tp
                )

        totals = {cond: sum(pooled_by_cond[cond].values()) for cond in conditions}
        if any(t == 0 for t in totals.values()):
            continue

        for unit in all_units:
            counts_list = []
            for cond in conditions:
                c = int(round(pooled_by_cond[cond][unit]))
                t = int(round(totals[cond]))
                counts_list.append([c, t - c])

            table = np.array(counts_list)
            try:
                odds_ratio, p_val = stats.fisher_exact(table, alternative="two-sided")
            except ValueError:
                odds_ratio, p_val = np.nan, 1.0

            row = {
                "timepoint": tp,
                "unit": unit,
                "odds_ratio": odds_ratio,
                "p_value": p_val,
            }
            for cond in conditions:
                row[f"{cond}_count"] = int(round(pooled_by_cond[cond][unit]))
                row[f"{cond}_total"] = int(round(totals[cond]))
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per timepoint
    dfs = []
    for tp in timepoints:
        mask = df["timepoint"] == tp
        sub = df.loc[mask].copy()
        sub["p_adj"] = bh_fdr(sub["p_value"].values)
        dfs.append(sub)
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["timepoint", "p_adj"])
