"""
Statistical tests for global codon and amino acid occupancy analysis.

Four analyses:
  1. Within-condition enrichment (binomial test vs transcriptome background)
  2. Between-condition overall (Wilcoxon rank-sum, BWM vs Control, n=6 vs n=6)
  3. Between-timepoint (Wilcoxon pooled across conditions + Fisher's within condition)
  4. Per-timepoint between-condition (Fisher's exact test)

Each test is run independently for codon-level and amino acid-level occupancy.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from ribostall.enrichment import bh_fdr


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
    # Calculates background frequency for each unit (codon or AA) as count in transcriptome divided by total background count
    bg_freq = {u: c / total_bg for u, c in transcriptome_counts.items()} if total_bg > 0 else {}

    # Either Codons or AAs sorted alphabetically
    all_units = sorted(transcriptome_counts.keys())
    rows = []
    
    # For each group (e.g. "control_day_0"), pools the raw counts across replicates in that group and compares to background frequencies 
    # using a binomial test
    for grp, reps in sorted(groups.items()):
        # Splits the group name into condition and time point
        # For example, "control_day_0" would be split into condition="control" and timepoint="day_0"
        parts = grp.split("_", 1)
        condition = parts[0]
        timepoint = parts[1] if len(parts) > 1 else grp

        # Pool raw counts across replicates in this group
        pooled = {}
        # For each unit (codon or AA), sums the raw counts across all replicates in the group to get a pooled count for that unit in the group
        for unit in all_units:
            pooled[unit] = sum(raw_counts_by_exp.get(rep, {}).get(unit, 0.0) for rep in reps)

        # Total number of reads in the group
        total_n = sum(pooled.values())
        if total_n == 0:
            continue

        for unit in all_units:
            # Rounds the pooled count to the nearest integer to get the observed count for this unit in the group
            k = int(round(pooled[unit]))
            # Get the background frequency for this unit
            p_bg = bg_freq.get(unit, 1e-6)
            # Gets the frequency of this unit in the group by dividing the observed count by the total number of reads in the group
            freq = k / total_n if total_n > 0 else 0.0

            # Calculate log2 enrichment as log2(observed frequency / background frequency), with a pseudocount to avoid log of zero
            log2_enrich = np.log2(freq / p_bg) if freq > 0 and p_bg > 0 else 0.0
            weighted_log2 = freq * log2_enrich
            # k is the observed count of the unit in the group, total_n is the total number of reads in the group,
            # p_bg is the background frequency of the unit. The binomial test checks if the observed 
            # count k is significantly different from what would be expected under a binomial distribution
            # with parameters total_n and p_bg.
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
        # Rates_a and rates_b are arrays of normalized occupancy rates for this unit across replicates in condition A
        # and condition B, respectively
        # For example, the normalized occupancy rates for codon "AAA" across all replicates in the
        # BWM condition would be collected into rates_a,
        rates_a = np.array([rates_by_exp[r][unit] for r in rates_by_exp
                            if rep_to_condition.get(r) == cond_a], dtype=float)
        rates_b = np.array([rates_by_exp[r][unit] for r in rates_by_exp
                            if rep_to_condition.get(r) == cond_b], dtype=float)

        # The medians are taken for the log2 FC calculation
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
    time_a: str = "day_10",
    time_b: str = "day_0",
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
        # All replicates with timepoint day_10
        rates_a = np.array([rates_by_exp[r][unit] for r in rates_by_exp
                            if rep_to_timepoint.get(r) == time_a], dtype=float)
        # All replicates with timepoint day_0
        rates_b = np.array([rates_by_exp[r][unit] for r in rates_by_exp
                            if rep_to_timepoint.get(r) == time_b], dtype=float)

        med_a = float(np.median(rates_a)) if len(rates_a) > 0 else 0.0
        med_b = float(np.median(rates_b)) if len(rates_b) > 0 else 0.0

        # Log fold change of time_a vs time_b (e.g. day_10 vs day_0)
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
    time_a: str = "day_10",
    time_b: str = "day_0",
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
    # BWM and Control conditions sorted alphabetically
    conditions = sorted(set(rep_to_condition.values()))
    
    # The units (codons or AAs)
    first_rep = next(iter(raw_counts_by_exp))
    all_units = sorted(raw_counts_by_exp[first_rep].keys())

    rows = []
    # For each condition, BWM & Control
    for cond in conditions:
        pooled = {}
        # For timepoint day_10 and day_0
        for tp in (time_a, time_b):
            pooled[tp] = {}
            for unit in all_units:
                # Sum the raw counts for this unit across all replicates that belong to this condition and timepoint
                # Ex all raw counts for BWM day_0 replicates would be summed to get the pooled count for this unit in the BWM day_0 group
                pooled[tp][unit] = sum(
                    raw_counts_by_exp.get(rep, {}).get(unit, 0.0)
                    for rep in raw_counts_by_exp
                    if rep_to_condition.get(rep) == cond and rep_to_timepoint.get(rep) == tp
                )
        
        # Pooled values for day_10 and day_0 for this condition
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
    # Conditions of interest sorted alphabetically (e.g. BWM, Control)
    conditions = sorted(set(rep_to_condition.values()))
    # Timepoints of interest sorted alphabetically (e.g. day_0, day_10, day_5)
    timepoints = sorted(set(rep_to_timepoint.values()))
    first_rep = next(iter(raw_counts_by_exp))
    all_units = sorted(raw_counts_by_exp[first_rep].keys())

    rows = []
    # Ex: For day_0
    for tp in timepoints:
        pooled_by_cond = {}
        # For control
        for cond in conditions:
            # Empty dict for control
            pooled_by_cond[cond] = {}
            # For all amino acids
            for unit in all_units:
                # Sum the raw counts for each amino acid across both replicates
                pooled_by_cond[cond][unit] = sum(
                    raw_counts_by_exp.get(rep, {}).get(unit, 0.0)
                    for rep in raw_counts_by_exp
                    if rep_to_condition.get(rep) == cond and rep_to_timepoint.get(rep) == tp
                )

        # Total number of reads for both conditions at this timepoint
        totals = {cond: sum(pooled_by_cond[cond].values()) for cond in conditions}
        if any(t == 0 for t in totals.values()):
            continue
        
        # For each Amino Acid
        for unit in all_units:
            counts_list = []
            # BWM first and then control (BWM on top)
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


# ---------------------------------------------------------------------------
# Shared helpers (used by global_codon_occ.py and global_codon_occ_stats.py)
# ---------------------------------------------------------------------------

def iter_trimmed_codons(seq: str, trim_start_codons: int, trim_stop_codons: int):
    """Yield (codon_str, cds_nt_idx) for the trimmed CDS sequence."""
    n = len(seq)
    start_nt = trim_start_codons * 3
    end_nt = n - (trim_stop_codons * 3)
    if start_nt >= end_nt:
        return
    last_full = end_nt - ((end_nt - start_nt) % 3)
    for i in range(start_nt, last_full, 3):
        codon = seq[i:i + 3]
        if len(codon) == 3 and "N" not in codon.upper():
            yield codon, i


def iter_trimmed_site_counts(
    cds_seq: str,
    cov,
    trim_start_codons: int,
    trim_stop_codons: int,
    site_shift: int,
):
    """Yield (site_codon, count) for each P-site codon in the trimmed CDS.

    The mental model
    ----------------
    Coverage here is P-site offset: the value at ``cov[X:X+3]`` is the number
    of reads from ribosomes whose P-site sits on the codon at CDS nt position
    X. For each such ribosome, three codons matter biologically — the codon
    in its E-site (1 codon upstream), P-site (on the codon), and A-site (1
    codon downstream). All three share the SAME read count (the ribosome is
    one ribosome); only the codon identity differs.

    So this function:
      * walks over every P-site position ``cds_nt_idx`` in the trimmed CDS,
      * reads the ribosome count at that position (always
        ``cov[cds_nt_idx:cds_nt_idx+3]`` — the P-site window),
      * picks which codon to report using ``site_shift``:
            -3  -> E-site codon (1 codon upstream of the P-site)
             0  -> P-site codon (the P-site itself)
            +3  -> A-site codon (1 codon downstream of the P-site)

    Example: at ``cds_nt_idx = 300`` with ``site_shift = -3`` (E-site),
    we return the codon at ``cds_seq[297:300]`` together with the read
    count from ``cov[300:303]``. The caller calls the function three
    times (once per site) to accumulate E/P/A totals.

    Skips positions where:
      * the P-site coverage window runs past the end of ``cov``,
      * the shifted site codon would fall outside the CDS (e.g. E-site at
        the very first P-site codon has no codon upstream),
      * the site codon contains an N (ambiguous base).
    """
    # CDS and coverage are expected to be the same length (one-to-one
    # nt mapping). If they disagree something upstream went wrong — warn
    # once per call and let the per-position bounds checks below silently
    # skip the positions that can't be read.
    n_seq = len(cds_seq)
    n_cov = len(cov)
    if n_seq != n_cov:
        logging.warning(
            "iter_trimmed_site_counts: CDS length (%d) != coverage length (%d); ",
            n_seq, n_cov,
        )

    # Convert codon-level trim into nt-level trim. trim_start_codons=10
    # means "skip the first 10 codons" = skip the first 30 nt, so we start
    # iterating P-site positions at nt index 30.
    start_nt = trim_start_codons * 3
    end_nt = n_seq - (trim_stop_codons * 3)
    
    if start_nt >= end_nt:
        return
    
    # Guard against a CDS length that isn't a multiple of 3 by rounding
    # down to the last full codon boundary inside the trimmed range.
    last_full = end_nt - ((end_nt - start_nt) % 3)

    for cds_nt_idx in range(start_nt, last_full, 3):

        # Need a full 3-nt P-site window in the coverage array.
        if cds_nt_idx + 3 > n_cov:
            continue

        # Compute where the site codon lives in the CDS.
        site_start = cds_nt_idx + site_shift
        site_end = site_start + 3

        # The site codon has to be fully inside the CDS.
        if site_start < 0 or site_end > n_seq:
            continue

        site_codon = cds_seq[site_start:site_end]
        # Defensive: guard against length-2 slices at the end of the CDS, and skip codons with ambiguous bases.
        if len(site_codon) != 3 or "N" in site_codon.upper():
            continue

        # Count is the ribosome's P-site signal — same value no matter what
        count = float(cov[cds_nt_idx:cds_nt_idx + 3].sum())
        yield site_codon, count


def parse_groups(groups_arg: str) -> dict:
    """Parse CLI group string into dict: {group_name: [rep1, rep2, ...]}."""
    groups = {}
    for block in groups_arg.split(";"):
        name, reps = block.split(":")
        groups[name] = reps.split(",")
    return groups


def aggregate_to_aa(codon_dict: dict) -> dict:
    """Aggregate a {codon: value} dict to {AA: value}, skipping stop codons."""
    from collections import defaultdict
    from ribostall.amino_acids import CODON2AA
    aa_dict = defaultdict(float)
    for codon, val in codon_dict.items():
        aa = CODON2AA.get(codon.upper())
        if aa and aa != "*":
            aa_dict[aa] += val
    return dict(aa_dict)
