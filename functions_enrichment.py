"""
Statistical enrichment tests for amino acid composition at ribosome stall sites.

Three analyses:
  1. Within-condition enrichment (binomial test vs genome background)
  2. Between-condition overall (Wilcoxon rank-sum, n=6 vs n=6)
  3. Between-condition per-timepoint (Fisher's exact test)

Each E/P/A site is tested INDEPENDENTLY — never accumulated across sites.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# BH-FDR correction
# ---------------------------------------------------------------------------
def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    adjusted = p * n / ranks
    # enforce monotonicity (largest rank first)
    adjusted_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted[order] = adjusted_sorted
    return np.clip(adjusted, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Analysis 1: Within-condition enrichment (Binomial test)
# ---------------------------------------------------------------------------
def within_condition_enrichment(
    replicate_counts: dict,
    bg_freq: pd.Series,
    rep_to_condition: dict,
) -> pd.DataFrame:
    """
    For each condition, pool stall sites across replicates and test whether
    each amino acid at each E/P/A site is enriched or depleted vs background.

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series(AA->count), "P": Series(AA->count), "A": Series(AA->count)}}
    bg_freq : pd.Series
        Background amino acid frequencies (sums to 1), indexed by single-letter AA.
    rep_to_condition : dict
        {replicate: "control" or "BWM"}

    Returns
    -------
    pd.DataFrame with columns:
        condition, site, amino_acid, stall_count, total_n, stall_freq,
        bg_freq, log2_enrichment, p_value, p_adj
    """
    conditions = sorted(set(rep_to_condition.values()))
    rows = []

    for condition in conditions:
        cond_reps = [r for r, c in rep_to_condition.items() if c == condition]

        for site in ("E", "P", "A"):
            # Pool counts across all replicates in this condition
            pooled = None
            for rep in cond_reps:
                if rep not in replicate_counts:
                    continue
                counts = replicate_counts[rep][site]
                if pooled is None:
                    pooled = counts.copy()
                else:
                    pooled = pooled.add(counts, fill_value=0)

            if pooled is None:
                continue

            total_n = int(pooled.sum())
            if total_n == 0:
                continue

            for aa in pooled.index:
                k = int(pooled[aa])
                p_bg = float(bg_freq.get(aa, 1e-6))
                freq = k / total_n
                log2_enrich = np.log2(freq / p_bg) if freq > 0 and p_bg > 0 else 0.0

                result = stats.binomtest(k, total_n, p_bg, alternative="two-sided")
                rows.append({
                    "condition": condition,
                    "site": site,
                    "amino_acid": aa,
                    "stall_count": k,
                    "total_n": total_n,
                    "stall_freq": freq,
                    "bg_freq": p_bg,
                    "log2_enrichment": log2_enrich,
                    "p_value": result.pvalue,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per condition
    dfs = []
    for condition in conditions:
        mask = df["condition"] == condition
        sub = df.loc[mask].copy()
        sub["p_adj"] = bh_fdr(sub["p_value"].values)
        dfs.append(sub)
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["condition", "site", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 2: Between-condition overall (Wilcoxon rank-sum)
# ---------------------------------------------------------------------------
def between_condition_wilcoxon(
    replicate_counts: dict,
    rep_to_condition: dict,
) -> pd.DataFrame:
    """
    For each AA at each E/P/A site, compare per-replicate stall frequencies
    between conditions using Wilcoxon rank-sum (Mann-Whitney U).

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series, "P": Series, "A": Series}}
    rep_to_condition : dict
        {replicate: "control" or "BWM"}

    Returns
    -------
    pd.DataFrame with columns:
        site, amino_acid, median_control, median_BWM, log2_FC, U_stat, p_value, p_adj
    """
    conditions = sorted(set(rep_to_condition.values()))
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, got {conditions}")
    cond_a, cond_b = conditions  # alphabetical: BWM, control

    # Compute per-replicate frequencies
    rep_freqs = {}
    for rep, site_counts in replicate_counts.items():
        rep_freqs[rep] = {}
        for site in ("E", "P", "A"):
            counts = site_counts[site]
            total = counts.sum()
            rep_freqs[rep][site] = counts / total if total > 0 else counts * 0.0

    # Get amino acid list from first replicate
    first_rep = next(iter(replicate_counts))
    aa_list = list(replicate_counts[first_rep]["E"].index)

    rows = []
    for site in ("E", "P", "A"):
        for aa in aa_list:
            freqs_a = [rep_freqs[r][site][aa] for r in rep_freqs if rep_to_condition.get(r) == cond_a]
            freqs_b = [rep_freqs[r][site][aa] for r in rep_freqs if rep_to_condition.get(r) == cond_b]

            freqs_a = np.array(freqs_a, dtype=float)
            freqs_b = np.array(freqs_b, dtype=float)

            med_a = float(np.median(freqs_a))
            med_b = float(np.median(freqs_b))

            # log2 fold change: cond_b / cond_a  (control / BWM if alphabetical)
            # Use BWM / control for biological interpretation
            if med_a > 0 and med_b > 0:
                log2_fc = np.log2(med_a / med_b)  # BWM / control
            else:
                log2_fc = 0.0

            if len(freqs_a) >= 2 and len(freqs_b) >= 2:
                try:
                    u_stat, p_val = stats.mannwhitneyu(freqs_a, freqs_b, alternative="two-sided")
                except ValueError:
                    u_stat, p_val = np.nan, 1.0
            else:
                u_stat, p_val = np.nan, 1.0

            rows.append({
                "site": site,
                "amino_acid": aa,
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
    return df.sort_values(["site", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 3: Per-timepoint between-condition (Fisher's exact test)
# ---------------------------------------------------------------------------
def per_timepoint_fisher(
    replicate_counts: dict,
    rep_to_condition: dict,
    rep_to_timepoint: dict,
) -> pd.DataFrame:
    """
    For each timepoint, pool 2 reps per condition and run Fisher's exact test
    on each AA at each E/P/A site.

    NOTE: pooling 2 biological replicates is pseudoreplication. P-values should
    be interpreted cautiously.

    Parameters
    ----------
    replicate_counts : dict
        {replicate: {"E": Series, "P": Series, "A": Series}}
    rep_to_condition : dict
        {replicate: "control" or "BWM"}
    rep_to_timepoint : dict
        {replicate: "day_0", "day_5", or "day_10"}

    Returns
    -------
    pd.DataFrame with columns:
        timepoint, site, amino_acid, control_count, control_total,
        BWM_count, BWM_total, odds_ratio, p_value, p_adj
    """
    conditions = sorted(set(rep_to_condition.values()))
    timepoints = sorted(set(rep_to_timepoint.values()))

    first_rep = next(iter(replicate_counts))
    aa_list = list(replicate_counts[first_rep]["E"].index)

    rows = []
    for tp in timepoints:
        for site in ("E", "P", "A"):
            # Pool counts per condition at this timepoint
            pooled_by_cond = {}
            for cond in conditions:
                pooled = None
                for rep in replicate_counts:
                    if rep_to_condition.get(rep) != cond or rep_to_timepoint.get(rep) != tp:
                        continue
                    counts = replicate_counts[rep][site]
                    if pooled is None:
                        pooled = counts.copy()
                    else:
                        pooled = pooled.add(counts, fill_value=0)
                pooled_by_cond[cond] = pooled

            # Check both conditions have data
            if any(v is None for v in pooled_by_cond.values()):
                continue

            for aa in aa_list:
                # 2x2 table: [[cond_a_AA, cond_a_notAA], [cond_b_AA, cond_b_notAA]]
                counts_list = []
                totals = {}
                for cond in conditions:
                    aa_count = int(pooled_by_cond[cond].get(aa, 0))
                    total = int(pooled_by_cond[cond].sum())
                    not_aa = total - aa_count
                    counts_list.append([aa_count, not_aa])
                    totals[cond] = total

                table = np.array(counts_list)
                try:
                    odds_ratio, p_val = stats.fisher_exact(table, alternative="two-sided")
                except ValueError:
                    odds_ratio, p_val = np.nan, 1.0

                row = {
                    "timepoint": tp,
                    "site": site,
                    "amino_acid": aa,
                    "odds_ratio": odds_ratio,
                    "p_value": p_val,
                }
                for cond in conditions:
                    row[f"{cond}_count"] = int(pooled_by_cond[cond].get(aa, 0))
                    row[f"{cond}_total"] = totals[cond]
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
    return df.sort_values(["timepoint", "site", "p_adj"])
