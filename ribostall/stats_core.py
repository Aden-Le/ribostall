"""
Shared statistical kernels for the enrichment and global-occupancy pipelines.

Both ``ribostall/enrichment.py`` (stall sites) and
``ribostall/global_occupancy.py`` (global occupancy) re-implemented the same
three tests — binomial vs background, Wilcoxon rank-sum, Fisher's exact — with
slightly different scaffolding. The per-row test math now lives here once; the
two modules are thin adapters that build their own column layouts around these
kernels.

Each kernel computes ONE row's worth of statistics and returns a plain dict of
generic keys. The adapters pull those values into their own explicit column
order (preserving the historical layout byte-for-byte).
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# BH-FDR correction
# ---------------------------------------------------------------------------
def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p.copy()
    return stats.false_discovery_control(p, method="bh")


def apply_bh_fdr(df, group_cols=None):
    """Add a ``p_adj`` column via BH-FDR, optionally within groups.

    - empty frame → returned unchanged.
    - ``group_cols`` falsy → one global correction over the whole ``p_value`` column.
    - else → ``groupby(group_cols, sort=False)`` and correct within each group,
      then concat (preserving first-appearance group order and within-group order).

    BH is order-independent within a family, so the per-row ``p_adj`` values
    depend only on group membership; the caller's final ``sort_values`` fully
    determines output row order.
    """
    import pandas as pd

    if df.empty:
        return df
    if not group_cols:
        df = df.copy()
        df["p_adj"] = bh_fdr(df["p_value"].values)
        return df
    dfs = []
    for _, sub in df.groupby(group_cols, sort=False):
        sub = sub.copy()
        sub["p_adj"] = bh_fdr(sub["p_value"].values)
        dfs.append(sub)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-row test kernels
# ---------------------------------------------------------------------------
def binom_row(k: int, n: int, p_bg: float) -> dict:
    """One binomial-vs-background row.

    Returns ``observed_count, total_n, observed_freq, bg_freq, log2_enrichment,
    weighted_log2_enrichment, p_value``.
    """
    freq = k / n if n > 0 else 0.0
    log2_enrich = np.log2(freq / p_bg) if freq > 0 and p_bg > 0 else 0.0
    weighted_log2 = freq * log2_enrich
    p_value = stats.binomtest(k, n, p_bg, alternative="two-sided").pvalue
    return {
        "observed_count": k,
        "total_n": n,
        "observed_freq": freq,
        "bg_freq": p_bg,
        "log2_enrichment": log2_enrich,
        "weighted_log2_enrichment": weighted_log2,
        "p_value": p_value,
    }


def wilcoxon_row(values_a, values_b) -> dict:
    """One Wilcoxon rank-sum (Mann-Whitney U) row.

    Returns ``median_a, median_b, log2_FC, U_stat, p_value``. Medians are 0.0
    for empty inputs; the test runs only when both arms have >= 2 values,
    otherwise ``(U_stat, p_value) = (nan, 1.0)``.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    med_a = float(np.median(values_a)) if len(values_a) > 0 else 0.0
    med_b = float(np.median(values_b)) if len(values_b) > 0 else 0.0

    if med_a > 0 and med_b > 0:
        log2_fc = np.log2(med_a / med_b)
    else:
        log2_fc = 0.0

    if len(values_a) >= 2 and len(values_b) >= 2:
        try:
            u_stat, p_val = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        except ValueError:
            u_stat, p_val = np.nan, 1.0
    else:
        u_stat, p_val = np.nan, 1.0

    return {
        "median_a": med_a,
        "median_b": med_b,
        "log2_FC": log2_fc,
        "U_stat": u_stat,
        "p_value": p_val,
    }


def fisher_row(count_a: int, total_a: int, count_b: int, total_b: int) -> dict:
    """One Fisher's exact (2x2) row.

    Table is ``[[count_a, total_a-count_a], [count_b, total_b-count_b]]``.
    Returns ``odds_ratio, p_value`` ((nan, 1.0) on ValueError).
    """
    table = np.array([[count_a, total_a - count_a], [count_b, total_b - count_b]])
    try:
        odds_ratio, p_val = stats.fisher_exact(table, alternative="two-sided")
    except ValueError:
        odds_ratio, p_val = np.nan, 1.0
    return {"odds_ratio": odds_ratio, "p_value": p_val}
