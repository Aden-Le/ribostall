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


def background_diff_row(
    stall_count_headline: int, stall_total_headline: int, bg_freq_headline: float,
    stall_count_other: int, stall_total_other: int, bg_freq_other: float,
) -> dict:
    """One background-aware between-condition row.

    Unlike ``fisher_row`` (which compares raw stall-site *shares* between two
    conditions), this compares each condition's enrichment OVER ITS OWN
    background. Per condition:

        expected_count = stall_total * bg_freq    # count if stalling tracked background
        enrichment     = stall_count / expected_count   # == Analysis-1 enrichment

    Effect size is the difference of the two log2 enrichments:

        delta_log2_enrichment = log2(enrichment_headline) - log2(enrichment_other)

    Test: under H0 (equal enrichment), conditional on
    ``combined_count = stall_count_headline + stall_count_other``,

        stall_count_headline | combined_count ~ Binomial(combined_count, null_share)
        null_share = expected_count_headline / (expected_count_headline + expected_count_other)

    This is the exact conditional test for two equal Poisson rates with
    background as the exposure/offset. The background frequencies are treated as
    known (they are estimated from millions of codons) and are taken as given:
    the caller is responsible for any pseudocount, mirroring ``binom_row`` whose
    ``p_bg`` is likewise caller-supplied (both adapters pass a pseudocounted,
    strictly-positive background, so ``expected_count`` is never zero here). When
    the two backgrounds are equal, ``null_share`` reduces to the raw stall-total
    split and the result converges to ``fisher_row``; the two diverge only when
    the transcriptome composition shifts between conditions.

    Returns ``expected_count_headline, expected_count_other, log2_enrich_headline,
    log2_enrich_other, delta_log2_enrichment, enrichment_ratio, null_share,
    observed_share, p_value``.
    """
    # This is what we expect to get if the stalling is random
    expected_count_headline = stall_total_headline * bg_freq_headline
    expected_count_other = stall_total_other * bg_freq_other

    # The log2 enrichment for each condition
    log2_enrich_headline = (
        np.log2(stall_count_headline / expected_count_headline)
        if stall_count_headline > 0 and expected_count_headline > 0 else 0.0
    )
    log2_enrich_other = (
        np.log2(stall_count_other / expected_count_other)
        if stall_count_other > 0 and expected_count_other > 0 else 0.0
    )
    # The difference in log2 enrichments
    delta_log2_enrichment = log2_enrich_headline - log2_enrich_other

    # Gets the total stalls of the two conditions
    combined_count = stall_count_headline + stall_count_other

    # Gets the expected total stalls from the  condition
    expected_total = expected_count_headline + expected_count_other
    if combined_count == 0 or expected_total == 0:
        return {
            "expected_count_headline": expected_count_headline,
            "expected_count_other": expected_count_other,
            "log2_enrich_headline": log2_enrich_headline,
            "log2_enrich_other": log2_enrich_other,
            "delta_log2_enrichment": 0.0,
            "enrichment_ratio": 2.0 ** delta_log2_enrichment,
            "null_share": np.nan,
            "observed_share": np.nan,
            "p_value": 1.0,
        }

    # The expected fraction of stalls that should be the headline
    null_share = expected_count_headline / expected_total
    # Out of the total stalls, what is the expected fraction for the headline
    # is the null_share
    p_value = stats.binomtest(
        stall_count_headline, combined_count, null_share, alternative="two-sided"
    ).pvalue
    return {
        "expected_count_headline": expected_count_headline,
        "expected_count_other": expected_count_other,
        "log2_enrich_headline": log2_enrich_headline,
        "log2_enrich_other": log2_enrich_other,
        "delta_log2_enrichment": delta_log2_enrichment,
        "enrichment_ratio": 2.0 ** delta_log2_enrichment,
        "null_share": null_share,
        "observed_share": stall_count_headline / combined_count,
        "p_value": p_value,
    }
