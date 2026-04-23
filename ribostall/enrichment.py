"""
Statistical enrichment tests for amino acid composition at ribosome stall sites.

Three analyses:
  1. Within-condition enrichment (binomial test vs genome background)
  2. Between-condition overall (Wilcoxon rank-sum, n=6 vs n=6)
  3. Between-condition per-timepoint (Fisher's exact test)

Each E/P/A site is tested INDEPENDENTLY — never accumulated across sites.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        condition, group, site, amino_acid, stall_count, total_n, stall_freq,
        bg_freq, log2_enrichment, weighted_log2_enrichment, p_value, p_adj
    """
    groups = sorted(set(rep_to_group[r] for r in replicate_counts if r in rep_to_group))
    rows = []

    for group in groups:
        # Background frequencies for this group
        bg_freq = bg_freq_per_group[group]
        # Replicates in this group | Ex: "control_day_0" -> ["ctrl_rep1", "ctrl_rep2"]
        group_reps = [r for r in replicate_counts if rep_to_group.get(r) == group]
        # Condition for this group (e.g. "control" or "BWM")
        condition = rep_to_condition.get(group_reps[0], "") if group_reps else ""

        for site in ("E", "P", "A"):
            # Pool counts across replicates in this group
            pooled = None
            for rep in group_reps:
                counts = replicate_counts[rep][site]
                if pooled is None:
                    pooled = counts.copy()
                else:
                    pooled = pooled.add(counts, fill_value=0)

            if pooled is None:
                continue
            
            # Total n is the total count of all amino acids at this site in this group (not background)
            total_n = int(pooled.sum())
            if total_n == 0:
                continue

            for aa in pooled.index:
                # k is the sum of target aa
                k = int(pooled[aa])
                # p_bg is the background frequency of this aa in this group (with small pseudocount to avoid zero)
                p_bg = float(bg_freq.get(aa, 1e-6))
                # freq is the observed frequency of this aa at this site in this group
                freq = k / total_n
                log2_enrich = np.log2(freq / p_bg) if freq > 0 and p_bg > 0 else 0.0
                weighted_log2 = freq * log2_enrich

                result = stats.binomtest(k, total_n, p_bg, alternative="two-sided")

                # For output, we want the raw p-value. Multiple testing correction will be done later per group.
                rows.append({
                    "condition": condition,
                    "group": group,
                    "site": site,
                    feature_col: aa,
                    "stall_count": k,
                    "total_n": total_n,
                    "stall_freq": freq,
                    "bg_freq": p_bg,
                    "log2_enrichment": log2_enrich,
                    "weighted_log2_enrichment": weighted_log2,
                    "p_value": result.pvalue,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH-FDR correction per group
    dfs = []
    for group in groups:
        mask = df["group"] == group
        sub = df.loc[mask].copy()
        sub["p_adj"] = bh_fdr(sub["p_value"].values)
        dfs.append(sub)
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["group", "site", "p_adj"])


# ---------------------------------------------------------------------------
# Analysis 2: Between-condition overall (Wilcoxon rank-sum)
# ---------------------------------------------------------------------------
def between_condition_wilcoxon(
    replicate_counts: dict,
    rep_to_condition: dict,
    *,
    feature_col: str = "amino_acid",
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
    # Gets the 2 conditions in alphabetical order for consistent output column naming (e.g. median_BWM, median_control)
    cond_a, cond_b = conditions  # alphabetical: BWM, control

    # Compute per-replicate frequencies
    rep_freqs = {}
    for rep, site_counts in replicate_counts.items():
        rep_freqs[rep] = {}
        for site in ("E", "P", "A"):
            # Gets each AA counts for each site
            counts = site_counts[site]
            # Gets the total counts of all AAs at this site in this replicate
            total = counts.sum()
            # Converts them into frequencies for each AA at this site in this replicate
            rep_freqs[rep][site] = counts / total if total > 0 else counts * 0.0

    # Get amino acid list from first replicate
    first_rep = next(iter(replicate_counts))
    aa_list = list(replicate_counts[first_rep]["E"].index)

    rows = []
    for site in ("E", "P", "A"):
        for aa in aa_list:
            # Get frequencies for this AA at this site across replicates in each condition
            freqs_a = [rep_freqs[r][site][aa] for r in rep_freqs if rep_to_condition.get(r) == cond_a]
            freqs_b = [rep_freqs[r][site][aa] for r in rep_freqs if rep_to_condition.get(r) == cond_b]
            # Convert to numpy arrays for median and stats functions
            freqs_a = np.array(freqs_a, dtype=float)
            freqs_b = np.array(freqs_b, dtype=float)
            # Get the median frequency for this AA at this site in each condition for fold change calculation
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
                feature_col: aa,
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
    *,
    feature_col: str = "amino_acid",
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
    # All the conditions and timepoints present in the data, sorted for consistent output
    conditions = sorted(set(rep_to_condition.values()))
    timepoints = sorted(set(rep_to_timepoint.values()))

    # Gets the AA list using the first replicate (assumes all replicates have the same AA index)
    first_rep = next(iter(replicate_counts))
    aa_list = list(replicate_counts[first_rep]["E"].index)

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
                    # The counts of all AAs at this site in this replicate
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
            
            # For each amino acid, build the 2x2 contingency table and run Fisher's exact test
            for aa in aa_list:
                # 2x2 table: [[cond_a_AA, cond_a_notAA], [cond_b_AA, cond_b_notAA]]
                counts_list = []
                totals = {}
                for cond in conditions:
                    # The count of this AA at this site in this condition (pooled across replicates)
                    aa_count = int(pooled_by_cond[cond].get(aa, 0))
                    # The total count of all AAs at this site in this condition (pooled across replicates)
                    total = int(pooled_by_cond[cond].sum())
                    # The count of all other AAs at this site in this condition is total - aa_count
                    not_aa = total - aa_count
                    # appends the counts for this condition to the contingency table list
                    counts_list.append([aa_count, not_aa])
                    totals[cond] = total

                # Table Looks like this: [[BWM_AA, BWM_notAA], [control_AA, control_notAA]]
                table = np.array(counts_list)
                try:
                    odds_ratio, p_val = stats.fisher_exact(table, alternative="two-sided")
                except ValueError:
                    odds_ratio, p_val = np.nan, 1.0

                row = {
                    "timepoint": tp,
                    "site": site,
                    feature_col: aa,
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
