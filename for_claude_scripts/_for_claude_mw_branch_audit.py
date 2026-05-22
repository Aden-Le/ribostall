"""Audit: did scipy.stats.mannwhitneyu hit the asymptotic-with-ties branch
on a between_condition or between_timepoint Wilcoxon? Per (site, feature),
report tie count in the pooled per-arm sample, exact-branch p,
asymptotic-branch p, and whether they materially differ.

NOTE: Authored by Claude (file prefix `_for_claude_`). Not part of the
pipeline; not a top-level entry point. Intended as a re-runnable audit
for verifying that the `asymptotic-with-ties` caveat does not bite on
the Wilcoxon CSVs (AA + codon resolution; between_condition and the
three between_timepoint pairs).

Usage examples:

  # CSV 1 (between_condition AA, default):
  python for_claude_scripts/_for_claude_mw_branch_audit.py

  # CSV 2 (between_condition codon):
  python for_claude_scripts/_for_claude_mw_branch_audit.py --level codon

  # CSV 3 (between_timepoint d10_vs_d0 AA):
  python for_claude_scripts/_for_claude_mw_branch_audit.py --design between_timepoint \\
      --timepoints day10 day0

  # CSV 4 (between_timepoint d10_vs_d0 codon):
  python for_claude_scripts/_for_claude_mw_branch_audit.py --design between_timepoint \\
      --timepoints day10 day0 --level codon
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

SITES = ("E", "P", "A")

AAS = list("ACDEFGHIKLMNPQRSTVWY")
# 61 sense codons (all 64 minus the three stops TAA/TAG/TGA).
BASES = "ACGT"
CODONS = sorted(a + b + c for a in BASES for b in BASES for c in BASES
                if (a + b + c) not in ("TAA", "TAG", "TGA"))


def per_rep_freqs(df: pd.DataFrame, level: str) -> dict:
    """{replicate: {site: {feature: freq}}}; freq = count / total stalls in that rep."""
    if level == "aa":
        feats = AAS
        cols = (("E_aa", "E"), ("P_aa", "P"), ("A_aa", "A"))
    else:
        feats = CODONS
        cols = (("E_codon", "E"), ("P_codon", "P"), ("A_codon", "A"))
    out: dict = {}
    for rep, sub in df.groupby("replicate"):
        n = len(sub)
        out[rep] = {}
        for col, site in cols:
            counts = sub[col].value_counts()
            out[rep][site] = {f: counts.get(f, 0) / n for f in feats}
    return out


def split_reps(reps: list[str], design: str, timepoints: tuple[str, str] | None) -> tuple[list[str], list[str], tuple[str, str]]:
    """Return (arm_A_reps, arm_B_reps, (arm_A_label, arm_B_label))."""
    if design == "between_condition":
        a = [r for r in reps if r.startswith("BWM")]
        b = [r for r in reps if r.startswith("control")]
        return a, b, ("BWM", "control")
    # between_timepoint: split by timepoint token in the rep name.
    assert timepoints is not None and len(timepoints) == 2
    tp_a, tp_b = timepoints
    a = [r for r in reps if f"_{tp_a}_" in r]
    b = [r for r in reps if f"_{tp_b}_" in r]
    return a, b, (tp_a, tp_b)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stalls", default=None,
                   help="Path to per-stall CSV (defaults to stall_sites_aa.csv or _codon.csv based on --level).")
    p.add_argument("--agg", default=None,
                   help="Path to the aggregated Wilcoxon CSV under analysis_stats/ (defaults built from --design + --timepoints + --level).")
    p.add_argument("--level", choices=("aa", "codon"), default="aa")
    p.add_argument("--design", choices=("between_condition", "between_timepoint"), default="between_condition")
    p.add_argument("--timepoints", nargs=2, metavar=("TP_A", "TP_B"), default=None,
                   help="For --design between_timepoint, the two timepoint tokens to compare, e.g. day10 day0.")
    return p.parse_args()


def default_paths(args: argparse.Namespace) -> tuple[str, str]:
    stalls = args.stalls or f"results/stall_sites/enrichment/stall_sites_{args.level}.csv"
    if args.agg:
        agg = args.agg
    elif args.design == "between_condition":
        agg = f"results/stall_sites/enrichment/analysis_stats/between_condition_wilcoxon_{args.level}.csv"
    else:
        tp_a, tp_b = args.timepoints
        # Normalize timepoints to "d10" / "d5" / "d0" tokens used in CSV filenames.
        norm = lambda t: t.replace("day", "d")
        agg = f"results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_{norm(tp_a)}_vs_{norm(tp_b)}_{args.level}.csv"
    return stalls, agg


def main() -> int:
    args = parse_args()
    if args.design == "between_timepoint" and args.timepoints is None:
        print("--timepoints TP_A TP_B is required when --design between_timepoint", file=sys.stderr)
        return 2
    stalls_path, agg_path = default_paths(args)
    print(f"Audit: design={args.design} level={args.level} stalls={stalls_path} agg={agg_path}")

    feats = AAS if args.level == "aa" else CODONS
    feat_col_in_agg = "amino_acid" if args.level == "aa" else "codon"

    df = pd.read_csv(stalls_path)
    rep_freqs = per_rep_freqs(df, args.level)
    arm_a, arm_b, (lbl_a, lbl_b) = split_reps(list(rep_freqs), args.design, args.timepoints)
    print(f"  arm_A ({lbl_a}, n={len(arm_a)}): {arm_a}")
    print(f"  arm_B ({lbl_b}, n={len(arm_b)}): {arm_b}")
    assert len(arm_a) > 0 and len(arm_b) > 0, (arm_a, arm_b)

    rows = []
    for site in SITES:
        for feat in feats:
            a = np.array([rep_freqs[r][site][feat] for r in arm_a], dtype=float)
            b = np.array([rep_freqs[r][site][feat] for r in arm_b], dtype=float)
            pooled = np.concatenate([a, b])

            vc = Counter(pooled.tolist())
            n_tied_values = sum(1 for _v, k in vc.items() if k > 1)
            has_ties = any(k > 1 for k in vc.values())

            u_exact, p_exact = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact")
            u_asym, p_asym = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
            u_auto, p_auto = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")

            if np.isclose(p_auto, p_exact, rtol=0, atol=1e-12):
                auto_branch = "exact"
            elif np.isclose(p_auto, p_asym, rtol=0, atol=1e-12):
                auto_branch = "asymptotic"
            else:
                auto_branch = "other"

            rows.append({
                "site": site,
                "feature": feat,
                "has_ties": has_ties,
                "n_distinct_values_with_ties": n_tied_values,
                "p_exact": p_exact,
                "p_asym": p_asym,
                "p_auto": p_auto,
                "auto_branch": auto_branch,
                "abs_diff_exact_vs_asym": abs(p_exact - p_asym),
            })

    out = pd.DataFrame(rows)

    agg = pd.read_csv(agg_path).rename(columns={feat_col_in_agg: "feature"})
    merged = out.merge(agg[["site", "feature", "p_value"]], on=["site", "feature"], how="left")
    merged["abs_diff_pipeline_vs_recomputed_auto"] = (merged["p_value"] - merged["p_auto"]).abs()

    n_total = len(merged)
    print(f"\n=== Summary across {n_total} (site, feature) tests ===")
    print(f"Tests with ties in pooled {len(arm_a)+len(arm_b)}-element sample: {merged['has_ties'].sum()} / {n_total}")
    print(f"Auto branch counts: {merged['auto_branch'].value_counts().to_dict()}")
    print(f"Max |p_pipeline - p_auto_recomputed|: {merged['abs_diff_pipeline_vs_recomputed_auto'].max():.2e}")
    print(f"Max |p_exact - p_asym|: {merged['abs_diff_exact_vs_asym'].max():.4f}")
    print(f"Median |p_exact - p_asym|: {merged['abs_diff_exact_vs_asym'].median():.4f}")

    flip = merged[((merged["p_exact"] < 0.05) != (merged["p_asym"] < 0.05))]
    print(f"\n=== Tests where raw-p<0.05 status differs between branches: {len(flip)} ===")
    if len(flip):
        with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", "{:.4f}".format):
            print(flip[["site", "feature", "has_ties", "p_exact", "p_asym"]].to_string(index=False))

    print("\n=== Per-site BH-FDR comparison: exact-branch min p_adj vs asym-branch min p_adj ===")

    def bh_fdr(pvals: np.ndarray) -> np.ndarray:
        p = np.asarray(pvals, dtype=float)
        m = len(p)
        order = np.argsort(p)
        ranked = p[order] * m / (np.arange(m) + 1)
        for i in range(m - 2, -1, -1):
            ranked[i] = min(ranked[i], ranked[i + 1])
        adj = np.empty_like(ranked)
        adj[order] = np.clip(ranked, 0, 1)
        return adj

    for site in SITES:
        sub = merged[merged.site == site].sort_values("feature")
        padj_exact = bh_fdr(sub["p_exact"].to_numpy())
        padj_asym = bh_fdr(sub["p_asym"].to_numpy())
        print(f"  {site}: min p_adj(exact) = {padj_exact.min():.4f}   min p_adj(asym) = {padj_asym.min():.4f}   "
              f"n_hits_fdr_005(exact) = {(padj_exact<0.05).sum()}   n_hits_fdr_005(asym) = {(padj_asym<0.05).sum()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
