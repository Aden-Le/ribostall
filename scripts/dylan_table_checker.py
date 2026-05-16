#!/usr/bin/env python3
"""dylan_table_checker.py — Reproduce Dylan's Top-hits selection per CSV family.

Use to verify, for any analysis_stats CSV, which rows Dylan's selection rule
picks for the Top-hits sub-tables in the Olive report. The output is grouped
the same way Olive splits its sub-tables (per group x direction-of-effect)
so you can eyeball the script's picks against the report.

Each family function corresponds to one (or a range of) entry numbers in
results/stall_sites/enrichment/olive_reports/_MANUAL_REVIEW.md:

    select_wilcoxon      -> #1-#8   between_condition_wilcoxon_*, between_timepoint_wilcoxon_*
    select_fisher_aa     -> #9      per_timepoint_fisher_aa.csv
    select_fisher_codon  -> #10     per_timepoint_fisher_codon.csv
    select_tfwc          -> #11-#16 timepoint_fisher_within_condition_*
    select_binom_aa      -> #17     within_condition_binomial_aa.csv (k threshold TBD)
    select_binom_codon   -> #18     within_condition_binomial_codon.csv

Usage:
    python scripts/dylan_table_checker.py <path-to-csv> --family <name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _feature_col(df: pd.DataFrame) -> str:
    for c in ("amino_acid", "codon"):
        if c in df.columns:
            return c
    raise ValueError("Neither 'amino_acid' nor 'codon' column present.")


def _print_group(title: str, df: pd.DataFrame, cols: list[str]) -> None:
    print()
    print(f"### {title}")
    if df.empty:
        print("  (no rows clear the filter)")
        return
    fmt = df.loc[:, cols].copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == "f":
            fmt[c] = fmt[c].map(
                lambda v: f"{v: .4f}" if not pd.isna(v) and abs(v) >= 1e-3 else (
                    f"{v: .3e}" if not pd.isna(v) else "nan"
                )
            )
    print(fmt.to_string(index=False))


# -----------------------------------------------------------------------------
# #1-#8: Wilcoxon (between_condition + between_timepoint)
#   Rule: Top 5 per direction (sign of log2_FC), ranked by raw p_value asc;
#         ties broken by |log2_FC| desc. No filter.
#
#   Low-count flag: a row is flagged `low-count` when min(median_arm_A,
#   median_arm_B) < threshold (default 0.005). At AA resolution every
#   feature is well above 0.005, so the audit returns 0 rows; at codon
#   resolution it picks up the rare-codon set (CGG, GTA, ATA, AGT, GGG,
#   TTA and similar) referenced in Dylan's `low-count-rare-codon-
#   instability` caveat.
# -----------------------------------------------------------------------------

def _wilcoxon_median_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Return the pair of `median_*` columns (BWM/control or day_X/day_Y)."""
    cols = [c for c in df.columns if c.startswith("median_")]
    if len(cols) != 2:
        raise ValueError(f"expected 2 median_* columns, got {cols}")
    return cols[0], cols[1]


def _wilcoxon_with_lowcount(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, str, str]:
    """Return (df_with_columns, arm_A_col, arm_B_col).

    Adds `min_median` and `low_count` columns based on the rule
    `min(median_arm_A, median_arm_B) < threshold`.
    """
    arm_a, arm_b = _wilcoxon_median_cols(df)
    df = df.copy()
    df["min_median"] = df[[arm_a, arm_b]].min(axis=1)
    df["low_count"] = df["min_median"] < threshold
    return df, arm_a, arm_b


def _print_lowcount_audit(df: pd.DataFrame, feat: str, arm_a: str, arm_b: str,
                          threshold: float) -> None:
    print()
    print(f"### low-count audit  (rule: min({arm_a}, {arm_b}) < {threshold})")
    flagged = df[df["low_count"]].copy()
    print(f"  {len(flagged)} of {len(df)} rows fall below the threshold.")
    if flagged.empty:
        return
    flagged = flagged.sort_values(["site", "min_median"])
    print("  per-site counts:", flagged.groupby("site").size().to_dict())
    print("  rows flagged low-count (sorted by site, then min_median asc):")
    cols = ["site", feat, arm_a, arm_b, "min_median", "log2_FC", "p_value", "p_adj"]
    _print_group("low-count rows (full list)", flagged, cols)


def select_wilcoxon(df: pd.DataFrame, top_n: int = 5,
                    low_count_threshold: float = 0.005) -> None:
    feat = _feature_col(df)
    df, arm_a, arm_b = _wilcoxon_with_lowcount(df, low_count_threshold)
    df["abs_effect"] = df["log2_FC"].abs()

    _print_lowcount_audit(df, feat, arm_a, arm_b, low_count_threshold)

    for site, g in df.groupby("site", sort=False):
        for direction, sub in (
            ("Enriched (log2_FC > 0)", g[g["log2_FC"] > 0]),
            ("Depleted (log2_FC < 0)", g[g["log2_FC"] < 0]),
        ):
            picked = sub.sort_values(
                by=["p_value", "abs_effect"], ascending=[True, False]
            ).head(top_n)
            _print_group(
                f"site {site} -- {direction}",
                picked,
                [feat, "log2_FC", "p_value", "p_adj", "min_median", "low_count"],
            )


# -----------------------------------------------------------------------------
# #9: per_timepoint_fisher (AA)
#   Rule: Rows with p_adj < 0.10; up to 5 per direction (sign of log2_OR);
#         ranked by |log2_OR| desc. No tiebreak needed (Fisher p's distinct).
# -----------------------------------------------------------------------------

def select_fisher_aa(df: pd.DataFrame, top_n: int = 5, p_thresh: float = 0.10) -> None:
    feat = _feature_col(df)
    df = df.copy()
    df["log2_OR"] = np.log2(df["odds_ratio"].replace(0, np.nan))
    df["abs_effect"] = df["log2_OR"].abs()
    df = df.dropna(subset=["log2_OR"])
    df = df[df["p_adj"] < p_thresh]
    for (tp, site), g in df.groupby(["timepoint", "site"], sort=False):
        for direction, sub in (
            ("Enriched (log2_OR > 0)", g[g["log2_OR"] > 0]),
            ("Depleted (log2_OR < 0)", g[g["log2_OR"] < 0]),
        ):
            picked = sub.sort_values(by="abs_effect", ascending=False).head(top_n)
            _print_group(
                f"{tp}, site {site} -- {direction}",
                picked,
                [feat, "log2_OR", "p_value", "p_adj"],
            )


# -----------------------------------------------------------------------------
# #10: per_timepoint_fisher (codon)
#   Rule: same as #9 + additional filter BWM_count + control_count >= 50.
# -----------------------------------------------------------------------------

def select_fisher_codon(df: pd.DataFrame, top_n: int = 5,
                        p_thresh: float = 0.10, k_min: int = 50) -> None:
    feat = _feature_col(df)
    df = df.copy()
    df["log2_OR"] = np.log2(df["odds_ratio"].replace(0, np.nan))
    df["abs_effect"] = df["log2_OR"].abs()
    df["combined_k"] = df["BWM_count"] + df["control_count"]
    df = df.dropna(subset=["log2_OR"])
    df = df[(df["p_adj"] < p_thresh) & (df["combined_k"] >= k_min)]
    for (tp, site), g in df.groupby(["timepoint", "site"], sort=False):
        for direction, sub in (
            ("Enriched (log2_OR > 0)", g[g["log2_OR"] > 0]),
            ("Depleted (log2_OR < 0)", g[g["log2_OR"] < 0]),
        ):
            picked = sub.sort_values(by="abs_effect", ascending=False).head(top_n)
            _print_group(
                f"{tp}, site {site} -- {direction}",
                picked,
                [feat, "log2_OR", "combined_k", "p_value", "p_adj"],
            )


# -----------------------------------------------------------------------------
# #11-#16: timepoint_fisher_within_condition (AA + codon)
#   Rule: same as #9 (p_adj<0.10 + top-5 by |log2_OR|) per (condition, site);
#         FALLBACK per cell: if <10 candidates clear FDR<0.10, drop the FDR
#         cutoff and rank by raw p_value asc (|log2_OR| as tiebreaker).
# -----------------------------------------------------------------------------

def select_tfwc(df: pd.DataFrame, top_n: int = 5,
                p_thresh: float = 0.10, min_candidates: int = 10) -> None:
    feat = _feature_col(df)
    df = df.copy()
    df["log2_OR"] = np.log2(df["odds_ratio"].replace(0, np.nan))
    df["abs_effect"] = df["log2_OR"].abs()
    df = df.dropna(subset=["log2_OR"])
    for (cond, site), g in df.groupby(["condition", "site"], sort=False):
        sig = g[g["p_adj"] < p_thresh]
        if len(sig) < min_candidates:
            print()
            print(f"# (condition={cond}, site={site}): only {len(sig)} candidates "
                  f"at p_adj<{p_thresh} -> fallback to raw-p ranking, no FDR cutoff")
            pool, sort_cols, asc = g, ["p_value", "abs_effect"], [True, False]
        else:
            pool, sort_cols, asc = sig, ["abs_effect"], [False]
        for direction, sub in (
            ("Enriched (log2_OR > 0)", pool[pool["log2_OR"] > 0]),
            ("Depleted (log2_OR < 0)", pool[pool["log2_OR"] < 0]),
        ):
            picked = sub.sort_values(by=sort_cols, ascending=asc).head(top_n)
            _print_group(
                f"{cond}, site {site} -- {direction}",
                picked,
                [feat, "log2_OR", "p_value", "p_adj"],
            )


# -----------------------------------------------------------------------------
# #17: within_condition_binomial (AA)
#   Rule: TBD against Dylan's interpretation. AA `k` threshold may differ
#         from codon's k>=50. For now, apply the codon rule and warn.
# -----------------------------------------------------------------------------

def select_binom_aa(df: pd.DataFrame, top_n: int = 5,
                    k_min: int = 50, p_thresh: float = 0.05) -> None:
    print()
    print("# NOTE: rule for AA binomial is TBD against Dylan's interpretation.")
    print(f"# Provisionally applying codon rule: observed_count >= {k_min} "
          f"AND p_adj < {p_thresh}.")
    _select_binom_impl(df, top_n=top_n, k_min=k_min, p_thresh=p_thresh)


# -----------------------------------------------------------------------------
# #18: within_condition_binomial (codon)
#   Rule: Rows with observed_count>=50 AND p_adj<0.05; up to 5 per direction
#         (sign of log2_enrichment); ranked by |log2_enrichment| desc.
# -----------------------------------------------------------------------------

def select_binom_codon(df: pd.DataFrame, top_n: int = 5,
                       k_min: int = 50, p_thresh: float = 0.05) -> None:
    _select_binom_impl(df, top_n=top_n, k_min=k_min, p_thresh=p_thresh)


def _select_binom_impl(df: pd.DataFrame, top_n: int, k_min: int, p_thresh: float) -> None:
    feat = _feature_col(df)
    df = df.copy()
    df["abs_effect"] = df["log2_enrichment"].abs()
    df = df[(df["observed_count"] >= k_min) & (df["p_adj"] < p_thresh)]
    for (group, site), g in df.groupby(["group", "site"], sort=False):
        for direction, sub in (
            ("Enriched (log2_enrichment > 0)", g[g["log2_enrichment"] > 0]),
            ("Depleted (log2_enrichment < 0)", g[g["log2_enrichment"] < 0]),
        ):
            picked = sub.sort_values(by="abs_effect", ascending=False).head(top_n)
            _print_group(
                f"{group}, site {site} -- {direction}",
                picked,
                [feat, "log2_enrichment", "observed_count", "p_value", "p_adj"],
            )


# -----------------------------------------------------------------------------
# Family-flag dispatch
# -----------------------------------------------------------------------------

FAMILY_MAP: dict[str, Callable[[pd.DataFrame], None]] = {
    "wilcoxon":     select_wilcoxon,      # _MANUAL_REVIEW entries #1-#8
    "fisher_aa":    select_fisher_aa,     # _MANUAL_REVIEW entry  #9
    "fisher_codon": select_fisher_codon,  # _MANUAL_REVIEW entry  #10
    "tfwc":         select_tfwc,          # _MANUAL_REVIEW entries #11-#16
    "binom_aa":     select_binom_aa,      # _MANUAL_REVIEW entry  #17
    "binom_codon":  select_binom_codon,   # _MANUAL_REVIEW entry  #18
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="path to the analysis_stats CSV")
    parser.add_argument("--family", required=True, choices=list(FAMILY_MAP),
                        help="selection-rule family (see header docstring)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="max rows per direction (default 5)")
    parser.add_argument("--low-count-threshold", type=float, default=0.005,
                        help="wilcoxon family only: a row is flagged `low-count` "
                             "if min(median_arm_A, median_arm_B) is strictly less "
                             "than this value (default 0.005, matching Dylan's "
                             "rare-codon `low-count` rule)")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)
    print("=" * 70)
    print(f"# {args.csv}")
    print(f"# family: {args.family}    rows: {len(df)}")
    print("=" * 70)

    if args.family == "wilcoxon":
        FAMILY_MAP[args.family](df, top_n=args.top_n,
                                low_count_threshold=args.low_count_threshold)
    else:
        FAMILY_MAP[args.family](df, top_n=args.top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
