#!/usr/bin/env python3
"""cross_tp_summary_checker.py — Reproduce the two cross-timepoint summary tables.

For any analysis CSV that runs the same Fisher test at multiple timepoints with a
(site, feature) cell layout, this script reproduces the two summary tables that
appear under "Cross-timepoint summary (two ranking tables)" in the Olive report:

    1. Cross-timepoint direction concordance
         Cells whose per-timepoint `log2_OR` values all share the same sign.
         Sorted by |mean log2OR| desc.
    2. Direction-flip cells across timepoints
         Cells with at least one sign change across the per-timepoint values.
         Sorted by max single-cell |log2OR| desc; truncated to --top-flip rows
         with a footer count of the omitted lower-magnitude flip cells.

Currently covers `_MANUAL_REVIEW.md` entries:

    #9   per_timepoint_fisher_aa.csv      (feature column: amino_acid)
    #10  per_timepoint_fisher_codon.csv   (feature column: codon)

The script is feature-agnostic — any future CSV with columns
`site, timepoint, <feature>, odds_ratio, p_adj, BWM_count, control_count`
will work without code changes.

Usage:
    python scripts/cross_tp_summary_checker.py <path-to-csv>
        [--top-flip 15] [--rare-k-threshold 100]
        [--iid-amp-threshold 1e-10] [--sig-threshold 0.05]

Output prints to stdout; nothing is written to disk. Compare each block against
the matching `Cross-timepoint summary` section of the corresponding `.qmd`.

Copy-paste blocks (only the entries in `shell_scripts/run_dylan_table_checker.sh`
that actually invoke this script — numbering matches `_MANUAL_REVIEW.md` /
`_OLIVE_PLAN.md`). Run from the repo root; literal relative paths work in
both bash and PowerShell:

    # ============================================================
    # 9b. per_timepoint_fisher_aa -- cross-timepoint summary tables
    # ============================================================
    echo "---- 9b. per_timepoint_fisher_aa cross-timepoint summary ----"
    python3 scripts/cross_tp_summary_checker.py "results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_aa.csv"

    # ============================================================
    # 10b. per_timepoint_fisher_codon -- cross-timepoint summary tables
    # ============================================================
    echo "---- 10b. per_timepoint_fisher_codon cross-timepoint summary ----"
    python3 scripts/cross_tp_summary_checker.py "results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_codon.csv"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

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


def _tp_sort_key(tp: str) -> float:
    """Extract the numeric suffix from a timepoint label ('day_10' -> 10)."""
    m = re.search(r"-?\d+(?:\.\d+)?", str(tp))
    return float(m.group()) if m else float("inf")


def _fmt_signed(x: float, prec: int = 2) -> str:
    if pd.isna(x):
        return "nan"
    return f"{x:+.{prec}f}"


def _fmt_p(x: float) -> str:
    if pd.isna(x):
        return "nan"
    return f"{x: .2e}".strip()


def _concordance_flag(n_sig: int, any_rare: bool, any_iid_amp: bool,
                      rare_label: str) -> str:
    """All applicable concordance flags, semicolon-joined.

    Significance label first (multi-tp-coherent / single-tp-driven, or
    nothing when n_sig == 0), then caveats (iid-amp, rare-aa / rare-codon).
    Differs from the qmd, which keeps only the first match by precedence;
    here we surface every applicable flag so caveats aren't masked when a
    cell is also significant.
    """
    flags: list[str] = []
    if n_sig >= 2:
        flags.append("multi-tp-coherent")
    elif n_sig == 1:
        flags.append("single-tp-driven")
    if any_iid_amp:
        flags.append("iid-amp")
    if any_rare:
        flags.append(rare_label)
    return "; ".join(flags)


def _flip_flag(any_rare: bool, any_iid_amp: bool, rare_label: str) -> str:
    """All applicable flip-table flags, semicolon-joined.

    Flip-table cells have no significance label; only caveats apply. The
    qmd shows iid-amp OR rare via precedence; here we surface both when
    both are true.
    """
    flags: list[str] = []
    if any_iid_amp:
        flags.append("iid-amp")
    if any_rare:
        flags.append(rare_label)
    return "; ".join(flags)


# -----------------------------------------------------------------------------
# Per-cell aggregation
# -----------------------------------------------------------------------------

def _aggregate_cells(df: pd.DataFrame, feat: str, rare_k: int,
                     iid_amp_thresh: float, sig_thresh: float) -> pd.DataFrame:
    df = df.copy()
    # Takes the log2 of the odds ratio, replacing any zero values with NaN to avoid -inf. 
    df["log2_OR"] = np.log2(df["odds_ratio"].replace(0, np.nan))
    # Drop rows where log2_OR is NaN
    df = df.dropna(subset=["log2_OR"])

    # Sort timepoints in natural numeric order (Check if this is sorted correctly)
    timepoints = sorted(df["timepoint"].unique(), key=_tp_sort_key)
    
    # Number of timepoints
    n_tp = len(timepoints)

    rows = []
    # Group by site and feature (AA or codon)
    for (site, feat_val), g in df.groupby(["site", feat], sort=False):

        # Reset the index to timepoint for easy lookup
        g_by_tp = g.set_index("timepoint")

        # Extract the log2_OR values for each timepoint, using NaN for any missing timepoints.
        per_tp_log2 = [g_by_tp["log2_OR"].get(tp, np.nan) for tp in timepoints]
        # Extract the p_adj values for each timepoint, using NaN for any missing timepoints.
        per_tp_padj = [g_by_tp["p_adj"].get(tp, np.nan) for tp in timepoints]

        # Extract the BWM_count and control_count for each timepoint, using NaN for any missing timepoints.
        per_tp_bwmk = [g_by_tp["BWM_count"].get(tp, np.nan) for tp in timepoints]
        per_tp_ctlk = [g_by_tp["control_count"].get(tp, np.nan) for tp in timepoints]

        # Only count valid (non-nan) timepoints toward concordance / sign logic.
        valid_log2 = [v for v in per_tp_log2 if not pd.isna(v)]

        if len(valid_log2) != n_tp:
            # Incomplete cell — record but mark as such; concordance/flip undefined.
            same_sign = False
        else:
            # signs is a set of either {+1}, {-1}, or {+1, -1} depending on the directions of the log2_OR values;
            # if 0 is present, it would be its own sign and break concordance, so we exclude it from the set and check separately.
            signs = {np.sign(v) for v in valid_log2 if v != 0}
            # Returns True if all share the same sign
            same_sign = len(signs) <= 1 and 0 not in signs

        # Number of signficant timepoints (p_adj < sig_thresh); only count non-nan p_adj values.
        n_sig = int(sum((p < sig_thresh) for p in per_tp_padj if not pd.isna(p)))
        # Checks if any Rare (Flag as True or False)
        any_rare = any(
            (k_b < rare_k) or (k_c < rare_k)
            for k_b, k_c in zip(per_tp_bwmk, per_tp_ctlk)
            if not (pd.isna(k_b) or pd.isna(k_c))
        )
        # Checks for any iid-amp timepoints among the non-nan p_adj values. (Flag as True or False)
        any_iid_amp = any(
            (p < iid_amp_thresh) for p in per_tp_padj if not pd.isna(p)
        )

        # Identify which timepoints' sig rows are positive
        sig_pos_tps = [
            tp for tp, lg, p in zip(timepoints, per_tp_log2, per_tp_padj)
            if not pd.isna(lg) and not pd.isna(p) and p < sig_thresh and lg > 0
        ]
        # Identify which timepoints' sig rows are negative
        sig_neg_tps = [
            tp for tp, lg, p in zip(timepoints, per_tp_log2, per_tp_padj)
            if not pd.isna(lg) and not pd.isna(p) and p < sig_thresh and lg < 0
        ]
        # If there is at least one significant positive and at least one significant negative timepoint, then sig_flip is True; otherwise False.
        sig_flip = bool(sig_pos_tps) and bool(sig_neg_tps)

        rows.append({
            "site": site,
            feat: feat_val,
            "cell": f"{site}:{feat_val}",
            "per_tp_log2OR": per_tp_log2,
            "per_tp_p_adj": per_tp_padj,
            "per_tp_BWM_k": per_tp_bwmk,
            "per_tp_ctrl_k": per_tp_ctlk,
            "mean_log2OR": float(np.nanmean(per_tp_log2)),
            "min_log2OR": float(np.nanmin(per_tp_log2)),
            "max_log2OR": float(np.nanmax(per_tp_log2)),
            "max_abs_log2OR": float(np.nanmax(np.abs(per_tp_log2))),
            "min_p_adj": float(np.nanmin(per_tp_padj)) if any(not pd.isna(p) for p in per_tp_padj) else np.nan,
            "n_sig": n_sig,
            "n_valid_tp": len(valid_log2),
            "any_rare": any_rare,
            "any_iid_amp": any_iid_amp,
            "same_sign": same_sign,
            "sig_pos_tps": sig_pos_tps,
            "sig_neg_tps": sig_neg_tps,
            "sig_flip": sig_flip,
        })

    return pd.DataFrame(rows), timepoints


# -----------------------------------------------------------------------------
# Table printers
# -----------------------------------------------------------------------------

def _print_concordance(cells: pd.DataFrame, timepoints: list[str],
                       rare_label: str) -> None:
    # Must have same_sign == True and a valid log2OR value at every timepoint (n_valid_tp == len(timepoints))
    concord = cells[cells["same_sign"] & (cells["n_valid_tp"] == len(timepoints))].copy()

    # Takes abs of log2OR mean for sorting
    concord["abs_mean"] = concord["mean_log2OR"].abs()
    concord = concord.sort_values("abs_mean", ascending=False)

    # Split into enriched and depleted for printing purposes
    n_enriched = int((concord["mean_log2OR"] > 0).sum())
    n_depleted = int((concord["mean_log2OR"] < 0).sum())

    print()
    print(f"### Cross-timepoint direction concordance "
          f"({len(concord)} of {len(cells)} cells share the same OR direction in all "
          f"{len(timepoints)} timepoints; {n_enriched} enriched, {n_depleted} depleted)")
    if concord.empty:
        print("  (no concordant cells)")
        return

    tp_label = ", ".join(timepoints)
    header_cols = ["cell", "mean log2OR",
                   f"per-tp log2OR ({tp_label})",
                   f"#sig (p_adj<thresh)", "min p_adj", "flag"]

    # For each subset (enriched vs depleted), print a markdown table. The flag column is determined 
    # by the _concordance_flag function, which prioritizes multi/single-tp significance over rare/iid-amp status.
    for direction_label, sub in (
        ("Enriched (mean log2_OR > 0)", concord[concord["mean_log2OR"] > 0]),
        ("Depleted (mean log2_OR < 0)", concord[concord["mean_log2OR"] < 0]),
    ):
        print()
        print(f"#### {direction_label}  ({len(sub)} cells)")
        if sub.empty:
            print("  (no cells in this direction)")
            continue
        print()
        print("  | " + " | ".join(header_cols) + " |")
        print("  | " + " | ".join("---" for _ in header_cols) + " |")

        for _, row in sub.iterrows():
            # Joins the list as a string
            per_tp = ", ".join(_fmt_signed(v) for v in row["per_tp_log2OR"])
            flag = _concordance_flag(
                n_sig=row["n_sig"],
                any_rare=row["any_rare"],
                any_iid_amp=row["any_iid_amp"],
                rare_label=rare_label,
            )
            print("  | " + " | ".join([
                row["cell"],
                _fmt_signed(row["mean_log2OR"], prec=3),
                per_tp,
                f"{row['n_sig']}/{len(timepoints)}",
                _fmt_p(row["min_p_adj"]),
                flag,
            ]) + " |")


def _print_flip(cells: pd.DataFrame, timepoints: list[str],
                rare_label: str, top_flip: int, sig_thresh: float) -> None:
    
    # Filters for rows where same_sign is False
    #  and n_valid_tp equals the total number of timepoints
    flip = cells[(~cells["same_sign"]) & (cells["n_valid_tp"] == len(timepoints))].copy()
    # Sort by number of FDR-significant timepoints desc (primary), then by
    # max |log2OR| desc (tiebreak). Cells whose direction-flip lands on
    # significant timepoints rank above cells that flip only at point
    # estimates.
    flip = flip.sort_values(["n_sig", "max_abs_log2OR"], ascending=[False, False])

    n_total = len(flip)
    # Counts how many of the flip cells have the sig_flip flag set to True
    n_sig_flip = int(flip["sig_flip"].sum())
    shown = flip.head(top_flip)

    print()
    print(f"### Direction-flip cells across timepoints "
          f"({n_total} of {len(cells)} cells show at least one sign change; "
          f"{n_sig_flip} have the flip register at p_adj<{sig_thresh} on both "
          f"opposite-sign rows; sorted by #sig desc, then |log2OR| desc)")

    if flip.empty:
        print("  (no flip cells)")
        return

    tp_label = ", ".join(timepoints)
    header_cols = ["cell", "#sig", "log2OR range",
                   f"per-tp log2OR ({tp_label}; * = p_adj<{sig_thresh})",
                   "sig-flip", "flag"]
    print()
    print("  | " + " | ".join(header_cols) + " |")
    print("  | " + " | ".join("---" for _ in header_cols) + " |")

    # For each row in flip cells
    for _, row in shown.iterrows():
        per_tp_marked = []
        # lg and p are pairs of log2OR and p_adj values for each timepoint;
        # we mark the log2OR with a star if its corresponding p_adj is below the significance threshold.
        for lg, p in zip(row["per_tp_log2OR"], row["per_tp_p_adj"]):
            star = "*" if (not pd.isna(p) and p < sig_thresh) else ""
            per_tp_marked.append(f"{_fmt_signed(lg)}{star}")
        # Join as 1 string for printing
        per_tp_str = ", ".join(per_tp_marked)

        # log2OR range string for printing
        rng = f"[{_fmt_signed(row['min_log2OR'], prec=3)}, {_fmt_signed(row['max_log2OR'], prec=3)}]"

        # sig-flip string for printing: "yes (tp1 vs tp2 both p_adj<thresh)" or "no"
        if row["sig_flip"]:
            pair = f"{row['sig_pos_tps'][0]} vs {row['sig_neg_tps'][0]} both p_adj<{sig_thresh}"
            sig_flip_str = f"yes ({pair})"
        else:
            sig_flip_str = "no"

        # Returns applicable flags
        flag = _flip_flag(
            any_rare=row["any_rare"],
            any_iid_amp=row["any_iid_amp"],
            rare_label=rare_label,
        )
        # Print the row
        print("  | " + " | ".join([
            row["cell"],
            f"{row['n_sig']}/{len(timepoints)}",
            rng,
            per_tp_str,
            sig_flip_str,
            flag,
        ]) + " |")

    omitted = n_total - len(shown)
    if omitted > 0:
        # Cutoff is the lexicographic (n_sig, max_abs_log2OR) of the last shown row.
        last = shown.iloc[-1]
        print()
        print(f"  ({omitted} more flip cells below the cutoff "
              f"(#sig, |log2OR|) = ({last['n_sig']}/{len(timepoints)}, "
              f"{last['max_abs_log2OR']:.3f}))")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="path to the analysis_stats CSV")
    parser.add_argument("--top-flip", type=int, default=15,
                        help="max flip-table rows to print (default 15)")
    parser.add_argument("--rare-k-threshold", type=int, default=100,
                        help="any timepoint with BWM_count or control_count below "
                             "this threshold flags the cell as rare-aa / rare-codon "
                             "(default 100, matching the qmd glossary rule)")
    parser.add_argument("--iid-amp-threshold", type=float, default=1e-10,
                        help="any timepoint with p_adj below this threshold flags "
                             "the cell as iid-amp (default 1e-10)")
    parser.add_argument("--sig-threshold", type=float, default=0.05,
                        help="significance threshold for #sig counts and sig-flip "
                             "detection (default 0.05)")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
        return 2

    # Import the csv and check for the required columns. We allow extra columns but require at least
    df = pd.read_csv(args.csv)
    required = {"site", "timepoint", "odds_ratio", "p_adj",
                "BWM_count", "control_count"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing required columns: {sorted(missing)}", file=sys.stderr)
        return 2

    # Determine the feature column (amino_acid vs codon) and set the rare label accordingly.
    feat = _feature_col(df)
    rare_label = "rare-aa" if feat == "amino_acid" else "rare-codon"

    # Print Message Block
    print("=" * 70)
    print(f"# {args.csv}")
    print(f"# feature: {feat}    rows: {len(df)}")
    print(f"# thresholds: sig p_adj<{args.sig_threshold}, "
          f"iid-amp p_adj<{args.iid_amp_threshold}, "
          f"rare-k <{args.rare_k_threshold}")
    print("=" * 70)

    # Timepoint sort-order check: mirrors the `sorted(...)` call in
    # _aggregate_cells so you can eyeball the chronological order before
    # trusting any downstream per-tp aggregation.
    sorted_tps = sorted(df["timepoint"].unique(), key=_tp_sort_key)
    print()
    print("### timepoint sort-order check")
    print(f"  sorted (numeric) : {sorted_tps}")

    # NaN guard audit #1: rows where odds_ratio == 0. These produce -inf
    # under log2 and are silently dropped in _aggregate_cells. Report
    # counts here so the reader knows what was excluded before per-cell
    # aggregation runs.
    zero_or = df[df["odds_ratio"] == 0]
    print()
    print("### NaN guard audit -- odds_ratio == 0 rows (dropped before aggregation)")
    print(f"  {len(zero_or)} of {len(df)} rows have odds_ratio == 0.")
    if not zero_or.empty:
        breakdown = zero_or.groupby(["timepoint", "site"]).size().to_dict()
        print(f"  per (timepoint, site) dropped counts: {breakdown}")

    # Main event Function
    cells, timepoints = _aggregate_cells(
        df, feat,
        rare_k=args.rare_k_threshold,
        iid_amp_thresh=args.iid_amp_threshold,
        sig_thresh=args.sig_threshold,
    )
    print(f"\n# {len(cells)} (site, {feat}) cells x {len(timepoints)} timepoints "
          f"({', '.join(timepoints)})")

    # NaN guard audit #2: (site, feature) cells that lack a row at every
    # timepoint. These survive aggregation but are excluded from both
    # summary tables via the n_valid_tp == len(timepoints) filter.
    incomplete = cells[cells["n_valid_tp"] != len(timepoints)]
    print()
    print("### NaN guard audit -- incomplete cells (excluded from summary tables)")
    print(f"  {len(incomplete)} of {len(cells)} cells have <{len(timepoints)} "
          f"non-NaN timepoints (typically because the (site, {feat}) pair is "
          f"absent at one or more timepoints, or its odds_ratio was 0).")
    if not incomplete.empty:
        coverage = incomplete["n_valid_tp"].value_counts().sort_index().to_dict()
        print(f"  n_valid_tp distribution among incomplete cells: {coverage}")


    # Printing the two summary tables:
    _print_concordance(cells, timepoints, rare_label)
    _print_flip(cells, timepoints, rare_label,
                top_flip=args.top_flip, sig_thresh=args.sig_threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
