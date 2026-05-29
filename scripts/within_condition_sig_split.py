#!/usr/bin/env python3
"""within_condition_sig_split.py — Split a within-condition timepoint-contrast
CSV into three significance sections (both / BWM-only / control-only).

For a `timepoint_fisher_within_condition_<contrast>_<feat>.csv` file, each
(site, condition, feature) row carries its own Fisher's-exact `odds_ratio`
for the later-vs-earlier timepoint contrast (e.g. day_10 vs day_0) *within*
one condition. BWM and control each get their own `odds_ratio` -> `log2_OR`.

This script pairs the BWM and control rows for every (site, feature) cell and
sorts the cells into three sections by which condition(s) the contrast clears
FDR in:

    1. Significant in both      (BWM sig AND control sig)
    2. Significant in BWM        (BWM sig, control not)
    3. Significant in control    (control sig, BWM not)

Cells significant in neither condition are dropped. Each section prints a
paired table with one row per (site, feature):

    | Site | <Feature> | BWM log2_OR | control log2_OR | Effect change | Flag |

`log2_OR` sign convention follows the contrast direction encoded in the
column names: positive = enriched at the LATER timepoint relative to the
earlier one; negative = depleted at the later timepoint. `Effect change` is
BWM `log2_OR` minus control `log2_OR` (the BWM-vs-control divergence in the
later-vs-earlier shift); rows within each section are grouped by site in
A / P / E order, then sorted by `Effect change` descending. The `Flag` column
carries `low-count (BWM, C)` / `low-count (BWM)` / `low-count (C)` naming which
arm(s) fall below the per-condition count threshold (`C` = control).

The script is feature-agnostic (amino_acid or codon) and contrast-agnostic
(it discovers the two `day_<n>_count` columns and orders them numerically),
so the same code works for every file in the
`timepoint_fisher_within_condition` family.

Usage:
    python scripts/within_condition_sig_split.py <path-to-csv>
        [--sig-threshold 0.05]
        [--rare-bwm-threshold 100] [--rare-control-threshold 200]

Output prints markdown to stdout; nothing is written to disk.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Biology-canonical ribosome-site order (A / P / E), overriding alphabetical.
SITE_ORDER = ["A", "P", "E"]

# One-letter -> three-letter amino-acid abbreviations (matches the abbreviation
# style already used in the within-condition interpretation .md Top-hits tables).
AA_THREE = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _feature_col(df: pd.DataFrame) -> str:
    for c in ("amino_acid", "codon"):
        if c in df.columns:
            return c
    raise ValueError("Neither 'amino_acid' nor 'codon' column present.")


def _timepoint_count_cols(df: pd.DataFrame) -> tuple[str, str, str, str]:
    """Discover the two `day_<n>_count` columns and order them earlier->later.

    Returns (earlier_count, later_count, earlier_label, later_label), e.g.
    ('day_0_count', 'day_10_count', 'day_0', 'day_10'). The numeric suffix
    drives the ordering, so this works for d10_vs_d0, d10_vs_d5, d5_vs_d0.
    """
    found: list[tuple[int, str, str]] = []
    for c in df.columns:
        m = re.fullmatch(r"day_(\d+)_count", c)
        if m:
            found.append((int(m.group(1)), c, f"day_{m.group(1)}"))
    if len(found) != 2:
        raise ValueError(
            f"Expected exactly two 'day_<n>_count' columns, found {len(found)}: "
            f"{[f[1] for f in found]}"
        )
    found.sort(key=lambda t: t[0])
    earlier, later = found[0], found[1]
    return earlier[1], later[1], earlier[2], later[2]


def _fmt_signed(x: float, prec: int = 3) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:+.{prec}f}"


def _feature_label(feat: str, val: str) -> str:
    """For amino acids, append the three-letter abbreviation: 'N (Asn)'.

    Codons (and anything without a known one-letter mapping) print bare.
    """
    if feat == "amino_acid" and val in AA_THREE:
        return f"{val} ({AA_THREE[val]})"
    return str(val)


def _site_sort_key(site: str) -> tuple[int, str]:
    """A -> 0, P -> 1, E -> 2; unknown sites sort after, alphabetically."""
    if site in SITE_ORDER:
        return (SITE_ORDER.index(site), "")
    return (len(SITE_ORDER), str(site))


# -----------------------------------------------------------------------------
# Per-cell aggregation
# -----------------------------------------------------------------------------

def _build_cells(df: pd.DataFrame, feat: str, earlier_count: str,
                 later_count: str, sig_thresh: float, rare_bwm: int,
                 rare_control: int) -> pd.DataFrame:
    """One row per (site, feature) pairing the BWM and control condition rows.

    Each condition contributes its own log2_OR, p_adj, significance flag, and
    rare-low-count flag (BWM and control carry different count thresholds
    because control is pooled over far more stalls).
    """
    df = df.copy()
    # log2 of the within-condition later-vs-earlier odds ratio; OR == 0 -> NaN
    # (avoids -inf). The Fisher OR is already directional, so log2 is symmetric.
    df["log2_OR"] = np.log2(df["odds_ratio"].replace(0, np.nan))

    rare_thresh = {"BWM": rare_bwm, "control": rare_control}

    rows = []
    for (site, feat_val), g in df.groupby(["site", feat], sort=False):
        by_cond = g.set_index("condition")

        rec: dict = {"site": site, feat: feat_val,
                     "cell": f"{site}:{feat_val}"}
        for cond in ("BWM", "control"):
            if cond in by_cond.index:
                r = by_cond.loc[cond]
                log2 = float(r["log2_OR"])
                p_adj = float(r["p_adj"])
                k_later = r[later_count]
                k_earlier = r[earlier_count]
                sig = (not pd.isna(p_adj)) and (p_adj < sig_thresh)
                rare = (min(k_later, k_earlier) < rare_thresh[cond])
            else:
                log2, p_adj, sig, rare = np.nan, np.nan, False, False
            rec[f"{cond}_log2"] = log2
            rec[f"{cond}_p_adj"] = p_adj
            rec[f"{cond}_sig"] = sig
            rec[f"{cond}_rare"] = rare

        rec["n_sig"] = int(rec["BWM_sig"]) + int(rec["control_sig"])
        rows.append(rec)

    return pd.DataFrame(rows)


def _section(cells: pd.DataFrame, which: str) -> pd.DataFrame:
    """Filter to a significance section and sort (site A/P/E, then change desc).

    Within each site, rows are ordered by `Effect change`
    descending — most BWM-positive divergence at the top, most control-positive
    at the bottom.
    """
    if which == "both":
        sub = cells[cells["BWM_sig"] & cells["control_sig"]].copy()
    elif which == "bwm":
        sub = cells[cells["BWM_sig"] & ~cells["control_sig"]].copy()
    elif which == "control":
        sub = cells[cells["control_sig"] & ~cells["BWM_sig"]].copy()
    else:
        raise ValueError(which)

    sub["change"] = sub["BWM_log2"] - sub["control_log2"]
    sub["site_key"] = sub["site"].map(_site_sort_key)
    return sub.sort_values(["site_key", "change"], ascending=[True, False])


# -----------------------------------------------------------------------------
# Table printer
# -----------------------------------------------------------------------------

# Short arm tokens for the low-count flag suffix: control abbreviates to `C`.
_ARM_TOKEN = {"BWM": "BWM", "control": "C"}


def _flag_cell(row: pd.Series, rare_label: str) -> str:
    arms = [_ARM_TOKEN[c] for c in ("BWM", "control") if row[f"{c}_rare"]]
    if not arms:
        return ""
    return f"{rare_label} ({', '.join(arms)})"


def _print_section(sub: pd.DataFrame, title: str, feat: str, feat_header: str,
                   rare_label: str, later_label: str, earlier_label: str) -> None:
    print()
    print(f"### {title} (n = {len(sub)} site x {feat_header.lower()} cells)")
    if sub.empty:
        print()
        print("_(no cells)_")
        return

    header = ["Site", feat_header,
              "BWM `log2_OR`", "control `log2_OR`",
              "Effect change", "Flag"]
    print()
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")

    for _, row in sub.iterrows():
        change = row["BWM_log2"] - row["control_log2"]
        print("| " + " | ".join([
            str(row["site"]),
            _feature_label(feat, row[feat]),
            _fmt_signed(row["BWM_log2"]),
            _fmt_signed(row["control_log2"]),
            _fmt_signed(change),
            _flag_cell(row, rare_label),
        ]) + " |")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="path to the analysis_stats CSV")
    parser.add_argument("--sig-threshold", type=float, default=0.05,
                        help="p_adj threshold for significance (default 0.05)")
    parser.add_argument("--rare-bwm-threshold", type=int, default=100,
                        help="BWM cell with either timepoint count below this "
                             "is flagged rare-low-count (default 100)")
    parser.add_argument("--rare-control-threshold", type=int, default=200,
                        help="control cell with either timepoint count below "
                             "this is flagged rare-low-count (default 200)")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)
    required = {"site", "condition", "odds_ratio", "p_adj"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing required columns: {sorted(missing)}",
              file=sys.stderr)
        return 2

    feat = _feature_col(df)
    feat_header = "Amino Acid" if feat == "amino_acid" else "Codon"
    rare_label = "low-count"
    earlier_count, later_count, earlier_label, later_label = _timepoint_count_cols(df)

    print("=" * 70)
    print(f"# {args.csv}")
    print(f"# feature: {feat}    contrast: {later_label} vs {earlier_label}    rows: {len(df)}")
    print(f"# sig p_adj<{args.sig_threshold}; rare BWM<{args.rare_bwm_threshold}, "
          f"control<{args.rare_control_threshold}")
    print("=" * 70)
    print()
    print(f"`log2_OR` is the within-condition Fisher effect for the "
          f"**{later_label} vs {earlier_label}** contrast: positive = enriched at "
          f"{later_label} relative to {earlier_label}, negative = depleted. Each "
          f"row pairs the BWM and control values for one site x "
          f"{feat_header.lower()} cell. `Effect change` is the "
          f"BWM `log2_OR` minus the control `log2_OR` — large magnitude marks "
          f"conditions whose {later_label}-vs-{earlier_label} trajectories "
          f"diverge. Cells significant in neither condition are omitted.")

    cells = _build_cells(
        df, feat, earlier_count, later_count,
        sig_thresh=args.sig_threshold,
        rare_bwm=args.rare_bwm_threshold,
        rare_control=args.rare_control_threshold,
    )

    _print_section(_section(cells, "both"),
                   "Significant in both conditions", feat, feat_header,
                   rare_label, later_label, earlier_label)
    _print_section(_section(cells, "bwm"),
                   "Significant in BWM only", feat, feat_header,
                   rare_label, later_label, earlier_label)
    _print_section(_section(cells, "control"),
                   "Significant in control only", feat, feat_header,
                   rare_label, later_label, earlier_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
