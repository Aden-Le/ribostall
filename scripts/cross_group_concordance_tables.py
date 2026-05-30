#!/usr/bin/env python3
"""cross_group_concordance_tables.py — Generate the three cross-group
concordance / discordance Top-10 tables for a within-condition-binomial CSV.

For a `within_condition_binomial_<feat>.csv` file, each (site, group, feature)
row carries a one-sample-binomial `log2_enrichment` (observed stall frequency
vs that group's transcriptome-window `bg_freq`) and its BH-FDR `p_adj`. There
are 6 groups (BWM and control x day_0 / day_5 / day_10) x 3 sites (A/P/E) x N
features, so every (site, feature) cell has six per-group `log2_enrichment`
values.

This script pools the six per-group rows for each (site, feature) cell and
sorts the cells into three Top-10 tables:

    1. Concordant enrichment   all 6 groups have log2_enrichment > 0
    2. Concordant depletion     all 6 groups have log2_enrichment < 0
    3. Discordant               at least one sign disagrees across the 6 groups

Each table is selected as the Top 10 cells by `#sig` descending (number of the
6 groups with `p_adj` < the significance threshold), then `min count`
descending (the smallest observed_count across the 6 groups — the weakest
group's data support) as the tiebreaker. The selected rows are then displayed
sorted by Site (A / P / E), then `#sig` desc, then `min count` desc.

Columns (amino-acid CSV; the amino-acid name is spelled out on every row):

    | Site | amino acid | log2_enrichment | min count | #sig | flag |

Columns (codon CSV — an `aa` column is inserted per the ribostall codon-table
convention):

    | Site | codon | aa | log2_enrichment | min count | #sig | flag |

The `log2_enrichment` cell lists the six per-group values on two lines (the
break renders in both HTML and PDF — see `_CELL_BREAK`): BWM day_0, day_5,
day_10 on the first line, then control day_0, day_5, day_10 on the second. The `flag` column aggregates the
report's existing caveat tokens across the six groups:

    iid-amp     any group's p_adj < the iid-amp threshold (default 1e-10)
    bg-tight    the feature's bg_freq > 0.05 in any group (tight binomial null)
    rare-aa /   one or more groups have observed_count below the rare-k
    rare-codon  threshold (default 100); the flag names the low-count groups
                (e.g. `rare-codon (BWM d0, ctrl d5)`), collapsing to `(all)`
                when every group is below threshold

The script is feature-agnostic (amino_acid or codon).

Usage:
    python scripts/cross_group_concordance_tables.py <path-to-csv>
        [--top-n 10] [--sig-threshold 0.05]
        [--iid-amp-threshold 1e-10] [--bg-tight-threshold 0.05]
        [--rare-k-threshold 100]

Output prints markdown to stdout; nothing is written to disk. Paste each block
into the matching `## Top hits` section of the corresponding `.qmd`.
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

# Short condition tokens for the rare-flag group list (`control` -> `ctrl` to
# keep the flag cell compact); mirrors the `control` -> `C` abbreviation in
# `within_condition_sig_split.py`'s low-count suffix.
COND_TOKEN = {"BWM": "BWM", "control": "ctrl"}

# Codon -> one-letter amino acid (sense codons; stops map to "*"). Inlined so
# the script stays self-contained, matching the other checker/generator
# scripts in scripts/.
CODON2AA = {
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "AAT": "N", "AAC": "N", "GAT": "D", "GAC": "D", "TGT": "C", "TGC": "C",
    "GAA": "E", "GAG": "E", "CAA": "Q", "CAG": "Q",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "CAT": "H", "CAC": "H", "ATT": "I", "ATC": "I", "ATA": "I",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "AAA": "K", "AAG": "K", "ATG": "M", "TTT": "F", "TTC": "F",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "TGG": "W",
    "TAT": "Y", "TAC": "Y", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TAA": "*", "TAG": "*", "TGA": "*",
}

# One-letter -> full amino-acid name (matches the "K (lysine)" expansion already
# used in the within_condition_binomial_aa.qmd Top-hits tables, including the
# "aspartate" / "glutamate" spellings).
AA_FULL = {
    "A": "alanine", "R": "arginine", "N": "asparagine", "D": "aspartate",
    "C": "cysteine", "Q": "glutamine", "E": "glutamate", "G": "glycine",
    "H": "histidine", "I": "isoleucine", "L": "leucine", "K": "lysine",
    "M": "methionine", "F": "phenylalanine", "P": "proline", "S": "serine",
    "T": "threonine", "W": "tryptophan", "Y": "tyrosine", "V": "valine",
}


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


def _site_sort_key(site: str) -> tuple[int, str]:
    """A -> 0, P -> 1, E -> 2; unknown sites sort after, alphabetically."""
    if site in SITE_ORDER:
        return (SITE_ORDER.index(site), "")
    return (len(SITE_ORDER), str(site))


def _fmt_signed(x: float, prec: int = 2) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:+.{prec}f}"


def _group_token(cond: str, tp: str) -> str:
    """Compact group label for the rare-flag list, e.g. 'BWM d0', 'ctrl d10'."""
    c = COND_TOKEN.get(cond, str(cond))
    m = re.search(r"\d+", str(tp))
    return f"{c} d{m.group()}" if m else f"{c} {tp}"


def _flag_cell(any_iid_amp: bool, bg_tight: bool, rare_groups: list[str],
               group_tokens: list[str], rare_label: str) -> str:
    """Caveat tokens in the report's canonical order, comma-joined.

    The rare flag names the low-count groups (the granularity the removed
    per-group sub-tables used to carry), following the `low-count (BWM, C)`
    pattern in `within_condition_sig_split.py`. To stay compact with 6 groups
    it collapses near-uniform cases: `<rare_label> (all)` when every group is
    below threshold, and `<rare_label> (all but <group>)` when only one group
    clears it; otherwise it lists the low-count groups explicitly.
    """
    flags: list[str] = []
    if any_iid_amp:
        flags.append("iid-amp")
    if bg_tight:
        flags.append("bg-tight")
    if rare_groups:
        n = len(group_tokens)
        if len(rare_groups) == n:
            flags.append(f"{rare_label} (all)")
        elif len(rare_groups) == n - 1:
            kept = next(g for g in group_tokens if g not in rare_groups)
            flags.append(f"{rare_label} (all but {kept})")
        else:
            flags.append(f"{rare_label} ({', '.join(rare_groups)})")
    return ", ".join(flags)


# -----------------------------------------------------------------------------
# Per-cell aggregation
# -----------------------------------------------------------------------------

def _aggregate_cells(df: pd.DataFrame, feat: str, conditions: list[str],
                     timepoints: list[str], group_tokens: list[str],
                     sig_thresh: float, iid_amp_thresh: float,
                     bg_tight_thresh: float, rare_k: int) -> pd.DataFrame:
    """One row per (site, feature), pooling the 6 (condition, timepoint) groups.

    `groups` is the flattened condition-major, timepoint-minor order used for
    the `log2_enrichment` display cell (BWM d0/d5/d10 then control d0/d5/d10).
    Concordance is decided on the sign of the per-group log2_enrichment values;
    a cell missing any of the 6 groups is recorded but excluded from the tables
    via the `n_valid == 6` filter.
    """
    df = df.copy()
    n_groups = len(conditions) * len(timepoints)

    rows = []
    for (site, feat_val), g in df.groupby(["site", feat], sort=False):
        by_ct = g.set_index(["condition", "timepoint"])

        # Per-condition rows of per-timepoint log2_enrichment (for the display
        # cell), plus flat lists for the aggregate statistics.
        cond_log2_lines: list[list[float]] = []
        flat_log2: list[float] = []
        flat_padj: list[float] = []
        flat_count: list[float] = []
        flat_bg: list[float] = []
        for cond in conditions:
            line: list[float] = []
            for tp in timepoints:
                if (cond, tp) in by_ct.index:
                    r = by_ct.loc[(cond, tp)]
                    log2 = float(r["log2_enrichment"])
                    padj = float(r["p_adj"])
                    cnt = float(r["observed_count"])
                    bg = float(r["bg_freq"])
                else:
                    log2 = padj = cnt = bg = np.nan
                line.append(log2)
                flat_log2.append(log2)
                flat_padj.append(padj)
                flat_count.append(cnt)
                flat_bg.append(bg)
            cond_log2_lines.append(line)

        valid_log2 = [v for v in flat_log2 if not pd.isna(v)]
        n_valid = len(valid_log2)

        if n_valid == n_groups:
            # signs excludes exact zeros; a zero would be its own sign and
            # break concordance, so we treat any exact zero as non-concordant.
            signs = {np.sign(v) for v in valid_log2 if v != 0}
            same_sign = len(signs) == 1 and 0 not in signs
            all_pos = same_sign and all(v > 0 for v in valid_log2)
            all_neg = same_sign and all(v < 0 for v in valid_log2)
        else:
            same_sign = all_pos = all_neg = False

        n_sig = int(sum((p < sig_thresh) for p in flat_padj if not pd.isna(p)))
        any_iid_amp = any((p < iid_amp_thresh) for p in flat_padj
                          if not pd.isna(p))
        bg_tight = any((b > bg_tight_thresh) for b in flat_bg
                       if not pd.isna(b))
        # Groups whose observed_count dips below the rare-k threshold, in the
        # canonical condition-major / timepoint-minor order (matches the
        # log2_enrichment display cell), so the flag can name exactly which of
        # the per-group values are low-count.
        rare_groups = [tok for tok, c in zip(group_tokens, flat_count)
                       if not pd.isna(c) and c < rare_k]

        # Smallest observed_count across the groups present — the count support
        # for the cell (how much data the weakest group contributes). Doubles
        # as the #sig tiebreaker so better-supported cells rank first.
        min_count = (int(np.nanmin(flat_count))
                     if any(not pd.isna(c) for c in flat_count) else 0)

        rows.append({
            "site": site,
            feat: feat_val,
            "cond_log2_lines": cond_log2_lines,
            "min_count": min_count,
            "n_sig": n_sig,
            "n_valid": n_valid,
            "same_sign": same_sign,
            "all_pos": all_pos,
            "all_neg": all_neg,
            "any_iid_amp": any_iid_amp,
            "bg_tight": bg_tight,
            "rare_groups": rare_groups,
        })

    return pd.DataFrame(rows)


def _select_and_sort(cells: pd.DataFrame, mask: pd.Series,
                     top_n: int) -> pd.DataFrame:
    """Top-n by (#sig desc, Max Change desc), then display-sorted.

    Selection ranks candidates by `#sig` desc, then `min count` desc, and
    keeps the top `top_n`. The kept rows are displayed sorted by Site (A/P/E),
    then `#sig` desc, then `min count` desc. `mask` already restricts to
    complete cells (the concordance masks require all 6 groups present, and the
    discordant mask carries an explicit `n_valid == n_groups` term).
    """
    sub = cells[mask].copy()
    sub = sub.sort_values(["n_sig", "min_count"], ascending=[False, False])
    sub = sub.head(top_n).copy()
    sub["site_key"] = sub["site"].map(_site_sort_key)
    return sub.sort_values(["site_key", "n_sig", "min_count"],
                           ascending=[True, False, False])


# -----------------------------------------------------------------------------
# Table printer
# -----------------------------------------------------------------------------

# Two-line separator for the log2_enrichment cell. `<br>` drives the line break
# in HTML (where a bare `\newline` is dropped); the format-gated
# `` `\newline`{=latex} `` drives it in PDF (where the raw-HTML `<br>` is
# dropped). Using both makes the two-line layout render in either output.
_CELL_BREAK = " <br>`\\newline`{=latex} "


def _log2_cell(cond_log2_lines: list[list[float]]) -> str:
    """Render the six per-group log2_enrichment values as a two-line cell."""
    lines = [", ".join(_fmt_signed(v) for v in line)
             for line in cond_log2_lines]
    return _CELL_BREAK.join(lines)


def _feature_cells(feat: str, val: str) -> list[str]:
    """Return the feature column cell(s) for one row.

    Amino-acid CSV: a single cell expanded to `K (lysine)` on every row (the
    full name is repeated rather than abbreviated after first use). Codon CSV:
    two cells, the bare three-letter codon plus its single-letter `aa`
    translation.
    """
    if feat == "amino_acid":
        if val in AA_FULL:
            return [f"{val} ({AA_FULL[val]})"]
        return [str(val)]
    return [str(val), CODON2AA.get(str(val), "?")]


def _print_table(sub: pd.DataFrame, title: str, feat: str,
                 group_tokens: list[str], rare_label: str) -> None:
    n_groups = len(group_tokens)
    print()
    print(f"### {title} (top {len(sub)} by #sig, then min count)")
    print()

    if feat == "amino_acid":
        header = ["Site", "amino acid", "log2_enrichment",
                  "min count", "#sig", "flag"]
    else:
        header = ["Site", "codon", "aa", "log2_enrichment",
                  "min count", "#sig", "flag"]

    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")

    if sub.empty:
        return

    for _, row in sub.iterrows():
        cells = [str(row["site"])]
        cells += _feature_cells(feat, row[feat])
        cells += [
            _log2_cell(row["cond_log2_lines"]),
            str(row["min_count"]),
            f"{row['n_sig']}/{n_groups}",
            _flag_cell(row["any_iid_amp"], row["bg_tight"],
                       row["rare_groups"], group_tokens, rare_label),
        ]
        print("| " + " | ".join(cells) + " |")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="path to the analysis_stats CSV")
    parser.add_argument("--top-n", type=int, default=10,
                        help="rows per table (default 10)")
    parser.add_argument("--sig-threshold", type=float, default=0.05,
                        help="p_adj threshold for the #sig count (default 0.05)")
    parser.add_argument("--iid-amp-threshold", type=float, default=1e-10,
                        help="any group p_adj below this flags the cell iid-amp "
                             "(default 1e-10)")
    parser.add_argument("--bg-tight-threshold", type=float, default=0.05,
                        help="bg_freq above this in any group flags the cell "
                             "bg-tight (default 0.05)")
    parser.add_argument("--rare-k-threshold", type=int, default=100,
                        help="min observed_count across the 6 groups below this "
                             "flags the cell rare-aa / rare-codon (default 100)")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)
    required = {"site", "condition", "timepoint", "observed_count",
                "bg_freq", "log2_enrichment", "p_adj"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing required columns: {sorted(missing)}",
              file=sys.stderr)
        return 2

    feat = _feature_col(df)
    rare_label = "rare-aa" if feat == "amino_acid" else "rare-codon"

    conditions = sorted(df["condition"].unique())
    timepoints = sorted(df["timepoint"].unique(), key=_tp_sort_key)
    n_groups = len(conditions) * len(timepoints)
    group_order = [f"{c} {t}" for c in conditions for t in timepoints]
    group_tokens = [_group_token(c, t) for c in conditions for t in timepoints]

    print("=" * 70)
    print(f"# {args.csv}")
    print(f"# feature: {feat}    rows: {len(df)}    groups ({n_groups}): "
          f"{', '.join(group_order)}")
    print(f"# sig p_adj<{args.sig_threshold}, iid-amp p_adj<{args.iid_amp_threshold}, "
          f"bg-tight bg_freq>{args.bg_tight_threshold}, rare-k <{args.rare_k_threshold}")
    print("=" * 70)
    print()
    print(f"Each (site, {feat}) cell pools the {n_groups} per-group "
          f"`log2_enrichment` values (binomial effect vs that group's "
          f"`bg_freq`). The `log2_enrichment` cell lists them in two lines: "
          f"{conditions[0]} {', '.join(timepoints)} on the first line, then "
          f"{conditions[1]} {', '.join(timepoints)} on the second. "
          f"**Concordant** cells share one sign across all {n_groups} groups; "
          f"**discordant** cells have at least one sign disagreement. `#sig` is "
          f"the count of groups with `p_adj` < {args.sig_threshold}; `min count` "
          f"is the smallest `observed_count` across the {n_groups} groups (the "
          f"weakest group's data support). Each table is the top {args.top_n} by "
          f"`#sig` desc then `min count` desc, displayed sorted by Site (A/P/E), "
          f"then `#sig` desc, then `min count` desc.")

    cells = _aggregate_cells(
        df, feat, conditions, timepoints, group_tokens,
        sig_thresh=args.sig_threshold,
        iid_amp_thresh=args.iid_amp_threshold,
        bg_tight_thresh=args.bg_tight_threshold,
        rare_k=args.rare_k_threshold,
    )

    incomplete = int((cells["n_valid"] != n_groups).sum())
    if incomplete:
        print()
        print(f"_(NaN guard: {incomplete} of {len(cells)} cells lack a row in "
              f"all {n_groups} groups and are excluded from the tables.)_")

    n_conc_enr = int(cells["all_pos"].sum())
    n_conc_dep = int(cells["all_neg"].sum())
    n_disc = int(((~cells["same_sign"]) & (cells["n_valid"] == n_groups)).sum())
    print()
    print(f"_(Pool: {n_conc_enr} concordant-enriched, {n_conc_dep} "
          f"concordant-depleted, {n_disc} discordant cells in the data; each "
          f"table below shows the top {args.top_n}.)_")

    _print_table(_select_and_sort(cells, cells["all_pos"], args.top_n),
                 "Concordant enrichment", feat, group_tokens, rare_label)
    _print_table(_select_and_sort(cells, cells["all_neg"], args.top_n),
                 "Concordant depletion", feat, group_tokens, rare_label)
    _print_table(
        _select_and_sort(
            cells, (~cells["same_sign"]) & (cells["n_valid"] == n_groups),
            args.top_n),
        "Discordant", feat, group_tokens, rare_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
