#!/usr/bin/env python3
"""
stall_sites_consensus_stats.py

Statistical half of the *consensus* stall-site pipeline. Consumes ONE of the
two CSVs emitted by ``stall_sites_consensus.py``:

  * ``stall_sites_codon.csv`` → codon-level enrichment (alphabet = 61 sense codons)
  * ``stall_sites_aa.csv``    → amino-acid-level enrichment (alphabet = AA_ORDER)

Level is auto-detected from the input columns.

The consensus experiment is a flat **control vs treatment** design with no
timepoints. After consensus there is exactly one stall set per group (the
consensus collapses replicates), so only two comparisons are meaningful:

  * Analysis 1 — within-condition enrichment (binomial vs each group's background)
  * Analysis 2 — between-condition Fisher's exact (control vs treatment)
  * Analysis 3 — between-condition background-aware diff (control vs treatment)

Analysis 3 is the background-aware counterpart of Analysis 2: instead of
comparing raw stall-site shares between conditions (Fisher), it compares each
condition's enrichment over its OWN background, so a shift in the expressed
transcriptome between conditions cannot masquerade as differential stalling. It
complements rather than replaces Fisher — the two agree when the per-condition
backgrounds match and diverge only when they differ.

The Wilcoxon and per-/between-timepoint analyses from the non-consensus stats
script are N/A here (n=1 per arm, no timepoints) and are deliberately omitted.

Like ``stall_sites_non_consensus_stats.py`` this is intentionally ribopy-free:
per-group background frequencies are read from the
``per_group_background_{codon,aa}.csv`` CSVs emitted by
``stall_sites_consensus.py``, so the stats run on a machine without ribopy / the
source ``.ribo`` file.

Run it twice, once per CSV, to get codon-level and AA-level outputs side by
side. Output filenames are suffixed with ``_codon`` or ``_aa`` accordingly.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from ribostall.amino_acids import AA_ORDER, SENSE_CODONS
from ribostall.enrichment import (
    within_condition_enrichment,
    between_condition_fisher,
    between_condition_background_diff,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run within-condition binomial + between-condition Fisher tests on a "
                    "consensus stall-site CSV produced by stall_sites_consensus.py."
    )
    parser.add_argument("--stall-sites", required=True,
                        help="Path to stall_sites_codon.csv or stall_sites_aa.csv")
    parser.add_argument("--groups", required=True,
                        help="Experimental groups, e.g. 'control:control;treatment:treatment' "
                             "(consensus sets replicate == group name)")
    parser.add_argument("--background", required=True,
                        help="Path to per_group_background_{level}.csv written by the consensus call script.")
    parser.add_argument("--out-dir", default="results/stall_sites/enrichment",
                        help="Output directory for enrichment CSVs")
    parser.add_argument("--headline-condition", default=None,
                        help="Condition treated as the headline (numerator) in BOTH between-"
                             "condition tests: a positive log2(odds ratio) [Fisher, Analysis 2] or "
                             "delta_log2_enrichment [background-aware, Analysis 3] means enriched in "
                             "this condition. Must match one of the two group labels "
                             "(e.g. 'treatment'). Default: alphabetical (first condition is headline).")
    return parser.parse_args()


def parse_groups(groups_arg):
    groups = {}
    for block in groups_arg.split(";"):
        name, reps = block.split(":")
        groups[name] = reps.split(",")
    return groups


def detect_level(df: pd.DataFrame) -> tuple[str, tuple[str, str, str], list, str]:
    """Return (level, (E_col, P_col, A_col), alphabet, feature_col_name)."""
    if {"E_codon", "P_codon", "A_codon"}.issubset(df.columns):
        return "codon", ("E_codon", "P_codon", "A_codon"), list(SENSE_CODONS), "codon"
    if {"E_aa", "P_aa", "A_aa"}.issubset(df.columns):
        return "aa", ("E_aa", "P_aa", "A_aa"), list(AA_ORDER), "amino_acid"
    raise ValueError(
        "Input CSV must contain either (E_codon,P_codon,A_codon) or (E_aa,P_aa,A_aa) columns."
    )


def build_replicate_counts(df: pd.DataFrame, site_cols, alphabet) -> dict:
    """{rep: {"E": Series, "P": Series, "A": Series}} indexed by ``alphabet``."""
    out = {}
    # Groups by control or treatment
    for rep, sub in df.groupby("replicate"):
        rep_map = {}
        # The name of the site and the column that site lives in
        for site_name, col in zip(("E", "P", "A"), site_cols):
            # Counts the occurrences of each Amino Acid or Codon in the column
            # For example: {"A": 10, "C": 5, "G": 3, "T": 2}
            counts = sub[col].value_counts()
            # Reindex the counts to match the alphabet and fill missing values with 0
            # It will look like {E: {"A": 10, "C": 5, "G": 3, "T": 2}, P: {"A": 10, "C": 5, "G": 3, "T": 2}, A: {"A": 10, "C": 5, "G": 3, "T": 2}}
            rep_map[site_name] = counts.reindex(alphabet, fill_value=0).astype(int)
        # Stores the above dictionary for the replicate
        out[rep] = rep_map
    return out


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Load stall sites and detect granularity
    # --------------------------------------------------------------
    stall_path = Path(args.stall_sites)
    logging.info(f"Loading consensus stall sites from {stall_path} ...")
    df = pd.read_csv(stall_path)
    # level = "codon" or "aa"; site_cols = (E_col, P_col, A_col); alphabet = list of codons or AAs; feature_col = "codon" or "amino_acid"
    level, site_cols, alphabet, feature_col = detect_level(df)
    suffix = level  # "codon" or "aa"
    logging.info(f"Detected level: {level} ({len(alphabet)} categories; feature column '{feature_col}')")

    # --------------------------------------------------------------
    # Groups and condition mapping. Consensus collapses replicates into one set
    # per group and writes replicate == group, so each "replicate" is a group
    # and the condition is the group name itself (flat control vs treatment).
    # --------------------------------------------------------------
    # Ex: 'control:control;treatment:treatment'
    # {"control": ["control"], "treatment": ["treatment"]}
    groups = parse_groups(args.groups)
    # Would look like {"control": "control", "treatment": "treatment"}
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    # Flat design: no timepoints, so condition is the group name (e.g. control, treatment).
    # Would look the same as rep_to_group
    rep_to_condition = {rep: grp.split("_", 1)[0] for rep, grp in rep_to_group.items()}

    # --------------------------------------------------------------
    # Per-replicate counts (one "replicate" per group after consensus)
    # --------------------------------------------------------------
    # Builds
    replicate_counts = build_replicate_counts(df, site_cols, alphabet)

    # Print total counts per site across replicates (for sanity check)
    print(f"\n{'='*60}\nTOTAL CONSENSUS STALL SITE COUNTS PER SITE\n{'='*60}")
    for rep, site_counts in replicate_counts.items():
        totals = {s: int(site_counts[s].sum()) for s in ("E", "P", "A")}
        print(f"  [{rep}] counts per site: {totals}")

    # --------------------------------------------------------------
    # Load per-group background frequencies (written by the call script)
    # --------------------------------------------------------------
    bg_path = Path(args.background)
    bg_freq_per_group = {}
    bg_counts_per_group = {}

    logging.info(f"Loading per-group {level} backgrounds from {bg_path} ...")
    bg_df = pd.read_csv(bg_path)

    print(f"\n{'='*60}\nBACKGROUND {level.upper()} FREQUENCIES (per group)\n{'='*60}")

    for grp, sub in bg_df.groupby("group"):
        freq = sub.set_index(feature_col)["bg_freq"].reindex(alphabet).astype(float)
        counts = sub.set_index(feature_col)["bg_count"].reindex(alphabet).fillna(0).astype(int)
        bg_freq_per_group[grp] = freq
        bg_counts_per_group[grp] = counts
        print(f"  [{grp}] {int(counts.sum())} total {level}s")
    print(f"{'='*60}\n")

    # --------------------------------------------------------------
    # Analysis 1: Within-condition enrichment (binomial)
    # --------------------------------------------------------------
    print(f"\n{'='*60}\nANALYSIS 1: WITHIN-CONDITION ENRICHMENT (Binomial)\n{'='*60}")
    df_within = within_condition_enrichment(
        replicate_counts, bg_freq_per_group, rep_to_condition, rep_to_group,
        feature_col=feature_col,
    )
    n_sig = (df_within["p_adj"] < 0.05).sum() if not df_within.empty else 0
    print(f"  Tests: {len(df_within)}  |  Significant (p_adj<0.05): {n_sig}")

    # --------------------------------------------------------------
    # Analysis 2: Between-condition Fisher's exact (control vs treatment)
    # --------------------------------------------------------------
    print(f"\n{'='*60}\nANALYSIS 2: BETWEEN-CONDITION FISHER'S EXACT\n"
          f"  NOTE: consensus pools sites per group — interpret cautiously\n{'='*60}")
    if args.headline_condition is not None:
        print(f"  Headline condition: {args.headline_condition} "
              f"(positive log2 odds ratio = enriched in {args.headline_condition})")
    else:
        print("  Headline condition: alphabetical default "
              "(positive log2 odds ratio = enriched in the first condition)")
    df_fisher = between_condition_fisher(
        replicate_counts, rep_to_condition, feature_col=feature_col,
        headline_condition=args.headline_condition,
    )

    # Printing Purposes
    for site in sorted(df_fisher["site"].unique()) if not df_fisher.empty else []:
        site_df = df_fisher[df_fisher["site"] == site]
        n_sig_s = (site_df["p_adj"] < 0.05).sum()
        print(f"  [{site}] {len(site_df)} tests, {n_sig_s} significant")

    # --------------------------------------------------------------
    # Analysis 3: Between-condition background-aware diff (control vs treatment)
    # --------------------------------------------------------------
    # NOTE: same control-vs-treatment comparison as Analysis 2, but each
    # condition is normalized to its OWN background before comparison (Fisher
    # compares raw shares). Positive delta_log2_enrichment = more enriched vs
    # background in the headline condition. bg_freq_per_group keys by group, and
    # group == condition in this flat consensus design, so it passes through.
    print(f"\n{'='*60}\nANALYSIS 3: BETWEEN-CONDITION BACKGROUND-AWARE DIFF\n"
          f"  NOTE: enrichment-over-background ratio; consensus pools sites — interpret cautiously\n{'='*60}")
    if args.headline_condition is not None:
        print(f"  Headline condition: {args.headline_condition} "
              f"(positive delta_log2_enrichment = more enriched vs background in {args.headline_condition})")
    else:
        print("  Headline condition: alphabetical default "
              "(positive delta_log2_enrichment = more enriched vs background in the first condition)")
    df_bgdiff = between_condition_background_diff(
        replicate_counts, rep_to_condition, bg_freq_per_group,
        feature_col=feature_col, headline_condition=args.headline_condition,
    )

    # Printing Purposes
    for site in sorted(df_bgdiff["site"].unique()) if not df_bgdiff.empty else []:
        site_df = df_bgdiff[df_bgdiff["site"] == site]
        n_sig_s = (site_df["p_adj"] < 0.05).sum()
        print(f"  [{site}] {len(site_df)} tests, {n_sig_s} significant")

    # --------------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------------
    within_path = out_dir / f"within_condition_binomial_{suffix}.csv"
    fisher_path = out_dir / f"between_condition_fisher_{suffix}.csv"
    bgdiff_path = out_dir / f"between_condition_background_diff_{suffix}.csv"
    df_within.to_csv(within_path, index=False)
    df_fisher.to_csv(fisher_path, index=False)
    df_bgdiff.to_csv(bgdiff_path, index=False)

    print(f"\nSaved:")
    for p in (within_path, fisher_path, bgdiff_path):
        print(f"  {p}")
    logging.info(f"All {level}-level consensus enrichment results saved to {out_dir}")


if __name__ == "__main__":
    main()
