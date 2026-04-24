#!/usr/bin/env python3
"""
stall_sites_non_consensus_stats.py

Statistical half of the non-consensus stall-site pipeline. Consumes ONE of the
two CSVs emitted by ``stall_sites_non_consensus_call.py``:

  * ``stall_sites_codon.csv`` → codon-level enrichment (alphabet = 61 sense codons)
  * ``stall_sites_aa.csv``    → amino-acid-level enrichment (alphabet = AA_ORDER)

Level is auto-detected from the input columns. The three enrichment tests
(within-condition binomial, between-condition Wilcoxon, per-timepoint Fisher)
are run identically in both modes — the only things that differ are the
feature alphabet and the background-frequency helper.

This script is intentionally ribopy-free: per-group background frequencies
are read from the ``per_group_background_{codon,aa}.csv`` CSVs emitted by
``stall_sites_non_consensus_call.py``. That lets the stats run on a machine
without ribopy / the source ``.ribo`` file.

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
    between_condition_wilcoxon,
    per_timepoint_fisher,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run enrichment tests on a per-stall CSV produced by stall_sites_non_consensus_call.py."
    )
    parser.add_argument("--stall-sites", required=True,
                        help="Path to stall_sites_codon.csv or stall_sites_aa.csv")
    parser.add_argument("--groups", required=True,
                        help="Experimental groups, e.g. 'groupA:rep1,rep2;groupB:rep3,rep4'")
    parser.add_argument("--background", required=True,
                        help="Path to per_group_background_{level}.csv written by the call script.")
    parser.add_argument("--out-dir", default="results/stall_sites/enrichment",
                        help="Output directory for enrichment CSVs")
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
    for rep, sub in df.groupby("replicate"):
        rep_map = {}
        for site_name, col in zip(("E", "P", "A"), site_cols):
            counts = sub[col].value_counts()
            rep_map[site_name] = counts.reindex(alphabet, fill_value=0).astype(int)
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
    logging.info(f"Loading stall sites from {stall_path} ...")
    df = pd.read_csv(stall_path)
    # level = "codon" or "aa"; site_cols = (E_col, P_col, A_col); alphabet = list of codons or AAs; feature_col = "codon" or "amino_acid"
    level, site_cols, alphabet, feature_col = detect_level(df)
    suffix = level  # "codon" or "aa"
    logging.info(f"Detected level: {level} ({len(alphabet)} categories; feature column '{feature_col}')")

    # --------------------------------------------------------------
    # Groups, condition, timepoint mappings
    # --------------------------------------------------------------
    groups = parse_groups(args.groups)
    # Map reps to groups
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    rep_to_condition = {}
    rep_to_timepoint = {}
    for rep, grp in rep_to_group.items():
        # BWM_day_0 → condition = BWM, timepoint = day_0
        parts = grp.split("_", 1)
        rep_to_condition[rep] = parts[0]
        rep_to_timepoint[rep] = parts[1] if len(parts) > 1 else grp

    # --------------------------------------------------------------
    # Per-replicate counts
    # --------------------------------------------------------------
    # Returns the amino acid or codon stall site count for each replicate at each site (E, P, A).
    replicate_counts = build_replicate_counts(df, site_cols, alphabet)

    # Print total counts per site across replicates (for sanity check)
    print(f"\n{'='*60}\nTOTAL STALL SITE COUNTS PER SITE (summed across replicates)\n{'='*60}")
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
    # Analysis 2: Between-condition Wilcoxon
    # --------------------------------------------------------------
    print(f"\n{'='*60}\nANALYSIS 2: BETWEEN-CONDITION WILCOXON\n{'='*60}")
    df_wilcox = between_condition_wilcoxon(
        replicate_counts, rep_to_condition, feature_col=feature_col,
    )
    n_sig = (df_wilcox["p_adj"] < 0.05).sum() if not df_wilcox.empty else 0
    print(f"  Tests: {len(df_wilcox)}  |  Significant (p_adj<0.05): {n_sig}")

    # --------------------------------------------------------------
    # Analysis 3: Per-timepoint Fisher's
    # --------------------------------------------------------------
    print(f"\n{'='*60}\nANALYSIS 3: PER-TIMEPOINT FISHER'S EXACT\n"
          f"  NOTE: pooling replicates is pseudoreplication — interpret cautiously\n{'='*60}")
    df_fisher = per_timepoint_fisher(
        replicate_counts, rep_to_condition, rep_to_timepoint, feature_col=feature_col,
    )
    for tp in sorted(df_fisher["timepoint"].unique()) if not df_fisher.empty else []:
        tp_df = df_fisher[df_fisher["timepoint"] == tp]
        n_sig_tp = (tp_df["p_adj"] < 0.05).sum()
        print(f"  [{tp}] {len(tp_df)} tests, {n_sig_tp} significant")

    # --------------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------------
    within_path = out_dir / f"within_condition_enrichment_{suffix}.csv"
    wilcox_path = out_dir / f"between_condition_wilcoxon_{suffix}.csv"
    fisher_path = out_dir / f"per_timepoint_fisher_{suffix}.csv"
    df_within.to_csv(within_path, index=False)
    df_wilcox.to_csv(wilcox_path, index=False)
    df_fisher.to_csv(fisher_path, index=False)

    print(f"\nSaved:")
    for p in (within_path, wilcox_path, fisher_path):
        print(f"  {p}")
    logging.info(f"All {level}-level enrichment results saved to {out_dir}")


if __name__ == "__main__":
    main()
