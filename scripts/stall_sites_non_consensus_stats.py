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

Run it twice, once per CSV, to get codon-level and AA-level outputs side by
side. Output filenames are suffixed with ``_codon`` or ``_aa`` accordingly.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import ribopy
from ribopy import Ribo

from ribostall.amino_acids import (
    AA_ORDER,
    SENSE_CODONS,
    background_aa_freq,
    background_codon_freq,
)
from ribostall.sequence import get_sequence, get_cds_range_lookup
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
    parser.add_argument("--ribo", required=True, help="Path to ribo file")
    parser.add_argument("--reference", required=True, help="Reference FASTA file for background frequencies")
    parser.add_argument("--groups", required=True,
                        help="Experimental groups, e.g. 'groupA:rep1,rep2;groupB:rep3,rep4'")
    parser.add_argument("--filtered-tx",
                        help="Path to filtered_transcripts.json written by the call script. "
                             "If omitted, defaults to <stall-sites directory>/filtered_transcripts.json.")
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
    input_dir = out_dir / "input_data"
    input_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Load stall sites and detect granularity
    # --------------------------------------------------------------
    stall_path = Path(args.stall_sites)
    logging.info(f"Loading stall sites from {stall_path} ...")
    df = pd.read_csv(stall_path)
    level, site_cols, alphabet, feature_col = detect_level(df)
    suffix = level  # "codon" or "aa"
    logging.info(f"Detected level: {level} ({len(alphabet)} categories; feature column '{feature_col}')")

    # --------------------------------------------------------------
    # Groups, condition, timepoint mappings
    # --------------------------------------------------------------
    groups = parse_groups(args.groups)
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    rep_to_condition = {}
    rep_to_timepoint = {}
    for rep, grp in rep_to_group.items():
        parts = grp.split("_", 1)
        rep_to_condition[rep] = parts[0]
        rep_to_timepoint[rep] = parts[1] if len(parts) > 1 else grp

    # --------------------------------------------------------------
    # Per-replicate counts
    # --------------------------------------------------------------
    replicate_counts = build_replicate_counts(df, site_cols, alphabet)
    for rep, site_counts in replicate_counts.items():
        totals = {s: int(site_counts[s].sum()) for s in ("E", "P", "A")}
        print(f"  [{rep}] counts per site: {totals}")

    # --------------------------------------------------------------
    # Reconstruct per-group filtered transcripts for backgrounds
    # --------------------------------------------------------------
    filt_path = Path(args.filtered_tx) if args.filtered_tx else stall_path.parent / "filtered_transcripts.json"
    logging.info(f"Loading filtered transcripts from {filt_path} ...")
    with open(filt_path) as f:
        filt_tx_dict = {g: set(txs) for g, txs in json.load(f).items()}
    for g, txs in filt_tx_dict.items():
        print(f"  [{g}] {len(txs)} filtered transcripts (for background)")

    # --------------------------------------------------------------
    # Load sequences to compute backgrounds
    # --------------------------------------------------------------
    logging.info(f"Loading ribo object from {args.ribo} ...")
    ribo_object = Ribo(args.ribo, alias=None)
    logging.info("Looking up CDS ranges ...")
    cds_range = get_cds_range_lookup(ribo_object)
    logging.info(f"Loading sequences from {args.reference} ...")
    sequence = get_sequence(ribo_object, args.reference, alias=ribopy.api.alias.apris_human_alias)

    print(f"\n{'='*60}\nBACKGROUND {level.upper()} FREQUENCIES (per group)\n{'='*60}")
    bg_freq_per_group = {}
    bg_counts_per_group = {}
    for grp, grp_txs in filt_tx_dict.items():
        if level == "aa":
            bg_freq, bg_counts = background_aa_freq(grp_txs, cds_range, sequence, AA_ORDER)
        else:
            bg_freq, bg_counts = background_codon_freq(grp_txs, cds_range, sequence, SENSE_CODONS)
        bg_freq_per_group[grp] = bg_freq
        bg_counts_per_group[grp] = bg_counts
        print(f"  [{grp}] {len(grp_txs)} transcripts ({int(bg_counts.sum())} total {level}s)")
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

    # Per-replicate frequencies at this level
    freq_rows = []
    for rep, site_counts in replicate_counts.items():
        grp = rep_to_group.get(rep, "")
        cond = rep_to_condition.get(rep, "")
        tp = rep_to_timepoint.get(rep, "")
        for site_name in ("E", "P", "A"):
            counts = site_counts[site_name]
            total = int(counts.sum())
            for feat in counts.index:
                freq_rows.append({
                    "replicate": rep,
                    "group": grp,
                    "condition": cond,
                    "timepoint": tp,
                    "site": site_name,
                    feature_col: feat,
                    "stall_count": int(counts[feat]),
                    "total_stall_sites": total,
                    "stall_freq": counts[feat] / total if total > 0 else 0.0,
                })
    freq_path = out_dir / f"replicate_{suffix}_frequencies.csv"
    pd.DataFrame(freq_rows).to_csv(freq_path, index=False)

    # Per-group background
    bg_rows = []
    for grp in bg_freq_per_group:
        for feat in bg_freq_per_group[grp].index:
            bg_rows.append({
                "group": grp,
                feature_col: feat,
                "bg_count": int(bg_counts_per_group[grp][feat]),
                "bg_freq": float(bg_freq_per_group[grp][feat]),
            })
    bg_path = input_dir / f"per_group_background_{suffix}.csv"
    pd.DataFrame(bg_rows).to_csv(bg_path, index=False)

    print(f"\nSaved:")
    for p in (within_path, wilcox_path, fisher_path, freq_path, bg_path):
        print(f"  {p}")
    logging.info(f"All {level}-level enrichment results saved to {out_dir}")


if __name__ == "__main__":
    main()
