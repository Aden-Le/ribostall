#!/usr/bin/env python3
"""
stall_sites_non_consensus_call.py

Data-generation half of the non-consensus stall-site pipeline.

Loads coverage, calls stall sites per replicate, annotates each stall with its
E/P/A codon and amino acid identities, and writes two tidy CSVs plus matching
per-group background frequency CSVs:

  * ``stall_sites_codon.csv``       — one row per stall, with E_codon/P_codon/A_codon
  * ``stall_sites_aa.csv``          — one row per stall, with E_aa/P_aa/A_aa
  * ``per_group_background_codon.csv`` — per-group codon background counts/freqs
  * ``per_group_background_aa.csv``    — per-group AA background counts/freqs

All ribopy / reference-FASTA work is concentrated here so the sister stats
script can consume the CSV intermediates on a machine without ribopy installed
(e.g. outside the SSH environment that hosts the ``.ribo`` file).
"""

import argparse
import gzip
import logging
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from ribopy import Ribo

from ribostall.stall_sites import (
    filter_tx,
    codonize_counts_cds,
    call_stalls,
    stalls_to_long_df,
    consensus_stalls_across_reps,
)
from ribostall.amino_acids import (
    AA_ORDER,
    SENSE_CODONS,
    STOP_CODONS,
    annotate_stalls_epa,
    background_aa_freq,
    background_codon_freq,
)
from ribostall.sequence import get_sequence, get_cds_range_lookup
from ribostall.enrichment import plot_coverage_density


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Call per-replicate ribosome stall sites and emit codon-level and AA-level CSVs."
    )
    parser.add_argument("--pickle", required=True, help="Path to coverage pickle.gz file")
    parser.add_argument("--ribo", required=True, help="Path to ribo file")
    parser.add_argument("--reference", required=True,
                        help="Reference FASTA file used to look up CDS sequences for E/P/A annotation")
    parser.add_argument("--groups", required=True,
                        help="Experimental groups, e.g. 'groupA:rep1,rep2;groupB:rep3,rep4'")
    parser.add_argument("--tx_threshold", type=float, default=1.0,
                        help="Minimum reads/nt (in CDS) for filtering transcripts")
    parser.add_argument("--tx_min_reps", type=int, default=2,
                        help="Minimum number of replicates passing threshold for filtering transcripts")
    parser.add_argument("--min_z", type=float, default=2.0,
                        help="Minimum z-score to pass as stall site")
    parser.add_argument("--min_reads", type=int, default=5,
                        help="Minimum number of reads to pass as stall site")
    parser.add_argument("--trim-start", type=int, default=20,
                        help="Exclude first N codons from start (initiation ramp)")
    parser.add_argument("--trim-stop", type=int, default=10,
                        help="Exclude last N codons before stop (termination region)")
    parser.add_argument("--pseudocount", type=float, default=0.5,
                        help="Pseudocount for stall calling")
    parser.add_argument("--psite-offset", type=int, default=0,
                        help="Codon offset applied to each stall index before deriving E/P/A sites")
    parser.add_argument("--basis", choices=("P", "A"), default="P",
                        help="Register for E/P/A offsets (P: E=-1,P=0,A=+1; A: E=-2,P=-1,A=0)")
    parser.add_argument("--drop-stop-codons", choices=("True", "False"), default="True",
                        help="Drop stall windows whose E/P/A site hits a stop codon "
                             "(TAA/TAG/TGA) from the output CSVs. Default: True; pass "
                             "'--drop-stop-codons False' to keep them.")
    parser.add_argument("--out-dir", default="results/stall_sites/enrichment",
                        help="Output directory for stall-site CSVs")
    return parser.parse_args()


def parse_groups(groups_arg):
    groups = {}
    for block in groups_arg.split(";"):
        name, reps = block.split(":")
        groups[name] = reps.split(",")
    return groups


def main():
    args = parse_args()

    # Parse the groups argument into a dict of {group: [rep1, rep2, ...]} and also build a reverse mapping of {rep: group}.
    groups = parse_groups(args.groups)
    logging.info(f"Parsed {len(groups)} groups: {list(groups.keys())}")
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}

    # ------------------------------------------------------------------
    # Load inputs (Loads Coverage & Ribo object)
    # ------------------------------------------------------------------
    logging.info(f"Loading coverage from {args.pickle} ...")
    with gzip.open(args.pickle, "rb") as f:
        cov = pickle.load(f)
    logging.info(f"Coverage loaded: {len(cov)} experiments")

    logging.info(f"Loading ribo object from {args.ribo} ...")
    ribo_object = Ribo(args.ribo, alias=None)

    print(f"\n{'='*60}\nLOADING SEQUENCES\n{'='*60}")
    logging.info("Looking up CDS ranges ...")
    cds_range = get_cds_range_lookup(ribo_object)
    print(f"  CDS ranges: {len(cds_range)} transcripts")
    logging.info(f"Loading sequences from {args.reference} ...")
    sequence = get_sequence(ribo_object, args.reference, alias=False)
    print(f"  Sequences: {len(sequence)} transcripts")
    print(f"{'='*60}\n")

    missing = [r for rs in groups.values() for r in rs if r not in cov]
    if missing:
        print("Warning: the following replicates are missing from coverage:", ", ".join(missing))

    # ------------------------------------------------------------------
    # coverage density plot
    # ------------------------------------------------------------------
    # Total number of transcripts before filtering (for reporting how many were lost).
    n_before = len(next(iter(cov.values())))
    trim_start_nt = args.trim_start * 3
    trim_stop_nt = args.trim_stop * 3

    def _body(arr):
        arr = np.asarray(arr, float)
        if len(arr) <= trim_start_nt + trim_stop_nt:
            return arr[:0]
        return arr[trim_start_nt : len(arr) - trim_stop_nt]

    print(f"\n{'='*60}\nTRANSCRIPT FILTERING (per-group, no intersection)\n{'='*60}")
    print(f"Transcripts before filtering: {n_before}")
    print(f"Body window: first {args.trim_start} and last {args.trim_stop} codons removed")
    print(f"\n  {'Replicate':<25} {'Group':<15} {'Avg body cov/tx (reads/nt)':>28} {'SD':>10} {'Total body coverage':>20}")
    print(f"  {'-'*25} {'-'*15} {'-'*28} {'-'*10} {'-'*20}")

    for group, reps in groups.items():
        for rep in reps:
            tx_dict = cov[rep]
            bodies = [_body(v) for v in tx_dict.values()]
            means = [b.mean() for b in bodies if b.size]
            avg_cov = np.mean(means) if means else 0.0
            sd_cov = np.std(means) if means else 0.0
            total_cov = sum(b.sum() for b in bodies if b.size)
            print(f"  {rep:<25} {group:<15} {avg_cov:>28.4f} {sd_cov:>10.4f} {total_cov:>20,.0f}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_coverage_density(cov, groups, args.out_dir,
                          trim_start=args.trim_start, trim_stop=args.trim_stop)
    logging.info(f"Saved coverage density plot to {args.out_dir}/coverage_density.png")

    # ------------------------------------------------------------------
    # Transcript filtering (per-group, no intersection)
    # ------------------------------------------------------------------

    # Returns a dict of {group: set(transcripts)} passing the filter.
    filt_tx_dict = {
        group: filter_tx(cov, reps, min_reps=args.tx_min_reps, threshold=args.tx_threshold,
                         trim_start=args.trim_start, trim_stop=args.trim_stop)
        for group, reps in groups.items()
    }

    for group, txs in filt_tx_dict.items():
        print(f"  Per-group filter [{group}]: {len(txs)} transcripts  (lost {n_before - len(txs)})")

    # ------------------------------------------------------------------
    # Sanity check: every filtered transcript must resolve to both a CDS
    # range and a FASTA sequence, otherwise annotate_stalls_epa and the
    # background helpers will silently drop it.
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nSEQUENCE RESOLUTION SANITY CHECK\n{'='*60}")
    any_missing = False
    for group, txs in filt_tx_dict.items():
        missing_cds, missing_seq = [], []
        for tx in txs:
            key = tx if tx in cds_range else (tx.split("|")[4] if "|" in tx else tx)
            if key not in cds_range:
                missing_cds.append(tx)
            if tx not in sequence:
                missing_seq.append(tx)
        resolved = len(txs) - len(set(missing_cds) | set(missing_seq))
        print(f"  [{group}] {resolved}/{len(txs)} resolved  "
              f"(missing CDS range: {len(missing_cds)}, missing sequence: {len(missing_seq)})")
        if missing_cds:
            any_missing = True
            print(f"    first few missing CDS: {missing_cds[:5]}")
        if missing_seq:
            any_missing = True
            print(f"    first few missing seq: {missing_seq[:5]}")
    if not any_missing:
        print("  All filtered transcripts resolved to both CDS range and sequence.")
    print(f"{'='*60}\n")

    #-------------------------------------------------------------------
    cov_filt = {}
    for exp, tx_dict in cov.items():
        # Gets the group for this replicate (BWM or Control)
        grp = rep_to_group.get(exp)
        if grp is None:
            continue
        # Gets the set of transcripts that passed the filter for this group
        grp_txs = filt_tx_dict[grp]
        # For each experiment, keep only the transcripts that passed the filter for its group
        cov_filt[exp] = {tx: arr for tx, arr in tx_dict.items() if tx in grp_txs}


        

    # ------------------------------------------------------------------
    # Codonize + call stalls
    # ------------------------------------------------------------------
    logging.info("Codonizing coverage arrays ...")
    # Returns the coverage but codonized (i.e. summed in-frame) and filtered to the transcripts that passed the tx filter.
    codon_cov = {
        exp: {tx: codonize_counts_cds(arr) for tx, arr in tx_dict.items()}
        for exp, tx_dict in cov_filt.items()
    }

    logging.info(
        f"Calling stall sites (min_z={args.min_z}, min_reads={args.min_reads}, "
        f"trim_start={args.trim_start}, trim_stop={args.trim_stop}) ..."
    )
    # Get the stall sites per transcript per experiment, applying the specified parameters for calling.
    stalls = {
        exp: {
            tx: call_stalls(
                arr,
                min_z=args.min_z,
                min_obs=args.min_reads,
                trim_start=args.trim_start,
                trim_stop=args.trim_stop,
                pseudocount=args.pseudocount,
            )
            for tx, arr in tx_dict.items()
        }
        for exp, tx_dict in codon_cov.items()
    }

    # Print some summary stats about the stall sites called per experiment. (logging only)
    print(f"\n{'='*60}\nSTALL SITE CALLING\n{'='*60}")
    total_counts = {
        exp: sum(len(idxs) for idxs in tx_stalls.values())
        for exp, tx_stalls in stalls.items()
    }
    for exp, count in total_counts.items():
        grp = rep_to_group.get(exp, "?")
        print(f"  [{exp}] ({grp}): {count} stall sites  ({len(codon_cov[exp])} transcripts)")
    print(f"  Total across all experiments: {sum(total_counts.values())} stall sites")
    print(f"{'='*60}\n")

    # Reproducibility diagnostic (logging only).
    print(f"{'='*60}\nREPRODUCIBILITY (consensus across replicates)\n{'='*60}")
    for grp, reps in groups.items():
        if len(reps) < 2:
            print(f"  [{grp}]: skipped (only {len(reps)} replicate)")
            continue
        consensus = consensus_stalls_across_reps(stalls, reps, min_support=2, tol=0)
        n_consensus = sum(len(idxs) for idxs in consensus.values())
        n_total_union = len(set(
            (tx, d["index"])
            for r in reps
            for tx, site_list in stalls[r].items()
            for d in site_list
        ))
        pct = (n_consensus / n_total_union * 100) if n_total_union else 0
        print(f"  [{grp}] ({len(reps)} reps): "
              f"{n_consensus} consensus / {n_total_union} union stall sites "
              f"({pct:.1f}% reproducible)")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Long dataframe of stalls
    # ------------------------------------------------------------------
    logging.info("Converting stalls to long-format dataframe ...")
    df = stalls_to_long_df(stalls, rep_to_group)
    logging.info(f"Long-format dataframe: {len(df)} rows")

    # ------------------------------------------------------------------
    # Annotate each stall with E/P/A codon + AA
    # ------------------------------------------------------------------
    logging.info("Annotating stalls with E/P/A codons and amino acids ...")
    df_codon, df_aa = annotate_stalls_epa(
        df, cds_range, sequence,
        psite_offset_codons=args.psite_offset,
        basis=args.basis,
        drop_invalid=True,
    )
    logging.info(
        f"Annotation complete: {len(df_codon)} codon rows, {len(df_aa)} AA rows "
        f"(dropped {len(df) - len(df_codon)} rows where E/P/A fell outside the CDS or hit an unknown codon)"
    )

    # ------------------------------------------------------------------
    # Optionally drop stall windows whose E/P/A hits a stop codon
    # ------------------------------------------------------------------
    if args.drop_stop_codons == "True":
        n_before_stop = len(df_codon)
        codon_keep = ~df_codon[["E_codon", "P_codon", "A_codon"]].isin(STOP_CODONS).any(axis=1)
        aa_keep = ~df_aa[["E_aa", "P_aa", "A_aa"]].isin(["*"]).any(axis=1)
        df_codon = df_codon.loc[codon_keep].reset_index(drop=True)
        df_aa = df_aa.loc[aa_keep].reset_index(drop=True)
        logging.info(
            f"--drop-stop-codons: dropped {n_before_stop - len(df_codon)} stall windows "
            f"containing a stop codon; {len(df_codon)} codon rows / {len(df_aa)} AA rows remain"
        )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    codon_path = os.path.join(args.out_dir, "stall_sites_codon.csv")
    aa_path = os.path.join(args.out_dir, "stall_sites_aa.csv")
    df_codon.to_csv(codon_path, index=False)
    df_aa.to_csv(aa_path, index=False)
    logging.info(f"Saved codon-annotated stall sites to {codon_path}")
    logging.info(f"Saved AA-annotated stall sites to {aa_path}")

    # ------------------------------------------------------------------
    # Per-group background frequencies (codon + AA) for the stats script
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nBACKGROUND FREQUENCIES (per group)\n{'='*60}")
    for level, helper, alphabet, feature_col in (
        ("codon", background_codon_freq, SENSE_CODONS, "codon"),
        ("aa", background_aa_freq, AA_ORDER, "amino_acid"),
    ):
        rows = []
        for grp, grp_txs in filt_tx_dict.items():
            bg_freq, bg_counts = helper(
                grp_txs, cds_range, sequence, alphabet,
                trim_start=args.trim_start, trim_stop=args.trim_stop,
            )
            print(f"  [{grp}] ({level}) {len(grp_txs)} transcripts, {int(bg_counts.sum())} total {level}s")
            for feat in bg_freq.index:
                rows.append({
                    "group": grp,
                    feature_col: feat,
                    "bg_count": int(bg_counts[feat]),
                    "bg_freq": float(bg_freq[feat]),
                })
        bg_path = os.path.join(args.out_dir, f"per_group_background_{level}.csv")
        pd.DataFrame(rows).to_csv(bg_path, index=False)
        logging.info(f"Saved per-group {level} backgrounds to {bg_path}")


if __name__ == "__main__":
    main()
