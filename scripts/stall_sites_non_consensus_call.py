#!/usr/bin/env python3
"""
stall_sites_non_consensus_call.py

Data-generation half of the non-consensus stall-site pipeline.

Loads coverage, calls stall sites per replicate, annotates each stall with its
E/P/A codon and amino acid identities, and writes two tidy CSVs:

  * ``stall_sites_codon.csv`` — one row per stall, with E_codon/P_codon/A_codon
  * ``stall_sites_aa.csv``    — one row per stall, with E_aa/P_aa/A_aa

Also serialises the per-group filtered transcript sets to
``filtered_transcripts.json`` so the sister stats script can rebuild exactly
the same background distributions without re-running the filter.
"""

import argparse
import gzip
import json
import logging
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import ribopy
from ribopy import Ribo

from ribostall.stall_sites import (
    filter_tx,
    codonize_counts_cds,
    call_stalls,
    stalls_to_long_df,
    consensus_stalls_across_reps,
)
from ribostall.amino_acids import annotate_stalls_epa
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

    groups = parse_groups(args.groups)
    logging.info(f"Parsed {len(groups)} groups: {list(groups.keys())}")
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    logging.info(f"Loading coverage from {args.pickle} ...")
    with gzip.open(args.pickle, "rb") as f:
        cov = pickle.load(f)
    logging.info(f"Coverage loaded: {len(cov)} experiments")

    logging.info(f"Loading ribo object from {args.ribo} ...")
    ribo_object = Ribo(args.ribo, alias=None)

    missing = [r for rs in groups.values() for r in rs if r not in cov]
    if missing:
        print("Warning: the following replicates are missing from coverage:", ", ".join(missing))

    # ------------------------------------------------------------------
    # Per-group transcript filter + coverage density plot
    # ------------------------------------------------------------------
    n_before = len(next(iter(cov.values())))
    print(f"\n{'='*60}\nTRANSCRIPT FILTERING (per-group, no intersection)\n{'='*60}")
    print(f"Transcripts before filtering: {n_before}")
    print(f"\n  {'Replicate':<25} {'Group':<15} {'Avg cov/tx (reads/nt)':>22} {'SD':>10} {'Total coverage':>16}")
    print(f"  {'-'*25} {'-'*15} {'-'*22} {'-'*10} {'-'*16}")
    for group, reps in groups.items():
        for rep in reps:
            tx_dict = cov[rep]
            means = [np.asarray(v, float).mean() for v in tx_dict.values()]
            avg_cov = np.mean(means) if means else 0.0
            sd_cov = np.std(means) if means else 0.0
            total_cov = sum(np.asarray(v, float).sum() for v in tx_dict.values())
            print(f"  {rep:<25} {group:<15} {avg_cov:>22.4f} {sd_cov:>10.4f} {total_cov:>16,.0f}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_coverage_density(cov, groups, args.out_dir)
    logging.info(f"Saved coverage density plot to {args.out_dir}/coverage_density.png")

    filt_tx_dict = {
        group: set(filter_tx(cov, reps, min_reps=args.tx_min_reps, threshold=args.tx_threshold))
        for group, reps in groups.items()
    }
    for group, txs in filt_tx_dict.items():
        print(f"  Per-group filter [{group}]: {len(txs)} transcripts  (lost {n_before - len(txs)})")

    cov_filt = {}
    for exp, tx_dict in cov.items():
        grp = rep_to_group.get(exp)
        if grp is None:
            continue
        grp_txs = filt_tx_dict[grp]
        cov_filt[exp] = {tx: arr for tx, arr in tx_dict.items() if tx in grp_txs}

    # ------------------------------------------------------------------
    # Codonize + call stalls
    # ------------------------------------------------------------------
    logging.info("Codonizing coverage arrays ...")
    codon_cov = {
        exp: {tx: codonize_counts_cds(arr) for tx, arr in tx_dict.items()}
        for exp, tx_dict in cov_filt.items()
    }

    logging.info(
        f"Calling stall sites (min_z={args.min_z}, min_reads={args.min_reads}, "
        f"trim_start={args.trim_start}, trim_stop={args.trim_stop}) ..."
    )
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
    # Load sequences and annotate each stall with E/P/A codon + AA
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\nLOADING SEQUENCES\n{'='*60}")
    logging.info("Looking up CDS ranges ...")
    cds_range = get_cds_range_lookup(ribo_object)
    print(f"  CDS ranges: {len(cds_range)} transcripts")
    logging.info(f"Loading sequences from {args.reference} ...")
    sequence = get_sequence(ribo_object, args.reference, alias=ribopy.api.alias.apris_human_alias)
    print(f"  Sequences: {len(sequence)} transcripts")
    print(f"{'='*60}\n")

    logging.info("Annotating stalls with E/P/A codons and amino acids ...")
    df_codon, df_aa = annotate_stalls_epa(
        df, cds_range, sequence,
        psite_offset_codons=args.psite_offset,
        basis=args.basis,
        drop_invalid=True,
    )
    logging.info(
        f"Annotation complete: {len(df_codon)} codon rows, {len(df_aa)} AA rows "
        f"(dropped {len(df) - len(df_codon)} rows where E/P/A fell outside the CDS or hit stop/unknown)"
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

    filt_path = os.path.join(args.out_dir, "filtered_transcripts.json")
    with open(filt_path, "w") as f:
        json.dump({g: sorted(txs) for g, txs in filt_tx_dict.items()}, f)
    logging.info(f"Saved per-group filtered transcripts to {filt_path}")


if __name__ == "__main__":
    main()
