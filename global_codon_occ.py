#!/usr/bin/env python3
import argparse
import gzip
import logging
import pickle
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import ribopy
from ribopy import Ribo

from functions_folder.functions import get_cds_range_lookup, get_sequence
from functions_folder.functions_AA import CODONfrom functions_folder.functions_global_occupancy import (
    iter_trimmed_codons,
    parse_groups,
    aggregate_to_aa,
)

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute global codon and amino acid occupancy per experiment (CDS-only coverage)."
    )
    p.add_argument("--ribo", required=True, help="Path to .ribo file")
    p.add_argument("--pickle", required=True,
                   help="Path to gzipped pickle with coverage dict: {exp: {tx: np.ndarray (CDS coverage)}}")
    p.add_argument("--reference", required=True,
                   help="Path to reference fasta/2bit/etc used by get_sequence()")
    p.add_argument("--ofset", required=True, help="Offset applied to coverage data - P or A")
    p.add_argument("--out-dir", default="global_occupancy_results",
                   help="Output directory (default: global_occupancy_results)")
    p.add_argument("--trim-start", type=int, default=0,
                   help="Exclude the first N codons of the CDS (default: 0)")
    p.add_argument("--trim-stop", type=int, default=0,
                   help="Exclude the last N codons of the CDS (default: 0)")
    p.add_argument("--use-human-alias", action="store_true",
                   help="Use ribopy.api.alias.apris_human_alias when opening the Ribo file")
    p.add_argument("--groups",
                   help="Semicolon-separated group:rep1,rep2 definitions, "
                        "e.g. 'groupA:rep1,rep2;groupB:rep3,rep4' "
                        "(used to filter coverage_dict to declared replicates only)")
    return p.parse_args()


def main():
    args = parse_args()

    # ribo object for UTR and CDS data
    logging.info(f"Loading ribo object from {args.ribo} ...")
    if args.use_human_alias:
        ribo_object = Ribo(args.ribo, alias=ribopy.api.alias.apris_human_alias)
    else:
        ribo_object = Ribo(args.ribo)
    logging.info("Ribo object loaded")

    # coverage dict
    logging.info(f"Loading coverage from {args.pickle} ...")
    with gzip.open(args.pickle, "rb") as f:
        coverage_dict = pickle.load(f)
    logging.info(f"Coverage loaded: {len(coverage_dict)} experiments")

    if args.groups:
        declared_reps = {r for reps in parse_groups(args.groups).values() for r in reps}
        coverage_dict = {k: v for k, v in coverage_dict.items() if k in declared_reps}
        logging.info(f"Filtered to {len(coverage_dict)} experiments matching --groups")

    # Get CDS ranges and sequences
    cds_range = get_cds_range_lookup(ribo_object)
    sequence = get_sequence(ribo_object, args.reference,
                            alias=bool(args.use_human_alias))

    transcriptome_codon_counts = defaultdict(int)

    # iterate once over transcripts for background while applying trimming
    logging.info("Computing transcriptome codon counts (background) ...")
    for tx in cds_range:
        start, stop = cds_range[tx]
        cds_seq = sequence[tx][start:stop]
        for codon, _ in iter_trimmed_codons(cds_seq, args.trim_start, args.trim_stop):
            transcriptome_codon_counts[codon] += 1

    # Creates dictionary of form {exp: {codon: count}} for each experiment
    codon_occ_by_exp = {exp: defaultdict(float) for exp in coverage_dict}

    # per-experiment counts
    logging.info("Computing per-experiment codon occupancy ...")
    for exp, tx_map in coverage_dict.items():
        for tx, cov in tx_map.items():
            # If no coverage skip or if transcript not in CDS range skip
            if cov is None or tx not in cds_range:
                continue
            # Gets CDS sequence and iterates over codons, applying trimming if specified
            start, stop = cds_range[tx]
            cds_seq = sequence[tx][start:stop]
            # Returns the 3 nt and their index relative to the CDS start (0-based) for each codon in the trimmed CDS sequence
            for codon, cds_nt_idx in iter_trimmed_codons(cds_seq, args.trim_start, args.trim_stop):
                cov_start = cds_nt_idx
                cov_end = cov_start + 3
                if cov_end <= len(cov):
                    # Sums the coverage for the 3 nucleotides of the codon
                    count = cov[cov_start:cov_end].sum()
                    if count > 0:
                        codon_occ_by_exp[exp][codon] += float(count)

    # =========================================================================
    # Build codon-level output
    # =========================================================================
    # Gets all the codons in the transcriptome
    all_codons = set(transcriptome_codon_counts)
    # Gets all codons observed in any experiment
    for exp in codon_occ_by_exp:
        all_codons.update(codon_occ_by_exp[exp].keys())
    # Filter out stop codons and sorts codons alphabetically
    all_codons = {c for c in all_codons if CODON2AA.get(c.upper(), "*") != "*"}
    ordered_codons = sorted(all_codons)

    # Compute total reads per experiment (for RPM normalization)
    total_reads_per_exp = {}
    for exp in codon_occ_by_exp:
        total_reads_per_exp[exp] = sum(codon_occ_by_exp[exp].values())

    # Pre-compute per-experiment rate sums for within-replicate proportion normalization
    exp_codon_rate_sums = {}
    for exp in codon_occ_by_exp:
        exp_codon_rate_sums[exp] = sum(
            codon_occ_by_exp[exp].get(c, 0.0) / transcriptome_codon_counts[c]
            for c in ordered_codons
            if transcriptome_codon_counts.get(c, 0) > 0 # if a codon is not present in the transcriptome, skip it for rate sum to avoid division by zero
        )

    # Codon occupancy CSV: raw counts + per-instance rate + within-rep proportion + RPM
    # Centered around codons
    codon_rows = []
    for codon in ordered_codons:
        row = OrderedDict()
        row["Codon"] = codon
        row["AminoAcid"] = CODON2AA.get(codon.upper(), "X")
        row["Transcriptome"] = transcriptome_codon_counts.get(codon, 0)
        bg_count = transcriptome_codon_counts.get(codon, 0)
        for exp in sorted(codon_occ_by_exp.keys()):
            raw = codon_occ_by_exp[exp].get(codon, 0.0)
            rate = raw / bg_count if bg_count > 0 else 0.0
            rate_sum = exp_codon_rate_sums.get(exp, 0)
            proportion = rate / rate_sum if rate_sum > 0 else 0.0
            total = total_reads_per_exp.get(exp, 1)
            rpm = (raw / total) * 1e6 if total > 0 else 0.0
            row[f"{exp}_raw"] = raw
            row[f"{exp}_rate"] = rate
            row[f"{exp}_proportion"] = proportion
            row[f"{exp}_rpm"] = rpm
        codon_rows.append(row)
    df_codon = pd.DataFrame(codon_rows)

    # =========================================================================
    # Build amino acid-level output
    # =========================================================================
    logging.info("Aggregating to amino acid level ...")
    # Aggregate transcriptome codon counts to amino acid counts for background
    transcriptome_aa_counts = aggregate_to_aa(dict(transcriptome_codon_counts))

    # Aggregate each experiment's codon counts to amino acid counts
    aa_occ_by_exp = {}
    for exp in codon_occ_by_exp:
        aa_occ_by_exp[exp] = aggregate_to_aa(dict(codon_occ_by_exp[exp]))

    # Pre-compute per-experiment AA rate sums for within-replicate proportion normalisation
    exp_aa_rate_sums = {}
    for exp in aa_occ_by_exp:
        exp_aa_rate_sums[exp] = sum(
            aa_occ_by_exp[exp].get(aa, 0.0) / transcriptome_aa_counts[aa]
            for aa in AA_ORDER
            if transcriptome_aa_counts.get(aa, 0) > 0
        )

    # Builds Dataframe with rows for each amino acid
    aa_rows = []
    for aa in AA_ORDER:
        row = OrderedDict()
        row["AminoAcid"] = aa
        row["Transcriptome"] = transcriptome_aa_counts.get(aa, 0)
        bg_count = transcriptome_aa_counts.get(aa, 0)
        for exp in sorted(aa_occ_by_exp.keys()):
            raw = aa_occ_by_exp[exp].get(aa, 0.0)
            rate = raw / bg_count if bg_count > 0 else 0.0
            rate_sum = exp_aa_rate_sums.get(exp, 0)
            proportion = rate / rate_sum if rate_sum > 0 else 0.0
            total = total_reads_per_exp.get(exp, 1)
            rpm = (raw / total) * 1e6 if total > 0 else 0.0
            row[f"{exp}_raw"] = raw
            row[f"{exp}_rate"] = rate
            row[f"{exp}_proportion"] = proportion
            row[f"{exp}_rpm"] = rpm
        aa_rows.append(row)
    df_aa = pd.DataFrame(aa_rows)

    # =========================================================================
    # Save base CSVs
    # =========================================================================
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    codon_path = raw_dir / "codon_occupancy.csv"
    df_codon.to_csv(codon_path, index=False)
    logging.info(f"Saved {codon_path}")

    aa_path = raw_dir / "aa_occupancy.csv"
    df_aa.to_csv(aa_path, index=False)
    logging.info(f"Saved {aa_path}")

    logging.info("Done.")
on_raw_for_stats, rep_to_condition, rep_to_timepoint)
    save_csv(df, "codon_per_timepoint_fisher.csv")

    df = per_timepoint_fisher_occupancy(aa_raw_for_stats, rep_to_condition, rep_to_timepoint)
    save_csv(df, "aa_per_timepoint_fisher.csv")

    print(f"\n{'='*60}")
    print("All analyses complete.")
    print(f"Results saved to: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
