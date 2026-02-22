#!/usr/bin/env python3
import argparse
import gzip
import pickle
from collections import defaultdict, OrderedDict
from pathlib import Path

import pandas as pd
import ribopy
from ribopy import Ribo

from functions import get_cds_range_lookup, get_sequence


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute global codon occupancy per experiment (CDS-only coverage)."
    )
    p.add_argument("--ribo", required=True, help="Path to .ribo file")
    p.add_argument("--pickle", required=True,
                   help="Path to gzipped pickle with coverage dict: {exp: {tx: np.ndarray (CDS coverage)}}")
    p.add_argument("--reference", required=True,
                   help="Path to reference fasta/2bit/etc used by get_sequence()")
    p.add_argument("--ofset", required=True, help="Offset applied to coverage data - P or A")
    p.add_argument("--out", default="codon_occupancy.csv",
                   help="Output CSV (default: codon_occupancy.csv)")
    p.add_argument("--trim-start", type=int, default=0,
                   help="Exclude the first N codons of the CDS (default: 0)")
    p.add_argument("--trim-stop", type=int, default=0,
                   help="Exclude the last N codons of the CDS (default: 0)")
    p.add_argument("--use-human-alias", action="store_true",
                   help="Use ribopy.api.alias.apris_human_alias when opening the Ribo file")
    return p.parse_args()

# Takes a sequence and the trim arguments
def iter_trimmed_codons(seq: str, trim_start_codons: int, trim_stop_codons: int):
    """yield (codon_str, cds_nt_idx) for trimmed cds sequence"""
    # Lenght of the seqeunce
    n = len(seq)
    # Trimming if necessary
    start_nt = trim_start_codons * 3
    end_nt = n - (trim_stop_codons * 3)
    # If trimming gets rid of seq then return
    if start_nt >= end_nt:
        return
    # Ensures that slicing maintains codon frame in 3's
    last_full = end_nt - ((end_nt - start_nt) % 3) # adjust end_nt to last full codon, if 101, then ((end_nt - start_nt) % 3) is 2, so last_full is 99
    for i in range(start_nt, last_full, 3):
        codon = seq[i:i+3] # Looks at the sequence in windows of 3 (codons)
        if len(codon) == 3 and "N" not in codon.upper(): # Quality checks codons for full length and no ambiguous bases
            yield codon, i # yields the codon and the index of the first nt of the codon in the original sequence (not trimmed), ("ATG", 0), ("GAA", 3), ("TTT", 6)


def main():
    args = parse_args()
    # ribo object
    if args.use_human_alias:
        ribo_object = Ribo(args.ribo, alias=ribopy.api.alias.apris_human_alias)
    else:
        ribo_object = Ribo(args.ribo)
    # coverage dict
    with gzip.open(args.pickle, "rb") as f:
        coverage_dict = pickle.load(f)
    
    # Get the start and stop positions of the CDS for each transcript, and the sequences of each transcript
    # Retunrs a dictionary of the cds ranges (transcript -> (start, stop)) and a dictionary of sequences (transcript -> sequence)
    cds_range = get_cds_range_lookup(ribo_object)
    sequence = get_sequence(ribo_object, args.reference,
                            alias=bool(args.use_human_alias))

    codon_occ_by_exp = {exp: defaultdict(int) for exp in coverage_dict}
    transcriptome_codon_counts = defaultdict(int)

    # iterate once over transcripts for background / Overall codon counts in the transcriptome (not weighted by coverage)
    for tx in cds_range:
        # Get the start and stop in for each of the transcript
        start, stop = cds_range[tx]
        # Extracts only the cds sequence
        cds_seq = sequence[tx][start:stop]
        for codon, _ in iter_trimmed_codons(cds_seq, args.trim_start, args.trim_stop):
            transcriptome_codon_counts[codon] += 1 # Adds one to the codon in the transcriptome counts for each codon in the cds sequence of each transcript

    # per-experiment
    for exp, tx_map in coverage_dict.items():
        # For transcript and coverage in the experiment
        for tx, cov in tx_map.items():
            # No coverage or not in cds_range, dictionary, skip
            if cov is None or tx not in cds_range:
                continue
            # Gets the start and stop indexes
            start, stop = cds_range[tx]
            # Returns the cds sequence for the transcript
            cds_seq = sequence[tx][start:stop]
            # The codon and the index of the first nt of the codon in the original sequence ("ATG", 0), ("GAA", 3), ("TTT", 6)
            for codon, cds_nt_idx in iter_trimmed_codons(cds_seq, args.trim_start, args.trim_stop):
                # Start of the coverage
                cov_start = cds_nt_idx # 0, 3, 6
                # End o f the coverage
                cov_end = cov_start + 3 # 3, 6, 9
                # Check if the coverage window is within the bounds of the coverage array for the transcript,
                if cov_end <= len(cov): # If the end of the coverage window is within the bounds of the coverage array for the transcript cov_end = 9, len(cov) = 99 (Still within CDS)
                    count = cov[cov_start:cov_end].sum() # Sums the coverage counts in the window of the codon (3 nt)
                    if count > 0:
                        codon_occ_by_exp[exp][codon] += float(count) # Adds the count to the codon occupancy for the experiment, weighted by coverage (not just presence/absence)

    # build output dataframe
    all_codons = set(transcriptome_codon_counts) # All codons in the transcriptome
    for exp in codon_occ_by_exp:
        all_codons.update(codon_occ_by_exp[exp].keys()) # Rare chance that theres a codon in the coverage that isn't in the transcriptome counts
    ordered_codons = sorted(all_codons) # Sort codons alphabetically for consistent output order

    rows = []
    for codon in ordered_codons:
        row = OrderedDict()
        row["Codon"] = codon
        row["Transcriptome"] = transcriptome_codon_counts.get(codon, 0)
        for exp in codon_occ_by_exp:
            row[exp] = codon_occ_by_exp[exp].get(codon, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"saved {out_path.resolve()}")


if __name__ == "__main__":
    main()