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


def iter_trimmed_codons(seq: str, trim_start_codons: int, trim_stop_codons: int):
    """yield (codon_str, cds_nt_idx) for trimmed cds sequence"""
    n = len(seq)
    start_nt = trim_start_codons * 3
    end_nt = n - (trim_stop_codons * 3)
    if start_nt >= end_nt:
        return
    last_full = end_nt - ((end_nt - start_nt) % 3)
    for i in range(start_nt, last_full, 3):
        codon = seq[i:i+3]
        if len(codon) == 3 and "N" not in codon.upper():
            yield codon, i


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
    cds_range = get_cds_range_lookup(ribo_object)
    sequence = get_sequence(ribo_object, args.reference,
                            alias=bool(args.use_human_alias))

    codon_occ_by_exp = {exp: defaultdict(int) for exp in coverage_dict}
    transcriptome_codon_counts = defaultdict(int)

    # iterate once over transcripts for background
    for tx in cds_range:
        start, stop = cds_range[tx]
        cds_seq = sequence[tx][start:stop]
        for codon, _ in iter_trimmed_codons(cds_seq, args.trim_start, args.trim_stop):
            transcriptome_codon_counts[codon] += 1

    # per-experiment counts
    for exp, tx_map in coverage_dict.items():
        for tx, cov in tx_map.items():
            if cov is None or tx not in cds_range:
                continue
            start, stop = cds_range[tx]
            cds_seq = sequence[tx][start:stop]
            for codon, cds_nt_idx in iter_trimmed_codons(cds_seq, args.trim_start, args.trim_stop):
                cov_start = cds_nt_idx
                cov_end = cov_start + 3
                if cov_end <= len(cov):
                    count = cov[cov_start:cov_end].sum()
                    if count > 0:
                        codon_occ_by_exp[exp][codon] += float(count)

    # build output dataframe
    all_codons = set(transcriptome_codon_counts)
    for exp in codon_occ_by_exp:
        all_codons.update(codon_occ_by_exp[exp].keys())
    ordered_codons = sorted(all_codons)

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