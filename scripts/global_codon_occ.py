#!/usr/bin/env python3
import argparse
import gzip
import logging
import pickle
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import ribopy
from ribopy import Ribo

from ribostall.sequence import get_cds_range_lookup, get_sequence
from ribostall.amino_acids import CODON2AA, AA_ORDER
from ribostall.global_occupancy import (
    iter_trimmed_codons,
    iter_trimmed_site_counts,
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

# Ribosomal sites listed in biological order E -> P -> A (5' -> 3' on mRNA).
# SITE_SHIFT is the nt offset from the P-site codon to the site codon in the
# CDS: E is 1 codon upstream (-3), A is 1 codon downstream (+3). Matches
# annotate_stalls_epa's e_rel=-1, p_rel=0, a_rel=+1.
SITES = ("E", "P", "A")
SITE_SHIFT = {"E": -3, "P": 0, "A": 3}


def build_codon_df(codon_occ_by_exp, transcriptome_codon_counts, ordered_codons):
    """Build codon-level occupancy dataframe for one site; codon_occ_by_exp is {exp: {codon: count}}."""
    total_reads_per_exp = {exp: sum(codon_occ_by_exp[exp].values())
                           for exp in codon_occ_by_exp}
    exp_codon_rate_sums = {
        exp: sum(
            codon_occ_by_exp[exp].get(c, 0.0) / transcriptome_codon_counts[c]
            for c in ordered_codons
            if transcriptome_codon_counts.get(c, 0) > 0
        )
        for exp in codon_occ_by_exp
    }

    rows = []
    for codon in ordered_codons:
        bg_count = transcriptome_codon_counts.get(codon, 0)
        row = OrderedDict()
        row["Codon"] = codon
        row["AminoAcid"] = CODON2AA.get(codon.upper(), "X")
        row["Transcriptome"] = bg_count
        for exp in sorted(codon_occ_by_exp.keys()):
            raw = codon_occ_by_exp[exp].get(codon, 0.0)
            rate = raw / bg_count if bg_count > 0 else 0.0
            rate_sum = exp_codon_rate_sums.get(exp, 0)
            proportion = rate / rate_sum if rate_sum > 0 else 0.0
            total = total_reads_per_exp.get(exp, 0)
            rpm = (raw / total) * 1e6 if total > 0 else 0.0
            row[f"{exp}_raw"] = raw
            row[f"{exp}_rate"] = rate
            row[f"{exp}_proportion"] = proportion
            row[f"{exp}_rpm"] = rpm
        rows.append(row)
    return pd.DataFrame(rows)


def build_aa_df(aa_occ_by_exp, transcriptome_aa_counts):
    """Build AA-level occupancy dataframe for one site; aa_occ_by_exp is {exp: {aa: count}}."""
    total_reads_per_exp = {exp: sum(aa_occ_by_exp[exp].values())
                           for exp in aa_occ_by_exp}
    exp_aa_rate_sums = {
        exp: sum(
            aa_occ_by_exp[exp].get(aa, 0.0) / transcriptome_aa_counts[aa]
            for aa in AA_ORDER
            if transcriptome_aa_counts.get(aa, 0) > 0
        )
        for exp in aa_occ_by_exp
    }

    rows = []
    for aa in AA_ORDER:
        bg_count = transcriptome_aa_counts.get(aa, 0)
        row = OrderedDict()
        row["AminoAcid"] = aa
        row["Transcriptome"] = bg_count
        for exp in sorted(aa_occ_by_exp.keys()):
            raw = aa_occ_by_exp[exp].get(aa, 0.0)
            rate = raw / bg_count if bg_count > 0 else 0.0
            rate_sum = exp_aa_rate_sums.get(exp, 0)
            proportion = rate / rate_sum if rate_sum > 0 else 0.0
            total = total_reads_per_exp.get(exp, 0)
            rpm = (raw / total) * 1e6 if total > 0 else 0.0
            row[f"{exp}_raw"] = raw
            row[f"{exp}_rate"] = rate
            row[f"{exp}_proportion"] = proportion
            row[f"{exp}_rpm"] = rpm
        rows.append(row)
    return pd.DataFrame(rows)


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
    p.add_argument("--out-dir", default="results/global_occupancy",
                   help="Output directory (default: results/global_occupancy)")
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

    # Loads the ribo object for UTR and CDS data
    logging.info(f"Loading ribo object from {args.ribo} ...")
    if args.use_human_alias:
        ribo_object = Ribo(args.ribo, alias=ribopy.api.alias.apris_human_alias)
    else:
        ribo_object = Ribo(args.ribo)
    logging.info("Ribo object loaded")

    # Loads the coverage dict
    logging.info(f"Loading coverage from {args.pickle} ...")
    with gzip.open(args.pickle, "rb") as f:
        coverage_dict = pickle.load(f)
    logging.info(f"Coverage loaded: {len(coverage_dict)} experiments")

    if args.groups:
        # Creates set of all replicates declared in --groups and filters coverage_dict to only those replicates
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

    # Creates dictionary of form {exp: {site: {codon: count}}} for each experiment.
    # Sites ordered E -> P -> A to match biological layout on mRNA (5' -> 3').
    # For each ribosome whose P-site is at cds_nt_idx, the P-site coverage
    # window cov[cds_nt_idx:cds_nt_idx+3] is attributed to its E, P, and A
    # codons (identities shifted by SITE_SHIFT).
    codon_occ_by_exp = {exp: {site: defaultdict(float) for site in SITES}
                        for exp in coverage_dict}

    # per-experiment counts
    logging.info("Computing per-experiment codon occupancy for sites %s ...", SITES)
    for exp, tx_map in coverage_dict.items():
        for tx, cov in tx_map.items():
            # If no coverage skip or if transcript not in CDS range skip
            if cov is None or tx not in cds_range:
                continue
            # Gets CDS sequence; iter_trimmed_site_counts handles trimming
            start, stop = cds_range[tx]
            cds_seq = sequence[tx][start:stop]
            for site in SITES:
                # Yields (site_codon, count) per P-site position: the site codon
                # is shifted from the P-site codon by SITE_SHIFT[site], while
                # the count comes from the P-site coverage window.
                for site_codon, count in iter_trimmed_site_counts(
                    cds_seq, cov, args.trim_start, args.trim_stop, SITE_SHIFT[site]
                ):
                    if count > 0:
                        codon_occ_by_exp[exp][site][site_codon] += count

    # =========================================================================
    # Build and save per-site outputs
    # =========================================================================
    # Gets all the codons in the transcriptome
    ordered_codons = sorted(set(transcriptome_codon_counts))

    logging.info("Aggregating to amino acid level ...")
    # Aggregate transcriptome codon counts to amino acid counts (shared background)
    transcriptome_aa_counts = aggregate_to_aa(dict(transcriptome_codon_counts))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for site in SITES:
        codon_site = {exp: codon_occ_by_exp[exp][site] for exp in codon_occ_by_exp}
        df_codon = build_codon_df(codon_site, transcriptome_codon_counts, ordered_codons)
        codon_path = raw_dir / f"codon_occupancy_{site}.csv"
        df_codon.to_csv(codon_path, index=False)
        logging.info(f"Saved {codon_path}")

        aa_site = {exp: aggregate_to_aa(dict(codon_site[exp])) for exp in codon_site}
        df_aa = build_aa_df(aa_site, transcriptome_aa_counts)
        aa_path = raw_dir / f"aa_occupancy_{site}.csv"
        df_aa.to_csv(aa_path, index=False)
        logging.info(f"Saved {aa_path}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
