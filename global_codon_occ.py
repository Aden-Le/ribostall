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
from functions_folder.functions_AA import CODON2AA, AA_ORDER

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
                   help="Semicolon-separated group:rep1,rep2 definitions (required for --run-stats)")
    p.add_argument("--run-stats", action="store_true",
                   help="Run statistical tests (requires --groups)")
    return p.parse_args()


def iter_trimmed_codons(seq: str, trim_start_codons: int, trim_stop_codons: int):
    """yield (codon_str, cds_nt_idx) for trimmed cds sequence"""
    n = len(seq)
    start_nt = trim_start_codons * 3
    end_nt = n - (trim_stop_codons * 3)
    if start_nt >= end_nt:
        return
    last_full = end_nt - ((end_nt - start_nt) % 3)
    # i would be 0, 3, 6, ... relative to the trimmed CDS start
    for i in range(start_nt, last_full, 3):
        # nt for the codon triplet
        codon = seq[i:i+3]
        if len(codon) == 3 and "N" not in codon.upper():
            yield codon, i


def parse_groups(groups_arg):
    """Parse CLI group string into dict: {group_name: [rep1, rep2, ...]}"""
    groups = {}
    for block in groups_arg.split(";"):
        name, reps = block.split(":")
        groups[name] = reps.split(",")
    return groups


def aggregate_to_aa(codon_dict):
    """Aggregate a {codon: value} dict to {AA: value} using CODON2AA, skipping stop codons."""
    aa_dict = defaultdict(float)
    for codon, val in codon_dict.items():
        aa = CODON2AA.get(codon.upper())
        if aa and aa != "*":
            aa_dict[aa] += val
    return dict(aa_dict)


def main():
    args = parse_args()
    
    # Need --groups if --run-stats
    if args.run_stats and not args.groups:
        raise ValueError("--run-stats requires --groups")

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

    # Get CDS ranges and sequences
    cds_range = get_cds_range_lookup(ribo_object)
    sequence = get_sequence(ribo_object, args.reference,
                            alias=bool(args.use_human_alias))

    transcriptome_codon_counts = defaultdict(int)

    # iterate once over transcripts for background
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
    # Gets all in the transcriptome
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

    # Pre-compute per-experiment rate sums for within-replicate proportion normalisation
    exp_codon_rate_sums = {}
    for exp in codon_occ_by_exp:
        exp_codon_rate_sums[exp] = sum(
            codon_occ_by_exp[exp].get(c, 0.0) / transcriptome_codon_counts[c]
            for c in ordered_codons
            if transcriptome_codon_counts.get(c, 0) > 0
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
    aa_occ_by_exp = {}
    # Aggregate each experiment's codon counts to amino acid counts
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
    stats_dir = out_dir / "analysis"
    stats_dir.mkdir(parents=True, exist_ok=True)

    codon_path = raw_dir / "codon_occupancy.csv"
    df_codon.to_csv(codon_path, index=False)
    logging.info(f"Saved {codon_path}")

    aa_path = raw_dir / "aa_occupancy.csv"
    df_aa.to_csv(aa_path, index=False)
    logging.info(f"Saved {aa_path}")

    # =========================================================================
    # Statistical tests (if --run-stats)
    # =========================================================================
    if not args.run_stats:
        logging.info("Done (no statistical tests requested).")
        return

    # Import statiscal test functions
    from functions_folder.functions_global_occupancy import (
        within_condition_binomial_occupancy,
        between_condition_wilcoxon_occupancy,
        between_timepoint_wilcoxon_occupancy,
        between_timepoint_fisher_within_condition,
        per_timepoint_fisher_occupancy,
    )

    # Parse Groups
    groups = parse_groups(args.groups)
    logging.info(f"Parsed {len(groups)} groups: {list(groups.keys())}")

    # Build mapping dicts
    # Build dictionary mapping each replicate to its group, condition, and timepoint
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    rep_to_condition = {}
    rep_to_timepoint = {}
    for rep, grp in rep_to_group.items():
        parts = grp.split("_", 1)
        rep_to_condition[rep] = parts[0]       # "control" or "BWM"
        rep_to_timepoint[rep] = parts[1] if len(parts) > 1 else grp  # "day_0", etc.

    # Sanity check: warn if any declared replicate is absent
    all_reps = [r for reps in groups.values() for r in reps] # replicates declared in groups
    missing = [r for r in all_reps if r not in coverage_dict] # replicates missing from coverage data
    if missing:
        logging.warning(f"Replicates missing from coverage: {', '.join(missing)}")

    declared_reps = set(all_reps)

    # Pull all stats inputs from the saved CSVs — raw counts and within-rep
    # normalised proportions are already computed and stored there.
    logging.info("Reading stats inputs from saved CSVs ...")
    df_codon_csv = pd.read_csv(codon_path)
    df_aa_csv = pd.read_csv(aa_path)

    codon_raw_for_stats = {}
    codon_rates_for_stats = {}
    aa_raw_for_stats = {}
    aa_rates_for_stats = {}
    for exp in declared_reps:
        if f"{exp}_raw" in df_codon_csv.columns:
            codon_raw_for_stats[exp] = dict(zip(df_codon_csv["Codon"], df_codon_csv[f"{exp}_raw"]))
            codon_rates_for_stats[exp] = dict(zip(df_codon_csv["Codon"], df_codon_csv[f"{exp}_proportion"]))
        if f"{exp}_raw" in df_aa_csv.columns:
            aa_raw_for_stats[exp] = dict(zip(df_aa_csv["AminoAcid"], df_aa_csv[f"{exp}_raw"]))
            aa_rates_for_stats[exp] = dict(zip(df_aa_csv["AminoAcid"], df_aa_csv[f"{exp}_proportion"]))

    tc_codon = dict(zip(df_codon_csv["Codon"], df_codon_csv["Transcriptome"]))
    tc_aa = dict(zip(df_aa_csv["AminoAcid"], df_aa_csv["Transcriptome"]))

    # Save CSV Function
    def save_csv(df, name):
        path = stats_dir / name
        df.to_csv(path, index=False)
        logging.info(f"Saved {path} ({len(df)} rows)")

    # -----------------------------------------------------------------
    # Analysis 1: Within-condition binomial
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYSIS 1: WITHIN-CONDITION ENRICHMENT (Binomial Test)")
    print(f"{'='*60}")

    df = within_condition_binomial_occupancy(codon_raw_for_stats, tc_codon, groups, rep_to_group)
    save_csv(df, "codon_within_condition_binomial.csv")

    df = within_condition_binomial_occupancy(aa_raw_for_stats, tc_aa, groups, rep_to_group)
    save_csv(df, "aa_within_condition_binomial.csv")

    # -----------------------------------------------------------------
    # Analysis 2: Between-condition Wilcoxon (BWM vs Control)
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYSIS 2: BETWEEN-CONDITION WILCOXON (BWM vs Control)")
    print(f"{'='*60}")

    df = between_condition_wilcoxon_occupancy(codon_rates_for_stats, rep_to_condition)
    save_csv(df, "codon_wilcoxon_condition.csv")

    df = between_condition_wilcoxon_occupancy(aa_rates_for_stats, rep_to_condition)
    save_csv(df, "aa_wilcoxon_condition.csv")

    # -----------------------------------------------------------------
    # Analysis 3: Between-timepoint
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYSIS 3: BETWEEN-TIMEPOINT (Day 0 vs Day 10)")
    print(f"{'='*60}")

    # 3a: Wilcoxon pooled across conditions (n=4 vs n=4)
    print("\n  3a: Wilcoxon (pooled across conditions, n=4 vs n=4)")
    df = between_timepoint_wilcoxon_occupancy(codon_rates_for_stats, rep_to_timepoint)
    save_csv(df, "codon_wilcoxon_timepoint.csv")

    df = between_timepoint_wilcoxon_occupancy(aa_rates_for_stats, rep_to_timepoint)
    save_csv(df, "aa_wilcoxon_timepoint.csv")

    # 3b: Fisher's within each condition (pool 2 reps)
    print("\n  3b: Fisher's exact (within each condition, pooled replicates)")
    print("  WARNING: Pooling 2 biological replicates is pseudoreplication.")
    print("           P-values are anti-conservative and should be interpreted cautiously.")
    df = between_timepoint_fisher_within_condition(
        codon_raw_for_stats, groups, rep_to_condition, rep_to_timepoint)
    save_csv(df, "codon_timepoint_fisher_within_condition.csv")

    df = between_timepoint_fisher_within_condition(
        aa_raw_for_stats, groups, rep_to_condition, rep_to_timepoint)
    save_csv(df, "aa_timepoint_fisher_within_condition.csv")

    # -----------------------------------------------------------------
    # Analysis 4: Per-timepoint Fisher's (BWM vs Control at each day)
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYSIS 4: PER-TIMEPOINT FISHER'S (BWM vs Control at each day)")
    print(f"{'='*60}")
    print("WARNING: Pooling 2 biological replicates is pseudoreplication.")
    print("         P-values are anti-conservative and should be interpreted cautiously.")

    df = per_timepoint_fisher_occupancy(codon_raw_for_stats, rep_to_condition, rep_to_timepoint)
    save_csv(df, "codon_per_timepoint_fisher.csv")

    df = per_timepoint_fisher_occupancy(aa_raw_for_stats, rep_to_condition, rep_to_timepoint)
    save_csv(df, "aa_per_timepoint_fisher.csv")

    print(f"\n{'='*60}")
    print("All analyses complete.")
    print(f"Results saved to: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
