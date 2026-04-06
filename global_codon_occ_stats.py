#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import pandas as pd

from functions_folder.functions_global_occupancy import (
    parse_groups,
    within_condition_binomial_occupancy,
    between_condition_wilcoxon_occupancy,
    between_timepoint_wilcoxon_occupancy,
    between_timepoint_fisher_within_condition,
    per_timepoint_fisher_occupancy,
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
        description="Run statistical tests on global codon/AA occupancy CSVs produced by global_codon_occ.py."
    )
    p.add_argument("--out-dir", default="global_occupancy_results",
                   help="Output directory used by global_codon_occ.py (default: global_occupancy_results). "
                        "Reads from out_dir/raw/ and writes to out_dir/analysis/.")
    p.add_argument("--groups", required=True,
                   help="Semicolon-separated group:rep1,rep2 definitions, "
                        "e.g. 'groupA:rep1,rep2;groupB:rep3,rep4'")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    stats_dir = out_dir / "analysis"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Parse groups and build mapping dicts
    groups = parse_groups(args.groups)
    logging.info(f"Parsed {len(groups)} groups: {list(groups.keys())}")
    
    # Build mapping from replicate to group, condition, and timepoint
    # rep_to_group: control_day0_rep2 -> control_day_0
    # rep_to_condition: control_day0_rep2 -> control
    # rep_to_timepoint: control_day0_rep2 -> day_0
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    rep_to_condition = {}
    rep_to_timepoint = {}
    for rep, grp in rep_to_group.items():
        parts = grp.split("_", 1)
        rep_to_condition[rep] = parts[0]
        rep_to_timepoint[rep] = parts[1] if len(parts) > 1 else grp

    # Load CSVs from raw/
    codon_path = raw_dir / "codon_occupancy.csv"
    aa_path = raw_dir / "aa_occupancy.csv"
    logging.info(f"Reading {codon_path} ...")
    df_codon_csv = pd.read_csv(codon_path)
    logging.info(f"Reading {aa_path} ...")
    df_aa_csv = pd.read_csv(aa_path)

    # Sanity check: warn for replicates missing from the CSVs
    all_reps = [r for reps in groups.values() for r in reps]
    declared_reps = set(all_reps)
    missing = [r for r in all_reps if f"{r}_raw" not in df_codon_csv.columns]
    if missing:
        logging.warning(f"Replicates missing from codon CSV: {', '.join(missing)}")

    # Build stats input dicts from saved CSV columns
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

    # Transcriptome-wide codon/AA frequencies from the CSVs (for binomial tests)
    tc_codon = dict(zip(df_codon_csv["Codon"], df_codon_csv["Transcriptome"]))
    tc_aa = dict(zip(df_aa_csv["AminoAcid"], df_aa_csv["Transcriptome"]))

    def save_csv(df, name):
        path = stats_dir / name
        df.to_csv(path, index=False)
        logging.info(f"Saved {path} ({len(df)} rows)")

    # -----------------------------------------------------------------
    # Analysis 1: Within-condition binomial (Validated (AL) ~ 04/05/2026)
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYSIS 1: WITHIN-CONDITION ENRICHMENT (Binomial Test)")
    print(f"{'='*60}")

    df = within_condition_binomial_occupancy(codon_raw_for_stats, tc_codon, groups, rep_to_group)
    save_csv(df, "codon_within_condition_binomial.csv")

    df = within_condition_binomial_occupancy(aa_raw_for_stats, tc_aa, groups, rep_to_group)
    save_csv(df, "aa_within_condition_binomial.csv")

    # -----------------------------------------------------------------
    # Analysis 2: Between-condition Wilcoxon (BWM vs Control) (Validated (AL) ~ 04/05/2026)
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYSIS 2: BETWEEN-CONDITION WILCOXON (BWM vs Control)")
    print(f"{'='*60}")

    df = between_condition_wilcoxon_occupancy(codon_rates_for_stats, rep_to_condition)
    save_csv(df, "codon_wilcoxon_condition.csv")

    df = between_condition_wilcoxon_occupancy(aa_rates_for_stats, rep_to_condition)
    save_csv(df, "aa_wilcoxon_condition.csv")

    # -----------------------------------------------------------------
    # Analysis 3: Between-timepoint (Validated (AL) ~ 04/05/2026)
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
    # Analysis 4: Per-timepoint Fisher's (BWM vs Control at each day) (Validated (AL) ~ 04/05/2026)
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
    print(f"Results saved to: {out_dir.resolve() / 'analysis'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
