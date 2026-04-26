#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from ribostall.global_occupancy import (
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
        description="Run statistical tests on a single global occupancy CSV "
                    "(one site, one level: codon or AA) produced by global_codon_occ.py."
    )
    p.add_argument("--input-csv", required=True,
                   help="Path to a single raw occupancy CSV "
                        "(e.g. results/global_occupancy/raw/codon_occupancy_E.csv).")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write analysis CSVs into "
                        "(e.g. results/global_occupancy/analysis/E).")
    p.add_argument("--groups", required=True,
                   help="Experimental groups, e.g. 'groupA:rep1,rep2;groupB:rep3,rep4'")
    p.add_argument("--prefix", required=True,
                   help="Output filename prefix (e.g. 'codon' or 'aa').")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # groups is a dict: group_name -> list of replicates, e.g. control_day_0 -> [control_day0_rep1, control_day0_rep2]
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

    logging.info(f"Reading {args.input_csv} ...")
    df_csv = pd.read_csv(args.input_csv)

    # Auto-detect feature column (Codon for codon-level, AminoAcid for AA-level)
    if "Codon" in df_csv.columns:
        feature_col = "Codon"
    elif "AminoAcid" in df_csv.columns:
        feature_col = "AminoAcid"
    else:
        sys.exit(f"Input CSV must contain a 'Codon' or 'AminoAcid' column: {args.input_csv}")
    
    # Sanity check: warn for replicates missing from the CSV
    declared_reps = [r for reps in groups.values() for r in reps]
    missing = [r for r in declared_reps if f"{r}_raw" not in df_csv.columns]
    if missing:
        logging.warning(f"Replicates missing from CSV: {', '.join(missing)}")



    # Build stats input dicts from saved CSV columns
    raw_for_stats = {}
    rates_for_stats = {}
    for exp in declared_reps:
        if f"{exp}_raw" in df_csv.columns:
            raw_for_stats[exp] = dict(zip(df_csv[feature_col], df_csv[f"{exp}_raw"]))
            rates_for_stats[exp] = dict(zip(df_csv[feature_col], df_csv[f"{exp}_proportion"]))

    # Transcriptome-wide frequencies (for binomial test)
    tc = dict(zip(df_csv[feature_col], df_csv["Transcriptome"]))

    def save_csv(df, name):
        path = out_dir / f"{args.prefix}_{name}"
        df.to_csv(path, index=False)
        logging.info(f"Saved {path} ({len(df)} rows)")

    # -----------------------------------------------------------------
    # Analysis 1: Within-condition binomial (Validated (AL) ~ 04/05/2026)
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"ANALYSIS 1: WITHIN-CONDITION ENRICHMENT (Binomial Test)")
    print(f"{'='*60}")

    df = within_condition_binomial_occupancy(raw_for_stats, tc, groups, rep_to_group)
    save_csv(df, "within_condition_binomial.csv")

    # -----------------------------------------------------------------
    # Analysis 2: Between-condition Wilcoxon (BWM vs Control) (Validated (AL) ~ 04/05/2026)
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"ANALYSIS 2: BETWEEN-CONDITION WILCOXON (BWM vs Control)")
    print(f"{'='*60}")

    df = between_condition_wilcoxon_occupancy(rates_for_stats, rep_to_condition)
    save_csv(df, "wilcoxon_condition.csv")

    # -----------------------------------------------------------------
    # Analysis 3: Between-timepoint (Validated (AL) ~ 04/05/2026)
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"ANALYSIS 3: BETWEEN-TIMEPOINT")
    print(f"{'='*60}")

    # --- Day 10 vs Day 0 ---
    print(f"\n--- Day 10 vs Day 0 ---")

    # 3a: Wilcoxon pooled across conditions (n=4 vs n=4)
    print("\n  3a: Wilcoxon (pooled across conditions, n=4 vs n=4)")
    df = between_timepoint_wilcoxon_occupancy(
        rates_for_stats, rep_to_timepoint, time_a="day_10", time_b="day_0")
    save_csv(df, "wilcoxon_timepoint_d10_vs_d0.csv")

    # 3b: Fisher's within each condition (pool 2 reps)
    print("\n  3b: Fisher's exact (within each condition, pooled replicates)")
    print("  WARNING: Pooling 2 biological replicates is pseudoreplication.")
    print("           P-values are anti-conservative and should be interpreted cautiously.")
    df = between_timepoint_fisher_within_condition(
        raw_for_stats, groups, rep_to_condition, rep_to_timepoint,
        time_a="day_10", time_b="day_0")
    save_csv(df, "timepoint_fisher_within_condition_d10_vs_d0.csv")

    # --- Day 10 vs Day 5 ---
    print(f"\n--- Day 10 vs Day 5 ---")

    # 3c: Wilcoxon pooled across conditions (n=4 vs n=4)
    print("\n  3c: Wilcoxon (pooled across conditions, n=4 vs n=4)")
    df = between_timepoint_wilcoxon_occupancy(
        rates_for_stats, rep_to_timepoint, time_a="day_10", time_b="day_5")
    save_csv(df, "wilcoxon_timepoint_d10_vs_d5.csv")

    # 3d: Fisher's within each condition (pool 2 reps)
    print("\n  3d: Fisher's exact (within each condition, pooled replicates)")
    print("  WARNING: Pooling 2 biological replicates is pseudoreplication.")
    print("           P-values are anti-conservative and should be interpreted cautiously.")
    df = between_timepoint_fisher_within_condition(
        raw_for_stats, groups, rep_to_condition, rep_to_timepoint,
        time_a="day_10", time_b="day_5")
    save_csv(df, "timepoint_fisher_within_condition_d10_vs_d5.csv")

    # -----------------------------------------------------------------
    # Analysis 4: Per-timepoint Fisher's (BWM vs Control at each day) (Validated (AL) ~ 04/05/2026)
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"ANALYSIS 4: PER-TIMEPOINT FISHER'S (BWM vs Control at each day)")
    print(f"{'='*60}")
    print("WARNING: Pooling 2 biological replicates is pseudoreplication.")
    print("         P-values are anti-conservative and should be interpreted cautiously.")

    df = per_timepoint_fisher_occupancy(raw_for_stats, rep_to_condition, rep_to_timepoint)
    save_csv(df, "per_timepoint_fisher.csv")

    print(f"\n{'='*60}")
    print(f"Done. Results written to: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
