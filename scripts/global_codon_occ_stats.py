#!/usr/bin/env python3
import argparse
import logging
import shutil
import sys
import tempfile
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
        description="Run statistical tests on the global occupancy CSVs for ONE level "
                    "(codon or AA), processing all E/P/A sites in a single invocation. "
                    "Writes the merged analysis_corrected/ tree (the E/P/A per-site frames "
                    "concatenated with a prepended 'site' column). The per-site frames are an "
                    "internal intermediate and are NOT exported."
    )
    p.add_argument("--raw-dir", default="results/global_occupancy/raw",
                   help="Directory of raw occupancy CSVs from global_codon_occ.py "
                        "(reads {level}_occupancy_{site}.csv).")
    p.add_argument("--corrected-dir", default="results/global_occupancy/analysis_corrected",
                   help="Directory for the merged CSVs (one per analysis, with a 'site' column).")
    p.add_argument("--level", required=True, choices=["codon", "aa"],
                   help="Occupancy level to process.")
    p.add_argument("--sites", nargs="+", default=["E", "P", "A"],
                   help="Ribosome sites to process, in concatenation order for the merged tree.")
    p.add_argument("--groups", required=True,
                   help="Experimental groups, e.g. 'groupA:rep1,rep2;groupB:rep3,rep4'")
    p.add_argument("--headline-condition", default=None,
                   help="Condition treated as the headline (numerator / direction reference) in the "
                        "between-condition tests: the between-condition Wilcoxon (Analysis 2; positive "
                        "log2_FC = higher occupancy here) and the per-timepoint Fisher (Analysis 4; "
                        "positive log2 odds ratio = enriched here). Must match one of the two condition "
                        "labels (e.g. 'BWM'). Default: alphabetical (first condition is headline).")

    # Per-analysis toggles. Each takes the literal value true or false and
    # defaults to true (the analysis runs); pass e.g. --per-timepoint-fisher false
    # to skip it. The between-timepoint block (Analysis 3) is split into its two
    # sub-tests so each can be skipped independently. A skipped analysis is not
    # computed and is therefore absent from the merged analysis_corrected/ tree.
    p.add_argument("--within-condition", choices=["true", "false"], default="true",
                   help="Analysis 1: within-condition binomial occupancy. "
                        "Default: true (set false to skip).")
    p.add_argument("--between-condition-wilcoxon", choices=["true", "false"], default="true",
                   help="Analysis 2: between-condition Wilcoxon. "
                        "Default: true (set false to skip).")
    p.add_argument("--between-timepoint-wilcoxon", choices=["true", "false"], default="true",
                   help="Analysis 3a: between-timepoint Wilcoxon (pooled across conditions). "
                        "Default: true (set false to skip).")
    p.add_argument("--between-timepoint-fisher", choices=["true", "false"], default="true",
                   help="Analysis 3b: between-timepoint Fisher within each condition. "
                        "Default: true (set false to skip).")
    p.add_argument("--per-timepoint-fisher", choices=["true", "false"], default="true",
                   help="Analysis 4: per-timepoint Fisher's exact. "
                        "Default: true (set false to skip).")
    return p.parse_args()


def run_site_analyses(input_csv, out_dir, prefix, groups, rep_to_group,
                      rep_to_condition, rep_to_timepoint, headline_condition=None,
                      run_within_condition=True, run_between_condition_wilcoxon=True,
                      run_between_timepoint_wilcoxon=True, run_between_timepoint_fisher=True,
                      run_per_timepoint_fisher=True):
    """Run the selected analyses for one (site, level) CSV.

    Writes each result to ``out_dir/{prefix}_{name}`` (unchanged behaviour) and
    returns the ordered list of output basenames written, for later merging.
    Only basenames for analyses that actually ran are returned, so a skipped
    analysis is absent from both the per-site tree and the merge step.

    ``headline_condition`` sets the numerator / direction of the two
    between-condition tests (Wilcoxon Analysis 2, per-timepoint Fisher Analysis
    4); ``None`` keeps the alphabetical default (backward-compatible).

    The ``run_*`` flags toggle each analysis; all default to True (run). The
    between-timepoint block (Analysis 3) is split into its Wilcoxon and Fisher
    sub-tests so each can be skipped independently.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading {input_csv} ...")
    df_csv = pd.read_csv(input_csv)

    # Auto-detect feature column (Codon for codon-level, AminoAcid for AA-level)
    if "Codon" in df_csv.columns:
        feature_col = "Codon"
        out_feature_col = "codon"
    elif "AminoAcid" in df_csv.columns:
        feature_col = "AminoAcid"
        out_feature_col = "amino_acid"
    else:
        sys.exit(f"Input CSV must contain a 'Codon' or 'AminoAcid' column: {input_csv}")

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

    basenames = []

    def save_csv(df, name):
        basename = f"{prefix}_{name}"
        path = out_dir / basename
        df.to_csv(path, index=False)
        logging.info(f"Saved {path} ({len(df)} rows)")
        basenames.append(basename)

    # -----------------------------------------------------------------
    # Analysis 1: Within-condition binomial (Validated (AL) ~ 04/05/2026)
    # -----------------------------------------------------------------
    if run_within_condition:
        print(f"\n{'='*60}")
        print(f"ANALYSIS 1: WITHIN-CONDITION ENRICHMENT (Binomial Test)")
        print(f"{'='*60}")

        df = within_condition_binomial_occupancy(raw_for_stats, tc, groups, rep_to_group, feature_col=out_feature_col)
        save_csv(df, "within_condition_binomial.csv")
    else:
        print(f"\n{'='*60}\nANALYSIS 1: WITHIN-CONDITION ENRICHMENT (Binomial Test)  [SKIPPED]\n{'='*60}")

    # -----------------------------------------------------------------
    # Analysis 2: Between-condition Wilcoxon (BWM vs Control) (Validated (AL) ~ 04/05/2026)
    # -----------------------------------------------------------------
    if run_between_condition_wilcoxon:
        print(f"\n{'='*60}")
        print(f"ANALYSIS 2: BETWEEN-CONDITION WILCOXON (BWM vs Control)")
        print(f"{'='*60}")
        if headline_condition is not None:
            print(f"  Headline condition: {headline_condition} "
                  f"(positive log2_FC = higher occupancy in {headline_condition})")
        else:
            print("  Headline condition: alphabetical default "
                  "(positive log2_FC = higher occupancy in the first condition)")

        df = between_condition_wilcoxon_occupancy(
            rates_for_stats, rep_to_condition, feature_col=out_feature_col,
            headline_condition=headline_condition)
        save_csv(df, "wilcoxon_condition.csv")
    else:
        print(f"\n{'='*60}\nANALYSIS 2: BETWEEN-CONDITION WILCOXON (BWM vs Control)  [SKIPPED]\n{'='*60}")

    # -----------------------------------------------------------------
    # Analysis 3: Between-timepoint (Validated (AL) ~ 04/05/2026)
    # -----------------------------------------------------------------
    # The Wilcoxon and Fisher sub-tests are toggled independently
    # (run_between_timepoint_wilcoxon / run_between_timepoint_fisher); within each
    # day-pair, each sub-test runs only if its flag is on.
    if run_between_timepoint_wilcoxon or run_between_timepoint_fisher:
        print(f"\n{'='*60}")
        print(f"ANALYSIS 3: BETWEEN-TIMEPOINT")
        print(f"{'='*60}")

        # --- Day 10 vs Day 0 ---
        print(f"\n--- Day 10 vs Day 0 ---")

        # 3a: Wilcoxon pooled across conditions (n=4 vs n=4)
        if run_between_timepoint_wilcoxon:
            print("\n  3a: Wilcoxon (pooled across conditions, n=4 vs n=4)")
            df = between_timepoint_wilcoxon_occupancy(
                rates_for_stats, rep_to_timepoint, time_a="day_10", time_b="day_0",
                feature_col=out_feature_col)
            save_csv(df, "wilcoxon_timepoint_d10_vs_d0.csv")

        # 3b: Fisher's within each condition (pool 2 reps)
        if run_between_timepoint_fisher:
            print("\n  3b: Fisher's exact (within each condition, pooled replicates)")
            print("  WARNING: Pooling 2 biological replicates is pseudoreplication.")
            print("           P-values are anti-conservative and should be interpreted cautiously.")
            df = between_timepoint_fisher_within_condition(
                raw_for_stats, groups, rep_to_condition, rep_to_timepoint,
                time_a="day_10", time_b="day_0", feature_col=out_feature_col)
            save_csv(df, "timepoint_fisher_within_condition_d10_vs_d0.csv")

        # --- Day 10 vs Day 5 ---
        print(f"\n--- Day 10 vs Day 5 ---")

        # 3c: Wilcoxon pooled across conditions (n=4 vs n=4)
        if run_between_timepoint_wilcoxon:
            print("\n  3c: Wilcoxon (pooled across conditions, n=4 vs n=4)")
            df = between_timepoint_wilcoxon_occupancy(
                rates_for_stats, rep_to_timepoint, time_a="day_10", time_b="day_5",
                feature_col=out_feature_col)
            save_csv(df, "wilcoxon_timepoint_d10_vs_d5.csv")

        # 3d: Fisher's within each condition (pool 2 reps)
        if run_between_timepoint_fisher:
            print("\n  3d: Fisher's exact (within each condition, pooled replicates)")
            print("  WARNING: Pooling 2 biological replicates is pseudoreplication.")
            print("           P-values are anti-conservative and should be interpreted cautiously.")
            df = between_timepoint_fisher_within_condition(
                raw_for_stats, groups, rep_to_condition, rep_to_timepoint,
                time_a="day_10", time_b="day_5", feature_col=out_feature_col)
            save_csv(df, "timepoint_fisher_within_condition_d10_vs_d5.csv")

        # --- Day 5 vs Day 0 ---
        print(f"\n--- Day 5 vs Day 0 ---")

        # 3e: Wilcoxon pooled across conditions (n=4 vs n=4)
        if run_between_timepoint_wilcoxon:
            print("\n  3e: Wilcoxon (pooled across conditions, n=4 vs n=4)")
            df = between_timepoint_wilcoxon_occupancy(
                rates_for_stats, rep_to_timepoint, time_a="day_5", time_b="day_0",
                feature_col=out_feature_col)
            save_csv(df, "wilcoxon_timepoint_d5_vs_d0.csv")

        # 3f: Fisher's within each condition (pool 2 reps)
        if run_between_timepoint_fisher:
            print("\n  3f: Fisher's exact (within each condition, pooled replicates)")
            print("  WARNING: Pooling 2 biological replicates is pseudoreplication.")
            print("           P-values are anti-conservative and should be interpreted cautiously.")
            df = between_timepoint_fisher_within_condition(
                raw_for_stats, groups, rep_to_condition, rep_to_timepoint,
                time_a="day_5", time_b="day_0", feature_col=out_feature_col)
            save_csv(df, "timepoint_fisher_within_condition_d5_vs_d0.csv")
    else:
        print(f"\n{'='*60}\nANALYSIS 3: BETWEEN-TIMEPOINT  [SKIPPED]\n{'='*60}")

    # -----------------------------------------------------------------
    # Analysis 4: Per-timepoint Fisher's (BWM vs Control at each day) (Validated (AL) ~ 04/05/2026)
    # -----------------------------------------------------------------
    if run_per_timepoint_fisher:
        print(f"\n{'='*60}")
        print(f"ANALYSIS 4: PER-TIMEPOINT FISHER'S (BWM vs Control at each day)")
        print(f"{'='*60}")
        print("WARNING: Pooling 2 biological replicates is pseudoreplication.")
        print("         P-values are anti-conservative and should be interpreted cautiously.")
        if headline_condition is not None:
            print(f"  Headline condition: {headline_condition} "
                  f"(positive log2 odds ratio = enriched in {headline_condition})")
        else:
            print("  Headline condition: alphabetical default "
                  "(positive log2 odds ratio = enriched in the first condition)")

        df = per_timepoint_fisher_occupancy(
            raw_for_stats, rep_to_condition, rep_to_timepoint, feature_col=out_feature_col,
            headline_condition=headline_condition)
        save_csv(df, "per_timepoint_fisher.csv")
    else:
        print(f"\n{'='*60}\nANALYSIS 4: PER-TIMEPOINT FISHER'S (BWM vs Control at each day)  [SKIPPED]\n{'='*60}")

    return basenames


def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    corrected_dir = Path(args.corrected_dir)
    corrected_dir.mkdir(parents=True, exist_ok=True)

    # The per-site analysis CSVs are an internal intermediate, not an export: they
    # are written to a temp dir, re-read from disk for the byte-exact merge (the
    # disk round-trip reproduces the old 2-step pipeline — see the merge note
    # below), then discarded. Only the merged analysis_corrected/ tree is kept.
    analysis_dir = Path(tempfile.mkdtemp(prefix="occ_per_site_"))

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

    # For each site: run the 5 analyses and write the per-site CSVs. Collect the
    # set of basenames written (the same across sites) for the merge step.
    all_basenames = []
    for site in args.sites:
        input_csv = raw_dir / f"{args.level}_occupancy_{site}.csv"
        site_out_dir = analysis_dir / site

        print(f"\n{'#'*60}")
        print(f"SITE {site}  |  LEVEL {args.level}")
        print(f"{'#'*60}")

        basenames = run_site_analyses(
            input_csv, site_out_dir, args.level, groups,
            rep_to_group, rep_to_condition, rep_to_timepoint,
            headline_condition=args.headline_condition,
            run_within_condition=(args.within_condition == "true"),
            run_between_condition_wilcoxon=(args.between_condition_wilcoxon == "true"),
            run_between_timepoint_wilcoxon=(args.between_timepoint_wilcoxon == "true"),
            run_between_timepoint_fisher=(args.between_timepoint_fisher == "true"),
            run_per_timepoint_fisher=(args.per_timepoint_fisher == "true"),
        )
        if not all_basenames:
            all_basenames = basenames

    # -----------------------------------------------------------------
    # Merge the per-site analysis CSVs into analysis_corrected/ (folds the old
    # merge_global_occupancy_analysis.py step). For each analysis, RE-READ the
    # per-site CSVs from disk, prepend a 'site' column, and concatenate in
    # --sites order (E -> P -> A). Re-reading (rather than reusing the in-memory
    # frames) reproduces the old merge byte-for-byte: pandas' default CSV float
    # parser round-trips the written values identically to the 2-step pipeline.
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"MERGE: per-site analysis -> {corrected_dir}")
    print(f"{'='*60}")

    for basename in all_basenames:
        merged_parts = []
        for site in args.sites:
            part = pd.read_csv(analysis_dir / site / basename)
            part.insert(0, "site", site)
            merged_parts.append(part)
        merged = pd.concat(merged_parts, ignore_index=True)
        out_path = corrected_dir / basename
        merged.to_csv(out_path, index=False)
        logging.info(f"Wrote {out_path}  ({len(merged)} rows from {len(merged_parts)} sites)")

    # Discard the per-site intermediate; only the merged tree is exported.
    shutil.rmtree(analysis_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Done. Merged results in {corrected_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
