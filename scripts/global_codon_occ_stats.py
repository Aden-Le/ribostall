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
    between_condition_fisher_occupancy,
    between_timepoint_wilcoxon_occupancy,
    between_timepoint_fisher_within_condition,
    per_timepoint_fisher_occupancy,
)
from ribostall.stats_cli import (
    parse_timepoints,
    build_timepoint_pairs,
    build_rep_to_timepoint,
    validate_timepoints,
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
                    "Writes the merged analysis/ tree (the E/P/A per-site frames "
                    "concatenated with a prepended 'site' column). The per-site frames are an "
                    "internal intermediate and are NOT exported."
    )
    p.add_argument("--raw-dir", default="results/global_occupancy/raw",
                   help="Directory of raw occupancy CSVs from global_codon_occ.py "
                        "(reads {level}_occupancy_{site}.csv).")
    p.add_argument("--analysis-dir", default="results/global_occupancy/analysis",
                   help="Directory for the merged CSVs (one per analysis, with a 'site' column).")
    p.add_argument("--level", required=True, choices=["codon", "aa"],
                   help="Occupancy level to process.")
    p.add_argument("--sites", nargs="+", default=["E", "P", "A"],
                   help="Ribosome sites to process, in concatenation order for the merged tree.")
    p.add_argument("--groups", required=True,
                   help="Experimental groups, e.g. 'groupA:rep1,rep2;groupB:rep3,rep4'")
    p.add_argument("--headline-condition", default=None,
                   help="Condition treated as the headline (numerator / direction reference) in "
                        "the between-condition tests: A2 Wilcoxon (positive log2_FC = higher "
                        "occupancy here) and A3 Fisher (positive log2 odds ratio = enriched "
                        "here). Must match one of the two condition labels (e.g. 'BWM'). "
                        "Default: alphabetical (first condition).")
    p.add_argument("--timepoints", default=None,
                   help="Comma-separated timepoint labels in chronological order (earliest "
                        "first), e.g. 'day_0,day_5,day_10'. Declares the data has a timepoint "
                        "axis: A3 slices per-tp (per_timepoint_fisher.csv), A5 and A6 run per "
                        "day-pair. When absent, A5/A6 are skipped and A3 emits a single pooled "
                        "between_condition_fisher.csv. Timepoints are NOT sorted automatically.")

    # Per-analysis toggles. Each takes the literal value true or false (default
    # true); pass e.g. --within-condition false to skip it. A skipped analysis
    # is not computed and absent from analysis/. (Occupancy normalizes every
    # condition to one shared transcriptome background, so the background-aware
    # diffs the stall scripts run as A4/A7 do not apply here — hence the gap in
    # the A-numbering.)
    p.add_argument("--within-condition", choices=["true", "false"], default="true",
                   help="A1: within-condition binomial. Default: true.")
    p.add_argument("--between-condition-wilcoxon", choices=["true", "false"], default="true",
                   help="A2: between-condition Wilcoxon. Auto-skips when n<2/condition. "
                        "Default: true.")
    p.add_argument("--between-condition-fisher", choices=["true", "false"], default="true",
                   help="A3: between-condition Fisher (pooled; per-tp when --timepoints given). "
                        "Default: true.")
    p.add_argument("--between-timepoint-wilcoxon", choices=["true", "false"], default="true",
                   help="A5: between-timepoint Wilcoxon (pooled across conditions). Needs "
                        "--timepoints with >=2 timepoints. Default: true.")
    p.add_argument("--between-timepoint-fisher", choices=["true", "false"], default="true",
                   help="A6: between-timepoint Fisher within each condition. Needs --timepoints "
                        "with >=2 timepoints. Default: true.")
    return p.parse_args()


def run_site_analyses(input_csv, out_dir, prefix, groups, rep_to_group,
                      rep_to_condition, rep_to_timepoint, timepoint_order,
                      headline_condition=None,
                      run_within_condition=True,
                      run_between_condition_wilcoxon=True,
                      run_between_condition_fisher=True,
                      run_between_timepoint_wilcoxon=True,
                      run_between_timepoint_fisher=True):
    """Run the selected analyses for one (site, level) CSV.

    Writes each result to ``out_dir/{prefix}_{name}`` and returns the ordered
    list of output basenames written. Only basenames for analyses that actually
    ran are returned; a skipped analysis is absent from the per-site tree and
    the merge step.

    ``timepoint_order`` is the chronological timepoint list (earliest first);
    an empty list means no timepoints declared — A5/A6 are skipped and A3
    emits a single pooled ``between_condition_fisher.csv``.
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

    # Sanity check: every declared replicate must have its columns in the CSV.
    # A missing replicate means --groups disagrees with the raw occupancy CSV
    # (wrong group spec, wrong --raw-dir, or a typo), so fail fast rather than
    # silently running the stats on a subset of the declared replicates.

    # List of all declared replicates
    declared_reps = [r for reps in groups.values() for r in reps]
    # Any declared replicate whose columns are absent from the CSV
    missing = [r for r in declared_reps if f"{r}_raw" not in df_csv.columns]
    if missing:
        sys.exit(
            f"Replicates declared in --groups are missing from {input_csv}: "
            f"{', '.join(missing)}. Check --groups against the raw occupancy CSV columns."
        )

    # Build stats input dicts from saved CSV columns. Every declared rep is
    # guaranteed present (checked above), so no per-rep column guard is needed.
    raw_for_stats = {}
    rates_for_stats = {}
    for rep in declared_reps:
        # Creates a dictionary in dictionary {rep1: {CAG: "raw_value"}}
        raw_for_stats[rep] = dict(zip(df_csv[feature_col], df_csv[f"{rep}_raw"]))
        rates_for_stats[rep] = dict(zip(df_csv[feature_col], df_csv[f"{rep}_proportion"]))

    # Transcriptome-wide frequencies (for binomial test)
    tc = dict(zip(df_csv[feature_col], df_csv["Transcriptome"]))

    # Maps replicates to condition
    cond_reps = {}
    for rep in rates_for_stats:
        cond = rep_to_condition[rep]
        if cond not in cond_reps:
            cond_reps[cond] = []
        cond_reps[cond].append(rep)

    # Need to have guard, because might not have timepoint
    # If timepoint, maps the timepoint to the replicate
    tp_reps = {}
    for rep in rates_for_stats:
        tp = rep_to_timepoint.get(rep)
        if tp:
            if tp not in tp_reps:
                tp_reps[tp] = []
            tp_reps[tp].append(rep)

    basenames = []

    def save_csv(df, name):
        basename = f"{prefix}_{name}"
        path = out_dir / basename
        df.to_csv(path, index=False)
        logging.info(f"Saved {path} ({len(df)} rows)")
        basenames.append(basename)

    # -----------------------------------------------------------------
    # A1: Within-condition binomial
    # -----------------------------------------------------------------
    if run_within_condition:
        print(f"\n{'='*60}")
        print(f"A1: WITHIN-CONDITION ENRICHMENT (Binomial Test)")
        print(f"{'='*60}")
        df = within_condition_binomial_occupancy(
            raw_for_stats, tc, groups, rep_to_group, feature_col=out_feature_col)
        save_csv(df, "within_condition_binomial.csv")
    else:
        print(f"\n{'='*60}\nA1: WITHIN-CONDITION ENRICHMENT  [SKIPPED]\n{'='*60}")

    # -----------------------------------------------------------------
    # A2: Between-condition Wilcoxon (auto-skips when n<2/condition)
    # -----------------------------------------------------------------
    if run_between_condition_wilcoxon:
        print(f"\n{'='*60}")
        print(f"A2: BETWEEN-CONDITION WILCOXON")
        print(f"{'='*60}")
        a2_feasible = (
            len(cond_reps) == 2
            and all(len(v) >= 2 for v in cond_reps.values())
        )
        if not a2_feasible:
            print("  [SKIPPED — fewer than 2 replicates in at least one condition]")
        else:
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
        print(f"\n{'='*60}\nA2: BETWEEN-CONDITION WILCOXON  [SKIPPED]\n{'='*60}")

    # -----------------------------------------------------------------
    # A3: Between-condition Fisher
    #   With --timepoints: per-timepoint Fisher → per_timepoint_fisher.csv
    #   Pools the replicates in each timepoint and compares it to the opposing condition
    #   Without --timepoints: pooled Fisher → between_condition_fisher.csv
    #   Pools all replicates in a condition then compares
    # -----------------------------------------------------------------
    if run_between_condition_fisher:
        if timepoint_order:
            print(f"\n{'='*60}")
            print(f"A3: PER-TIMEPOINT FISHER'S (between-condition at each timepoint)")
            print(f"{'='*60}")
            print("WARNING: Pooling biological replicates is pseudoreplication.")
            print("         P-values are anti-conservative and should be interpreted cautiously.")
            if headline_condition is not None:
                print(f"  Headline condition: {headline_condition} "
                      f"(positive log2 odds ratio = enriched in {headline_condition})")
            else:
                print("  Headline condition: alphabetical default "
                      "(positive log2 odds ratio = enriched in the first condition)")
            df = per_timepoint_fisher_occupancy(
                raw_for_stats, rep_to_condition, rep_to_timepoint,
                feature_col=out_feature_col,
                headline_condition=headline_condition,
                timepoints=timepoint_order)
            save_csv(df, "per_timepoint_fisher.csv")
        else:
            print(f"\n{'='*60}")
            print(f"A3: BETWEEN-CONDITION FISHER'S (pooled, no timepoints declared)")
            print(f"{'='*60}")
            print("WARNING: Pooling biological replicates is pseudoreplication.")
            print("         P-values are anti-conservative and should be interpreted cautiously.")
            if headline_condition is not None:
                print(f"  Headline condition: {headline_condition} "
                      f"(positive log2 odds ratio = enriched in {headline_condition})")
            else:
                print("  Headline condition: alphabetical default "
                      "(positive log2 odds ratio = enriched in the first condition)")
            df = between_condition_fisher_occupancy(
                raw_for_stats, rep_to_condition,
                feature_col=out_feature_col,
                headline_condition=headline_condition)
            save_csv(df, "between_condition_fisher.csv")
    else:
        print(f"\n{'='*60}\nA3: BETWEEN-CONDITION FISHER'S  [SKIPPED]\n{'='*60}")

    # -----------------------------------------------------------------
    # A5: Between-timepoint Wilcoxon (pooled across conditions)
    #   Gated on --timepoints with ≥2 timepoints declared.
    # -----------------------------------------------------------------
    if run_between_timepoint_wilcoxon:
        if len(timepoint_order) < 2:
            print(f"\n{'='*60}\n"
                  f"A5: BETWEEN-TIMEPOINT WILCOXON  "
                  f"[SKIPPED — --timepoints not declared or fewer than 2]\n"
                  f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"A5: BETWEEN-TIMEPOINT WILCOXON (pooled across conditions)")
            print(f"{'='*60}")
            for time_a, time_b, tag in build_timepoint_pairs(timepoint_order):
                reps_a = tp_reps.get(time_a, [])
                reps_b = tp_reps.get(time_b, [])
                if len(reps_a) < 2 or len(reps_b) < 2:
                    print(f"\n  --- {time_a} vs {time_b}: "
                          f"[SKIPPED — fewer than 2 replicates in one timepoint]")
                    continue
                print(f"\n  --- {time_a} vs {time_b} ---")
                df = between_timepoint_wilcoxon_occupancy(
                    rates_for_stats, rep_to_timepoint, time_a=time_a, time_b=time_b,
                    feature_col=out_feature_col)
                save_csv(df, f"wilcoxon_timepoint_{tag}.csv")
    else:
        print(f"\n{'='*60}\nA5: BETWEEN-TIMEPOINT WILCOXON  [SKIPPED]\n{'='*60}")

    # -----------------------------------------------------------------
    # A6: Between-timepoint Fisher within each condition
    #   Gated on --timepoints with ≥2 timepoints declared.
    # -----------------------------------------------------------------
    if run_between_timepoint_fisher:
        if len(timepoint_order) < 2:
            print(f"\n{'='*60}\n"
                  f"A6: BETWEEN-TIMEPOINT FISHER  "
                  f"[SKIPPED — --timepoints not declared or fewer than 2]\n"
                  f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"A6: BETWEEN-TIMEPOINT FISHER (within each condition)")
            print(f"{'='*60}")
            print("WARNING: Pooling biological replicates is pseudoreplication.")
            print("         P-values are anti-conservative and should be interpreted cautiously.")
            for time_a, time_b, tag in build_timepoint_pairs(timepoint_order):
                print(f"\n  --- {time_a} vs {time_b} ---")
                df = between_timepoint_fisher_within_condition(
                    raw_for_stats, groups, rep_to_condition, rep_to_timepoint,
                    time_a=time_a, time_b=time_b, feature_col=out_feature_col)
                save_csv(df, f"timepoint_fisher_within_condition_{tag}.csv")
    else:
        print(f"\n{'='*60}\nA6: BETWEEN-TIMEPOINT FISHER  [SKIPPED]\n{'='*60}")

    return basenames


def main():
    args = parse_args()

    # Input directory
    raw_dir = Path(args.raw_dir)
    # Output directory
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # The per-site analysis CSVs are an intermediate, not an export: they
    # are written to a temp dir, re-read from disk for the byte-exact merge
    per_site_dir = Path(tempfile.mkdtemp(prefix="occ_per_site_"))

    groups = parse_groups(args.groups)
    logging.info(f"Parsed {len(groups)} groups: {list(groups.keys())}")

    # The reverse of the groups mapping: rep -> group
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    rep_to_condition = {}
    rep_to_timepoint = {}

    # Timepoint handling: optional. Only build rep_to_timepoint when --timepoints
    # is declared. If any group name contains an underscore
    if args.timepoints is None:
        # Exits because of syntax violation, could have timepoints
        if any("_" in grp for grp in groups):
            sys.exit(
                "Group names appear to carry timepoints (they contain '_'), but "
                "--timepoints was not declared. Pass --timepoints with the "
                "chronological order, e.g. --timepoints day_0,day_5,day_10."
            )
        # Leaves blank
        timepoint_order = []
        # Copy of the rep_to_group mapping
        rep_to_condition = dict(rep_to_group) 
        logging.info("--timepoints not declared: A5/A6 skipped; A3 runs as pooled "
                     "between-condition Fisher.")
    else:
        # Will create a timepoint order list
        timepoint_order = parse_timepoints(args.timepoints)
        # Will populate rep_to_condition with the condition without the timepoint
        for rep, grp in rep_to_group.items():
            # Split removes the timepoint
            rep_to_condition[rep] = grp.split("_", 1)[0]
        # Maps the rep to its timepoint
        rep_to_timepoint = build_rep_to_timepoint(rep_to_group)
        try:
            # Checks to see if the declared timepoints are valid
            undeclared = validate_timepoints(timepoint_order, rep_to_timepoint)
        except ValueError as e:
            # Reports if a declared timepoint is missing
            sys.exit(str(e))
        if undeclared:
            # Reports if a existing timepoint is not declared
            logging.warning(f"Timepoints {sorted(undeclared)} are present in --groups but not "
                            f"in --timepoints; they are excluded from the timepoint analyses.")

    # For each site: run the analyses and write the per-site CSVs. Collect the
    # set of basenames written (the same across sites) for the merge step.
    all_basenames = []
    # For EPA
    for site in args.sites:
        input_csv = raw_dir / f"{args.level}_occupancy_{site}.csv"
        site_out_dir = per_site_dir / site

        print(f"\n{'#'*60}")
        print(f"SITE {site}  |  LEVEL {args.level}")
        print(f"{'#'*60}")

        basenames = run_site_analyses(
            input_csv, site_out_dir, args.level, groups,
            rep_to_group, rep_to_condition, rep_to_timepoint, timepoint_order,
            headline_condition=args.headline_condition,
            run_within_condition=(args.within_condition == "true"),
            run_between_condition_wilcoxon=(args.between_condition_wilcoxon == "true"),
            run_between_condition_fisher=(args.between_condition_fisher == "true"),
            run_between_timepoint_wilcoxon=(args.between_timepoint_wilcoxon == "true"),
            run_between_timepoint_fisher=(args.between_timepoint_fisher == "true"),
        )
        if not all_basenames:
            all_basenames = basenames

    # -----------------------------------------------------------------
    # Merge the per-site analysis CSVs into analysis/ (folds the old
    # merge_global_occupancy_analysis.py step). For each analysis, RE-READ the
    # per-site CSVs from disk, prepend a 'site' column, and concatenate in
    # --sites order (E -> P -> A). Re-reading (rather than reusing the in-memory
    # frames) reproduces the old merge byte-for-byte: pandas' default CSV float
    # parser round-trips the written values identically to the 2-step pipeline.
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"MERGE: per-site analysis -> {analysis_dir}")
    print(f"{'='*60}")

    for basename in all_basenames:
        merged_parts = []
        for site in args.sites:
            part = pd.read_csv(per_site_dir / site / basename)
            part.insert(0, "site", site)
            merged_parts.append(part)
        merged = pd.concat(merged_parts, ignore_index=True)
        out_path = analysis_dir / basename
        merged.to_csv(out_path, index=False)
        logging.info(f"Wrote {out_path}  ({len(merged)} rows from {len(merged_parts)} sites)")

    # Discard the per-site intermediate; only the merged tree is exported.
    shutil.rmtree(per_site_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Done. Merged results in {analysis_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":

    main()
