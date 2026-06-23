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
                   help="Condition treated as the headline (numerator / direction reference) in the "
                        "between-condition tests: the between-condition Wilcoxon (Analysis 2; positive "
                        "log2_FC = higher occupancy here) and the per-timepoint Fisher (Analysis 4; "
                        "positive log2 odds ratio = enriched here). Must match one of the two condition "
                        "labels (e.g. 'BWM'). Default: alphabetical (first condition is headline).")
    p.add_argument("--timepoints", required=True,
                   help="Comma-separated timepoint labels in chronological order (earliest first), "
                        "e.g. 'day_0,day_5,day_10'. Sets the order of the per-timepoint Fisher "
                        "(Analysis 4) and generates the later-vs-earlier between-timepoint pairs "
                        "(Analysis 3). Timepoints are NOT sorted automatically — a string sort would "
                        "place 'day_10' before 'day_5'.")

    # Per-analysis toggles. Each takes the literal value true or false and
    # defaults to true (the analysis runs); pass e.g. --per-timepoint-fisher false
    # to skip it. The between-timepoint block (Analysis 3) is split into its two
    # sub-tests so each can be skipped independently. A skipped analysis is not
    # computed and is therefore absent from the merged analysis/ tree.
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


def parse_timepoints(timepoints_arg):
    """['day_0', 'day_5', 'day_10'] from 'day_0,day_5,day_10' (order preserved)."""
    return [t.strip() for t in timepoints_arg.split(",") if t.strip()]


def timepoint_token(label):
    """'day_10' -> 'd10' (legacy short tag); any other label passes through unchanged."""
    return "d" + label[len("day_"):] if label.startswith("day_") else label


def build_timepoint_pairs(timepoint_order):
    """All later-vs-earlier (time_a, time_b, tag) pairs from a chronological list.

    For ``['day_0', 'day_5', 'day_10']`` this yields, in order,
    ``('day_10', 'day_0', 'd10_vs_d0')``, ``('day_10', 'day_5', 'd10_vs_d5')``,
    ``('day_5', 'day_0', 'd5_vs_d0')`` — the same three pairs (and order) the
    script used to hard-code. ``time_a`` is the later timepoint (direction is
    later-vs-earlier).
    """
    pairs = []
    for j in range(len(timepoint_order) - 1, 0, -1):   # later: latest index down to 1
        for i in range(j):                              # earlier: 0 .. j-1
            time_a, time_b = timepoint_order[j], timepoint_order[i]
            tag = f"{timepoint_token(time_a)}_vs_{timepoint_token(time_b)}"
            pairs.append((time_a, time_b, tag))
    return pairs


def run_site_analyses(input_csv, out_dir, prefix, groups, rep_to_group,
                      rep_to_condition, rep_to_timepoint, timepoint_order, headline_condition=None,
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

    ``timepoint_order`` is the chronological timepoint list (earliest first) that
    sets the order of the per-timepoint Fisher (Analysis 4) and generates the
    later-vs-earlier between-timepoint pairs (Analysis 3).

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

        # All later-vs-earlier day-pairs generated from --timepoints (previously
        # three hard-coded, fully-unrolled blocks). Each pair runs (a) Wilcoxon
        # pooled across conditions and (b) Fisher's within each condition. For
        # day_0,day_5,day_10 this reproduces wilcoxon_timepoint_d10_vs_d0.csv etc.
        for time_a, time_b, tag in build_timepoint_pairs(timepoint_order):
            print(f"\n--- {time_a} vs {time_b} ---")

            # Wilcoxon pooled across conditions (n=4 vs n=4)
            if run_between_timepoint_wilcoxon:
                print("\n  Wilcoxon (pooled across conditions, n=4 vs n=4)")
                df = between_timepoint_wilcoxon_occupancy(
                    rates_for_stats, rep_to_timepoint, time_a=time_a, time_b=time_b,
                    feature_col=out_feature_col)
                save_csv(df, f"wilcoxon_timepoint_{tag}.csv")

            # Fisher's within each condition (pool 2 reps)
            if run_between_timepoint_fisher:
                print("\n  Fisher's exact (within each condition, pooled replicates)")
                print("  WARNING: Pooling 2 biological replicates is pseudoreplication.")
                print("           P-values are anti-conservative and should be interpreted cautiously.")
                df = between_timepoint_fisher_within_condition(
                    raw_for_stats, groups, rep_to_condition, rep_to_timepoint,
                    time_a=time_a, time_b=time_b, feature_col=out_feature_col)
                save_csv(df, f"timepoint_fisher_within_condition_{tag}.csv")
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
            headline_condition=headline_condition, timepoints=timepoint_order)
        save_csv(df, "per_timepoint_fisher.csv")
    else:
        print(f"\n{'='*60}\nANALYSIS 4: PER-TIMEPOINT FISHER'S (BWM vs Control at each day)  [SKIPPED]\n{'='*60}")

    return basenames


def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # The per-site analysis CSVs are an internal intermediate, not an export: they
    # are written to a temp dir, re-read from disk for the byte-exact merge (the
    # disk round-trip reproduces the old 2-step pipeline — see the merge note
    # below), then discarded. Only the merged analysis/ tree is kept.
    per_site_dir = Path(tempfile.mkdtemp(prefix="occ_per_site_"))

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

    # Declared chronological timepoint order (no automatic sorting — a string sort
    # would place "day_10" before "day_5"). Drives the per-timepoint Fisher's output
    # order (Analysis 4) and the later-vs-earlier comparison pairs (Analysis 3).
    timepoint_order = parse_timepoints(args.timepoints)
    present_timepoints = set(rep_to_timepoint.values())
    missing = [tp for tp in timepoint_order if tp not in present_timepoints]
    if missing:
        sys.exit(f"--timepoints lists {missing}, not found among the --groups timepoints "
                 f"{sorted(present_timepoints)}")
    undeclared = present_timepoints - set(timepoint_order)
    if undeclared:
        logging.warning(f"Timepoints {sorted(undeclared)} are present in --groups but not in "
                        f"--timepoints; they are excluded from the timepoint analyses.")

    # For each site: run the 5 analyses and write the per-site CSVs. Collect the
    # set of basenames written (the same across sites) for the merge step.
    all_basenames = []
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
            run_between_timepoint_wilcoxon=(args.between_timepoint_wilcoxon == "true"),
            run_between_timepoint_fisher=(args.between_timepoint_fisher == "true"),
            run_per_timepoint_fisher=(args.per_timepoint_fisher == "true"),
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
