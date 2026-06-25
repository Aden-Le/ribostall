1#!/usr/bin/env python3
"""
stall_sites_non_consensus_stats.py

Statistical half of the non-consensus stall-site pipeline. Consumes ONE of the
two CSVs emitted by ``stall_sites_non_consensus.py``:

  * ``stall_sites_codon.csv`` → codon-level enrichment (alphabet = 61 sense codons)
  * ``stall_sites_aa.csv``    → amino-acid-level enrichment (alphabet = AA_ORDER)

Level is auto-detected from the input columns.

This pipeline keeps each biological replicate as an INDEPENDENT observation, so
it runs ONLY the two analyses that never collapse replicate counts:

  * A2 — between-condition Wilcoxon rank-sum (per-replicate frequencies,
         control vs treatment)
  * A5 — between-timepoint Wilcoxon rank-sum (per-replicate frequencies, pooled
         across conditions within each timepoint), one CSV per later-vs-earlier
         day-pair

The count-collapsing tests — within-condition binomial, between-condition /
per-timepoint Fisher, the background-aware diffs, and the between-timepoint
Fisher — pool biological replicates into a pseudoreplicate, which is invalid on
per-replicate data. They live exclusively in the *consensus* stats scripts
(``stall_sites_consensus_union_stats.py`` for A1/A4/A7 and
``stall_sites_consensus_intersection_stats.py`` for A1/A3/A6), which run them at
n=1 per (condition, timepoint) cell on reproducibility-filtered sets (pooling a
single set is a no-op). The split is structural — this script contains no
count-collapsing code at all — so pseudoreplication is impossible by
construction. See the README's test-by-pipeline division section and
STATS_UNIFICATION_PLAN.md (Decisions 11 + 12).

``--timepoints`` is OPTIONAL: declare it only when the groups carry timepoints
(e.g. ``BWM_day_0``). With it, A5 runs over every later-vs-earlier day-pair;
without it, A5 is skipped and only A2 runs. Neither Wilcoxon reads a background,
so this script takes no ``--background`` argument.

Run it twice, once per CSV, to get codon-level and AA-level outputs side by
side. Output filenames are suffixed with ``_codon`` or ``_aa`` accordingly.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from ribostall.enrichment import (
    between_condition_wilcoxon,
    between_timepoint_wilcoxon,
)
from ribostall.stats_cli import (
    parse_groups,
    parse_timepoints,
    build_timepoint_pairs,
    build_rep_to_timepoint,
    validate_timepoints,
    detect_level,
    build_replicate_counts,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the two per-replicate Wilcoxon tests (A2 between-condition, "
                    "A5 between-timepoint) on a per-stall CSV produced by "
                    "stall_sites_non_consensus.py."
    )
    parser.add_argument("--stall-sites", required=True,
                        help="Path to stall_sites_codon.csv or stall_sites_aa.csv")
    parser.add_argument("--groups", required=True,
                        help="Experimental groups, e.g. 'groupA:rep1,rep2;groupB:rep3,rep4'")
    parser.add_argument("--out-dir", default="results/stall_sites_non_consensus/analysis",
                        help="Output directory for enrichment CSVs")
    parser.add_argument("--headline-condition", default=None,
                        help="Condition treated as the headline (numerator / direction reference) in "
                             "the between-condition Wilcoxon (A2): a positive log2_FC means a higher "
                             "per-replicate stall frequency in this condition. Must match one of the "
                             "two condition labels (e.g. 'BWM'). Default: alphabetical (first "
                             "condition is headline). The between-timepoint Wilcoxon (A5) is fixed to "
                             "later-vs-earlier and ignores this flag.")
    parser.add_argument("--timepoints", default=None,
                        help="OPTIONAL. Comma-separated timepoint labels in chronological order "
                             "(earliest first), e.g. 'day_0,day_5,day_10'. Declare only when the "
                             "--groups carry timepoints (e.g. BWM_day_0). When given (>=2 timepoints), "
                             "A5 runs over each later-vs-earlier day-pair; when absent, A5 is skipped "
                             "and only A2 runs. Timepoints are NOT sorted automatically — a string "
                             "sort would place 'day_10' before 'day_5'.")

    # Per-analysis toggles. Each takes the literal value true or false and
    # defaults to true (the analysis runs); pass e.g.
    # --between-timepoint-wilcoxon false to skip it. A skipped analysis is
    # announced and writes no output CSV.
    parser.add_argument("--between-condition-wilcoxon", choices=["true", "false"], default="true",
                        help="A2: between-condition Wilcoxon. "
                             "Default: true (set false to skip).")
    parser.add_argument("--between-timepoint-wilcoxon", choices=["true", "false"], default="true",
                        help="A5: between-timepoint Wilcoxon (pooled across conditions). Requires "
                             "--timepoints with >=2 timepoints. Default: true (set false to skip).")
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Load stall sites and detect granularity
    # --------------------------------------------------------------
    stall_path = Path(args.stall_sites)
    logging.info(f"Loading stall sites from {stall_path} ...")
    df = pd.read_csv(stall_path)
    # level = "codon" or "aa"; site_cols = (E_col, P_col, A_col); alphabet = list of codons or AAs; feature_col = "codon" or "amino_acid"
    level, site_cols, alphabet, feature_col = detect_level(df)
    suffix = level  # "codon" or "aa"
    logging.info(f"Detected level: {level} ({len(alphabet)} categories; feature column '{feature_col}')")

    # --------------------------------------------------------------
    # Groups, condition, and (optional) timepoint mappings
    # --------------------------------------------------------------
    groups = parse_groups(args.groups)
    # Map reps to groups
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    # Condition is the part of the group name before the first underscore
    # (BWM_day_0 → BWM); a flat group name (BWM) is its own condition.
    rep_to_condition = {rep: grp.split("_", 1)[0] for rep, grp in rep_to_group.items()}

    # --timepoints is OPTIONAL: build + validate the timepoint mapping only when
    # given. Declared chronological order (no automatic sorting — a string sort
    # would place "day_10" before "day_5") drives the A5 day-pair generation.
    timepoint_order = None
    rep_to_timepoint = None
    if args.timepoints:
        timepoint_order = parse_timepoints(args.timepoints)
        rep_to_timepoint = build_rep_to_timepoint(rep_to_group)
        try:
            undeclared = validate_timepoints(timepoint_order, rep_to_timepoint)
        except ValueError as e:
            sys.exit(str(e))
        if undeclared:
            logging.warning(f"Timepoints {sorted(undeclared)} are present in --groups but not in "
                            f"--timepoints; they are excluded from the timepoint analyses.")

    # --------------------------------------------------------------
    # Per-replicate counts
    # --------------------------------------------------------------
    # Returns the amino acid or codon stall site count for each replicate at each site (E, P, A).
    replicate_counts = build_replicate_counts(df, site_cols, alphabet)

    # Print total counts per site across replicates (for sanity check)
    print(f"\n{'='*60}\nTOTAL STALL SITE COUNTS PER SITE (summed across replicates)\n{'='*60}")
    for rep, site_counts in replicate_counts.items():
        totals = {s: int(site_counts[s].sum()) for s in ("E", "P", "A")}
        print(f"  [{rep}] counts per site: {totals}")

    # Decision 9 per-(condition,timepoint) cell gate: the Wilcoxons need >=2
    # replicates per cell. A cell is a group — (condition, timepoint) for a
    # timepoint design, just the condition for a flat one. Non-consensus always
    # has >=2 reps/cell, so this is a guard against misuse (e.g. n=1 consensus
    # data fed here), not an expected skip.
    min_reps_per_cell = min((len(reps) for reps in groups.values()), default=0)
    wilcoxon_feasible = min_reps_per_cell >= 2

    # Each analysis below runs only when its toggle is on (the default) AND it is
    # feasible. A skipped analysis is announced and writes no CSV. saved_paths
    # collects the outputs actually written, in run order, for the closing summary.
    saved_paths = []

    # --------------------------------------------------------------
    # A2: Between-condition Wilcoxon
    # --------------------------------------------------------------
    if args.between_condition_wilcoxon == "true" and wilcoxon_feasible:
        print(f"\n{'='*60}\nANALYSIS A2: BETWEEN-CONDITION WILCOXON\n{'='*60}")
        if args.headline_condition is not None:
            print(f"  Headline condition: {args.headline_condition} "
                  f"(positive log2_FC = higher per-replicate frequency in {args.headline_condition})")
        else:
            print("  Headline condition: alphabetical default "
                  "(positive log2_FC = higher per-replicate frequency in the first condition)")
        df_wilcox = between_condition_wilcoxon(
            replicate_counts, rep_to_condition, feature_col=feature_col,
            headline_condition=args.headline_condition,
        )
        n_sig = (df_wilcox["p_adj"] < 0.05).sum() if not df_wilcox.empty else 0
        print(f"  Tests: {len(df_wilcox)}  |  Significant (p_adj<0.05): {n_sig}")
        wilcox_path = out_dir / f"between_condition_wilcoxon_{suffix}.csv"
        df_wilcox.to_csv(wilcox_path, index=False)
        saved_paths.append(wilcox_path)
    elif args.between_condition_wilcoxon == "true":
        print(f"\n{'='*60}\nANALYSIS A2: BETWEEN-CONDITION WILCOXON  "
              f"[SKIPPED — fewer than 2 replicates per (condition,timepoint) cell]\n{'='*60}")
    else:
        print(f"\n{'='*60}\nANALYSIS A2: BETWEEN-CONDITION WILCOXON  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # A5: Between-timepoint Wilcoxon (pooled across conditions)
    # --------------------------------------------------------------
    # One CSV per later-vs-earlier day-pair generated from --timepoints (for
    # day_0,day_5,day_10 that is d10_vs_d0, d10_vs_d5, d5_vs_d0). Each pair pools
    # replicates ACROSS CONDITIONS within each timepoint (ignoring the BWM/control
    # split). Requires --timepoints with >=2 timepoints.
    a5_requested = args.between_timepoint_wilcoxon == "true"
    a5_has_timepoints = timepoint_order is not None and len(timepoint_order) >= 2
    if a5_requested and wilcoxon_feasible and a5_has_timepoints:
        print(f"\n{'='*60}\nANALYSIS A5: BETWEEN-TIMEPOINT WILCOXON (POOLED ACROSS CONDITIONS)\n{'='*60}")
        # All later-vs-earlier day-pairs generated from --timepoints.
        timepoint_pairs = build_timepoint_pairs(timepoint_order)
        for time_a, time_b, tag in timepoint_pairs:
            print(f"\n--- {time_a} vs {time_b} ---")
            print("  Wilcoxon (pooled across conditions)")
            df_w_tp = between_timepoint_wilcoxon(
                replicate_counts, rep_to_timepoint,
                feature_col=feature_col, time_a=time_a, time_b=time_b,
            )
            n_sig = (df_w_tp["p_adj"] < 0.05).sum() if not df_w_tp.empty else 0
            print(f"    Tests: {len(df_w_tp)}  |  Significant (p_adj<0.05): {n_sig}")
            w_path = out_dir / f"between_timepoint_wilcoxon_{tag}_{suffix}.csv"
            df_w_tp.to_csv(w_path, index=False)
            saved_paths.append(w_path)
    elif a5_requested and not a5_has_timepoints:
        print(f"\n{'='*60}\nANALYSIS A5: BETWEEN-TIMEPOINT WILCOXON (POOLED ACROSS CONDITIONS)  "
              f"[SKIPPED — needs --timepoints with >=2 timepoints]\n{'='*60}")
    elif a5_requested:
        print(f"\n{'='*60}\nANALYSIS A5: BETWEEN-TIMEPOINT WILCOXON (POOLED ACROSS CONDITIONS)  "
              f"[SKIPPED — fewer than 2 replicates per (condition,timepoint) cell]\n{'='*60}")
    else:
        print(f"\n{'='*60}\nANALYSIS A5: BETWEEN-TIMEPOINT WILCOXON (POOLED ACROSS CONDITIONS)  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # Write summary
    # --------------------------------------------------------------
    if saved_paths:
        print(f"\nSaved:")
        for p in saved_paths:
            print(f"  {p}")
    else:
        print("\nNo analyses selected — nothing written.")
    logging.info(f"All selected {level}-level enrichment results saved to {out_dir}")


if __name__ == "__main__":
    main()
