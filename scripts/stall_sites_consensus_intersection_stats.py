#!/usr/bin/env python3
"""
stall_sites_consensus_intersection_stats.py

Statistical half of the *consensus INTERSECTION* stall-site pipeline. Consumes
ONE of the two CSVs emitted by ``stall_sites_consensus_intersection.py``:

  * ``stall_sites_codon.csv`` → codon-level enrichment (alphabet = 61 sense codons)
  * ``stall_sites_aa.csv``    → amino-acid-level enrichment (alphabet = AA_ORDER)

Level is auto-detected from the input columns.

In the INTERSECTION variant every group is restricted to the transcripts that
pass filtering in ALL groups, so all conditions are called on one shared
transcript universe. Two consequences drive the test choice:

  * Raw stall-site shares are apples-to-apples across conditions, so FISHER's
    exact is a fair between-group comparison here.
  * The per-group backgrounds are computed from the same transcript set and are
    therefore IDENTICAL across groups, which makes the background-aware diff
    degenerate (the shared background cancels in delta_log2_enrichment, leaving
    the raw-share comparison Fisher already performs). The background-aware tests
    therefore live in the sibling ``stall_sites_consensus_union_stats.py`` (where
    the per-group transcript universes — and backgrounds — differ).

The consensus design has exactly one stall set per (condition, timepoint) cell
(consensus collapses replicates), so the per-replicate Wilcoxons (A2, A5) are
N/A and live in the non-consensus stats script. This script runs the
count-collapsing Fisher tests, which at n=1 per cell pool a single set — a
no-op, not a pseudoreplicate:

  * A1 — within-condition binomial enrichment (vs each group's background)
         [ribostall.enrichment.within_condition_enrichment]
  * A3 — between-condition Fisher's exact (sliced per-timepoint with --timepoints)
         [ribostall.enrichment.between_condition_fisher / per_timepoint_fisher]
  * A6 — between-timepoint Fisher within each condition (--timepoints only)
         [ribostall.enrichment.between_timepoint_fisher_within_condition]

A1/A3/A6 are this pipeline's analysis IDs. They are NOT the "Analysis 1/2/.../7"
section numbers in ``ribostall/enrichment.py`` — that module numbers its functions
on its own scheme and is shared across the union, intersection, and non-consensus
stats scripts (each selects a different subset of A-IDs). Cross-reference the two by
function name (in brackets above), not by number.

``--timepoints`` is OPTIONAL: the intersection consensus is not flat-only.
  * Without it (flat control vs treatment): A3 emits one between-condition Fisher
    (``between_condition_fisher``); A6 is skipped (no timepoint axis).
  * With it (groups carry timepoints, e.g. ``treatment_day_0``): A3 slices into a
    per-timepoint Fisher, and A6 runs across every later-vs-earlier day-pair.

Like ``stall_sites_non_consensus_stats.py`` this is intentionally ribopy-free.
The Fisher tests need no background; the within-condition binomial (A1) reads the
per-group background frequencies from the ``per_group_background_{codon,aa}.csv``
CSVs emitted by ``stall_sites_consensus_intersection.py`` (identical across
groups by construction), so the stats run on a machine without ribopy / the
source ``.ribo`` file.

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
    within_condition_enrichment,
    between_condition_fisher,
    per_timepoint_fisher,
    between_timepoint_fisher_within_condition,
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
        description="Run the Fisher count-collapsing enrichment tests (A1 within-condition binomial, "
                    "A3 between-condition Fisher, A6 between-timepoint Fisher) on a consensus "
                    "INTERSECTION stall-site CSV produced by stall_sites_consensus_intersection.py."
    )
    parser.add_argument("--stall-sites", required=True,
                        help="Path to stall_sites_codon.csv or stall_sites_aa.csv")
    parser.add_argument("--groups", required=True,
                        help="Experimental groups, e.g. 'control:control;treatment:treatment' "
                             "(consensus sets replicate == group name); for a timepoint design make "
                             "the group names carry the timepoint, e.g. "
                             "'treatment_day_0:treatment_day_0;control_day_0:control_day_0;...'")
    parser.add_argument("--background", default=None,
                        help="Path to per_group_background_{level}.csv written by the consensus call "
                             "script. Used ONLY by the within-condition binomial (A1); the Fisher "
                             "tests (A3/A6) compare raw shares and need no background. OPTIONAL — "
                             "required only when A1 runs (--within-condition true, the default); a "
                             "Fisher-only run may omit it. Identical across groups in the intersection "
                             "design.")
    parser.add_argument("--out-dir", default="results/stall_sites_consensus_intersection/analysis",
                        help="Output directory for enrichment CSVs")
    parser.add_argument("--headline-condition", default=None,
                        help="Condition treated as the headline (numerator) in the between-condition "
                             "Fisher test (A3): a positive log2(odds ratio) means enriched in this "
                             "condition. Must match one of the two condition labels (e.g. "
                             "'treatment'). Default: alphabetical (first condition is headline). The "
                             "between-timepoint Fisher (A6) is fixed to later-vs-earlier and ignores "
                             "this flag.")
    parser.add_argument("--timepoints", default=None,
                        help="OPTIONAL. Comma-separated timepoint labels in chronological order "
                             "(earliest first), e.g. 'day_0,day_5,day_10'. Declare only when the "
                             "--groups carry timepoints (e.g. treatment_day_0). With it (>=2 "
                             "timepoints), A3 slices per-timepoint and A6 runs across day-pairs; "
                             "without it, A3 emits a single between-condition Fisher and A6 "
                             "is skipped. Timepoints are NOT sorted automatically — a string sort "
                             "would place 'day_10' before 'day_5'.")

    # Per-analysis toggles. Each takes the literal value true or false and
    # defaults to true (the analysis runs); pass e.g.
    # --between-condition-fisher false to skip it. A skipped analysis is
    # announced and writes no output CSV.
    parser.add_argument("--within-condition", choices=["true", "false"], default="true",
                        help="A1: within-condition binomial enrichment. "
                             "Default: true (set false to skip).")
    parser.add_argument("--between-condition-fisher", choices=["true", "false"], default="true",
                        help="A3: between-condition Fisher's exact (sliced per-timepoint when "
                             "--timepoints is given). Default: true (set false to skip).")
    parser.add_argument("--between-timepoint-fisher", choices=["true", "false"], default="true",
                        help="A6: between-timepoint Fisher within each condition. Requires "
                             "--timepoints with >=2 timepoints. Default: true (set false to skip).")
    return parser.parse_args()


def main():
    args = parse_args()

    # The within-condition binomial (A1) is the only analysis that reads the
    # per-group background; the Fisher tests (A3/A6) compare raw shares. Require
    # --background only when A1 actually runs, so a Fisher-only invocation
    # (--within-condition false) does not need the file.
    if args.within_condition == "true" and args.background is None:
        sys.exit(
            "--background is required when the within-condition binomial (A1) runs "
            "(--within-condition is true, the default). Provide "
            "per_group_background_{codon,aa}.csv, or pass --within-condition false to skip A1."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Load stall sites and detect granularity
    # --------------------------------------------------------------
    stall_path = Path(args.stall_sites)
    logging.info(f"Loading consensus stall sites from {stall_path} ...")
    # Reads the input csv
    df = pd.read_csv(stall_path)
    # level = "codon" or "aa"; site_cols = (E_col, P_col, A_col); alphabet = list of codons or AAs; feature_col = "codon" or "amino_acid"
    level, site_cols, alphabet, feature_col = detect_level(df)
    suffix = level  # "codon" or "aa"
    logging.info(f"Detected level: {level} ({len(alphabet)} categories; feature column '{feature_col}')")

    # --------------------------------------------------------------
    # Groups, condition, and (optional) timepoint mappings. Consensus collapses
    # replicates into one set per group and writes replicate == group, so each
    # "replicate" is a group and the condition is the part of the group name
    # before the first underscore (treatment_day_0 → treatment; a flat group
    # name like treatment is its own condition).
    # --------------------------------------------------------------
    # Ex flat: 'control:control;treatment:treatment' → {"control": ["control"], "treatment": ["treatment"]}
    groups = parse_groups(args.groups)
    # Would look like {"control": "control", "treatment": "treatment"}
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    rep_to_condition = {rep: grp.split("_", 1)[0] for rep, grp in rep_to_group.items()}

    # Validate --headline-condition up front: it must name one of the two
    # conditions (the part of each group name before the first "_"). Failing fast
    # here with a clear message mirrors the --timepoints check below; otherwise a
    # typo surfaces only as an uncaught ValueError from deep inside the Fisher
    # test (A3). A6 (between-timepoint) ignores the headline.
    if args.headline_condition is not None:
        conditions = sorted(set(rep_to_condition.values()))
        if args.headline_condition not in conditions:
            sys.exit(
                f"--headline-condition {args.headline_condition!r} is not one of the conditions "
                f"{conditions} derived from --groups (condition = group name before the first '_'). "
                f"Pass one of {conditions}."
            )

    # --timepoints is OPTIONAL: build + validate the timepoint mapping only when
    # given. Declared chronological order (no automatic sorting — a string sort
    # would place "day_10" before "day_5") drives the per-tp slicing of A3 and
    # the A6 day-pair generation.
    timepoint_order = None
    rep_to_timepoint = None

    # Parse through the timepoints if declared
    if args.timepoints:
        # Gets the order
        timepoint_order = parse_timepoints(args.timepoints)
        # Builds the mapping from replicates to timepoints
        rep_to_timepoint = build_rep_to_timepoint(rep_to_group)
        try:
            undeclared = validate_timepoints(timepoint_order, rep_to_timepoint)
        except ValueError as e:
            # If declared replicate is not in the timepoint mapping
            sys.exit(str(e))
        if undeclared:
            # If replicate is not declared but exists in the timepoint mapping
            logging.warning(f"Timepoints {sorted(undeclared)} are present in --groups but not in "
                            f"--timepoints; they are excluded from the timepoint analyses.")
    has_timepoints = timepoint_order is not None and len(timepoint_order) >= 2

    # --------------------------------------------------------------
    # Per-replicate counts (one "replicate" per group after consensus)
    # --------------------------------------------------------------
    # Builds a replicate map to their counts by site and per unit
    replicate_counts = build_replicate_counts(df, site_cols, alphabet)

    # Print total counts per site across replicates (for sanity check)
    print(f"\n{'='*60}\nTOTAL CONSENSUS STALL SITE COUNTS PER SITE\n{'='*60}")
    for rep, site_counts in replicate_counts.items():
        totals = {s: int(site_counts[s].sum()) for s in ("E", "P", "A")}
        print(f"  [{rep}] counts per site: {totals}")

    # --------------------------------------------------------------
    # Load per-group background frequencies (written by the call script). Only
    # the within-condition binomial (A1) uses them; the Fisher tests compare raw
    # stall-site shares and need no background, so this load is skipped entirely
    # when A1 is off (--within-condition false). In the intersection design these
    # per-group backgrounds are identical across groups by construction.
    # --------------------------------------------------------------
    bg_freq_per_group = {}
    if args.within_condition == "true":
        bg_path = Path(args.background)
        logging.info(f"Loading per-group {level} backgrounds from {bg_path} ...")
        bg_df = pd.read_csv(bg_path)

        print(f"\n{'='*60}\nBACKGROUND {level.upper()} FREQUENCIES (per group)\n{'='*60}")

        # Stores the background frequencies per group and their total counts in a dictionary
        for grp, sub in bg_df.groupby("group"):
            freq = sub.set_index(feature_col)["bg_freq"].reindex(alphabet).astype(float)
            bg_freq_per_group[grp] = freq
            total = int(sub.set_index(feature_col)["bg_count"].reindex(alphabet).fillna(0).astype(int).sum())
            print(f"  [{grp}] {total} total {level}s")
        print(f"{'='*60}\n")

    # Each analysis below runs only when its toggle is on (the default). A
    # skipped analysis is announced and writes no CSV. saved_paths collects the
    # outputs actually written, in run order, for the closing summary.
    saved_paths = []

    # --------------------------------------------------------------
    # A1: Within-condition enrichment (binomial)
    # --------------------------------------------------------------
    if args.within_condition == "true":
        print(f"\n{'='*60}\nANALYSIS A1: WITHIN-CONDITION ENRICHMENT (Binomial)\n{'='*60}")
        df_within = within_condition_enrichment(
            replicate_counts, bg_freq_per_group, rep_to_condition, rep_to_group,
            feature_col=feature_col,
        )
        n_sig = (df_within["p_adj"] < 0.05).sum() if not df_within.empty else 0
        print(f"  Tests: {len(df_within)}  |  Significant (p_adj<0.05): {n_sig}")
        within_path = out_dir / f"within_condition_binomial_{suffix}.csv"
        df_within.to_csv(within_path, index=False)
        saved_paths.append(within_path)
    else:
        print(f"\n{'='*60}\nANALYSIS A1: WITHIN-CONDITION ENRICHMENT (Binomial)  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # A3: Between-condition Fisher's exact (per-timepoint when --timepoints given)
    # --------------------------------------------------------------
    if args.between_condition_fisher == "true":
        if has_timepoints:
            print(f"\n{'='*60}\nANALYSIS A3: PER-TIMEPOINT FISHER'S EXACT\n"
                  f"  NOTE: consensus pools a single set per (condition,timepoint) cell — a no-op\n{'='*60}")
            if args.headline_condition is not None:
                print(f"  Headline condition: {args.headline_condition} "
                      f"(positive log2 odds ratio = enriched in {args.headline_condition})")
            else:
                print("  Headline condition: alphabetical default "
                      "(positive log2 odds ratio = enriched in the first condition)")
            df_fisher = per_timepoint_fisher(
                replicate_counts, rep_to_condition, rep_to_timepoint, feature_col=feature_col,
                headline_condition=args.headline_condition, timepoints=timepoint_order,
            )
            for tp in (timepoint_order if not df_fisher.empty else []):
                tp_df = df_fisher[df_fisher["timepoint"] == tp]
                if tp_df.empty:
                    continue
                n_sig_tp = (tp_df["p_adj"] < 0.05).sum()
                print(f"  [{tp}] {len(tp_df)} tests, {n_sig_tp} significant")
            fisher_path = out_dir / f"per_timepoint_fisher_{suffix}.csv"
        else:
            print(f"\n{'='*60}\nANALYSIS A3: BETWEEN-CONDITION FISHER'S EXACT\n"
                  f"  NOTE: consensus pools sites per group — interpret cautiously\n{'='*60}")
            if args.headline_condition is not None:
                print(f"  Headline condition: {args.headline_condition} "
                      f"(positive log2 odds ratio = enriched in {args.headline_condition})")
            else:
                print("  Headline condition: alphabetical default "
                      "(positive log2 odds ratio = enriched in the first condition)")
            df_fisher = between_condition_fisher(
                replicate_counts, rep_to_condition, feature_col=feature_col,
                headline_condition=args.headline_condition,
            )
            for site in sorted(df_fisher["site"].unique()) if not df_fisher.empty else []:
                site_df = df_fisher[df_fisher["site"] == site]
                n_sig_s = (site_df["p_adj"] < 0.05).sum()
                print(f"  [{site}] {len(site_df)} tests, {n_sig_s} significant")
            fisher_path = out_dir / f"between_condition_fisher_{suffix}.csv"
        df_fisher.to_csv(fisher_path, index=False)
        saved_paths.append(fisher_path)
    else:
        print(f"\n{'='*60}\nANALYSIS A3: BETWEEN-CONDITION FISHER'S EXACT  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # A6: Between-timepoint Fisher's (within each condition) — --timepoints only
    # --------------------------------------------------------------
    if args.between_timepoint_fisher == "true":
        if has_timepoints:
            print(f"\n{'='*60}\nANALYSIS A6: BETWEEN-TIMEPOINT FISHER'S EXACT (WITHIN EACH CONDITION)\n"
                  f"  NOTE: consensus pools a single set per (condition,timepoint) cell — a no-op\n{'='*60}")
            timepoint_pairs = build_timepoint_pairs(timepoint_order)
            for time_a, time_b, tag in timepoint_pairs:
                print(f"\n--- {time_a} vs {time_b} ---")
                df_f_tp = between_timepoint_fisher_within_condition(
                    replicate_counts, rep_to_condition, rep_to_timepoint,
                    feature_col=feature_col, time_a=time_a, time_b=time_b,
                )
                for cond in sorted(df_f_tp["condition"].unique()) if not df_f_tp.empty else []:
                    cond_df = df_f_tp[df_f_tp["condition"] == cond]
                    n_sig_c = (cond_df["p_adj"] < 0.05).sum()
                    print(f"    [{cond}] {len(cond_df)} tests, {n_sig_c} significant")
                f_path = out_dir / f"timepoint_fisher_within_condition_{tag}_{suffix}.csv"
                df_f_tp.to_csv(f_path, index=False)
                saved_paths.append(f_path)
        else:
            print(f"\n{'='*60}\nANALYSIS A6: BETWEEN-TIMEPOINT FISHER'S EXACT (WITHIN EACH CONDITION)  "
                  f"[SKIPPED — needs --timepoints with >=2 timepoints]\n{'='*60}")
    else:
        print(f"\n{'='*60}\nANALYSIS A6: BETWEEN-TIMEPOINT FISHER'S EXACT (WITHIN EACH CONDITION)  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # Write summary
    # --------------------------------------------------------------
    if saved_paths:
        print(f"\nSaved:")
        for p in saved_paths:
            print(f"  {p}")
    else:
        print("\nNo analyses selected — nothing written.")
    logging.info(f"All selected {level}-level consensus intersection enrichment results saved to {out_dir}")


if __name__ == "__main__":
    main()
