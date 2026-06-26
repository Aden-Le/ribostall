#!/usr/bin/env python3
"""
stall_sites_consensus_union_stats.py

Statistical half of the *consensus UNION* stall-site pipeline. Consumes ONE of
the two CSVs emitted by ``stall_sites_consensus_union.py``:

  * ``stall_sites_codon.csv`` → codon-level enrichment (alphabet = 61 sense codons)
  * ``stall_sites_aa.csv``    → amino-acid-level enrichment (alphabet = AA_ORDER)

Level is auto-detected from the input columns.

In the UNION variant each group keeps its OWN filtered transcript set, so the
per-group sequence-composition backgrounds differ between conditions. The valid
between-group comparison here is therefore the BACKGROUND-AWARE diff, which
normalizes each condition/timepoint to its own background before comparing — a
shift in the expressed transcriptome cannot masquerade as differential stalling.
A raw Fisher test across differing transcript universes would be confounded, so
the Fisher tests live in the sibling
``stall_sites_consensus_intersection_stats.py`` (where every group shares one
intersected transcript set and the backgrounds are identical).

The consensus design has exactly one stall set per (condition, timepoint) cell
(consensus collapses replicates), so the per-replicate Wilcoxons (A2, A5) are
N/A and live in the non-consensus stats script. This script runs the
count-collapsing background-aware tests, which at n=1 per cell pool a single
set — a no-op, not a pseudoreplicate:

  * A1 — within-condition binomial enrichment (vs each group's background)
  * A4 — between-condition background-aware diff (per-timepoint with --timepoints)
  * A7 — between-timepoint background-aware diff, pooled across conditions (--timepoints only)

``--timepoints`` is OPTIONAL: the union consensus is not flat-only.
  * Without it (flat control vs treatment): A4 emits one between-condition
    comparison (``between_condition_background_diff``); A7 is skipped (no
    timepoint axis).
  * With it (groups carry timepoints, e.g. ``treatment_day_0``): A4 slices into a
    per-timepoint background-aware diff, and A7 runs across every later-vs-earlier
    day-pair.

Like ``stall_sites_non_consensus_stats.py`` this is intentionally ribopy-free:
per-group background frequencies are read from the
``per_group_background_{codon,aa}.csv`` CSVs emitted by
``stall_sites_consensus_union.py``, so the stats run on a machine without ribopy
/ the source ``.ribo`` file.

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
    between_condition_background_diff,
    between_timepoint_background_diff,
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
        description="Run the background-aware count-collapsing enrichment tests (A1 within-condition "
                    "binomial, A4 between-condition background-diff, A7 between-timepoint "
                    "background-diff) on a consensus UNION stall-site CSV produced by "
                    "stall_sites_consensus_union.py."
    )
    parser.add_argument("--stall-sites", required=True,
                        help="Path to stall_sites_codon.csv or stall_sites_aa.csv")
    parser.add_argument("--groups", required=True,
                        help="Experimental groups, e.g. 'control:control;treatment:treatment' "
                             "(consensus sets replicate == group name); for a timepoint design make "
                             "the group names carry the timepoint, e.g. "
                             "'treatment_day_0:treatment_day_0;control_day_0:control_day_0;...'")
    parser.add_argument("--background", required=True,
                        help="Path to per_group_background_{level}.csv written by the consensus call script.")
    parser.add_argument("--out-dir", default="results/stall_sites_consensus_union/analysis",
                        help="Output directory for enrichment CSVs")
    parser.add_argument("--headline-condition", default=None,
                        help="Condition treated as the headline (numerator) in the between-condition "
                             "background-aware diff (A4): a positive delta_log2_enrichment means more "
                             "enriched vs background in this condition. Must match one of the two "
                             "condition labels (e.g. 'treatment'). Default: alphabetical (first "
                             "condition is headline). The between-timepoint diff (A7) is fixed to "
                             "later-vs-earlier and ignores this flag.")
    parser.add_argument("--timepoints", default=None,
                        help="OPTIONAL. Comma-separated timepoint labels in chronological order "
                             "(earliest first), e.g. 'day_0,day_5,day_10'. Declare only when the "
                             "--groups carry timepoints (e.g. treatment_day_0). With it (>=2 "
                             "timepoints), A4 slices per-timepoint and A7 runs across day-pairs; "
                             "without it, A4 emits a single between-condition comparison and A7 "
                             "is skipped. Timepoints are NOT sorted automatically — a string sort "
                             "would place 'day_10' before 'day_5'.")

    # Per-analysis toggles. Each takes the literal value true or false and
    # defaults to true (the analysis runs); pass e.g.
    # --between-condition-background-diff false to skip it. A skipped analysis is
    # announced and writes no output CSV.
    parser.add_argument("--within-condition", choices=["true", "false"], default="true",
                        help="A1: within-condition binomial enrichment. "
                             "Default: true (set false to skip).")
    parser.add_argument("--between-condition-background-diff", choices=["true", "false"], default="true",
                        help="A4: between-condition background-aware diff (per-timepoint when "
                             "--timepoints is given). Default: true (set false to skip).")
    parser.add_argument("--between-timepoint-background-diff", choices=["true", "false"], default="true",
                        help="A7: between-timepoint background-aware diff (pooled across conditions). "
                             "Requires --timepoints with >=2 timepoints. Default: true (set false to skip).")
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Load stall sites and detect granularity
    # --------------------------------------------------------------
    stall_path = Path(args.stall_sites)
    logging.info(f"Loading consensus stall sites from {stall_path} ...")
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

    # --timepoints is OPTIONAL: build + validate the timepoint mapping only when
    # given. Declared chronological order (no automatic sorting — a string sort
    # would place "day_10" before "day_5") drives the per-tp slicing of A4 and
    # the A7 day-pair generation.
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
    has_timepoints = timepoint_order is not None and len(timepoint_order) >= 2

    # --------------------------------------------------------------
    # Per-replicate counts (one "replicate" per group after consensus)
    # --------------------------------------------------------------
    replicate_counts = build_replicate_counts(df, site_cols, alphabet)

    # Print total counts per site across replicates (for sanity check)
    print(f"\n{'='*60}\nTOTAL CONSENSUS STALL SITE COUNTS PER SITE\n{'='*60}")
    for rep, site_counts in replicate_counts.items():
        totals = {s: int(site_counts[s].sum()) for s in ("E", "P", "A")}
        print(f"  [{rep}] counts per site: {totals}")

    # --------------------------------------------------------------
    # Load per-group background frequencies (written by the call script)
    # --------------------------------------------------------------
    bg_path = Path(args.background)
    bg_freq_per_group = {}
    bg_counts_per_group = {}

    logging.info(f"Loading per-group {level} backgrounds from {bg_path} ...")
    bg_df = pd.read_csv(bg_path)

    print(f"\n{'='*60}\nBACKGROUND {level.upper()} FREQUENCIES (per group)\n{'='*60}")

    for grp, sub in bg_df.groupby("group"):
        freq = sub.set_index(feature_col)["bg_freq"].reindex(alphabet).astype(float)
        counts = sub.set_index(feature_col)["bg_count"].reindex(alphabet).fillna(0).astype(int)
        bg_freq_per_group[grp] = freq
        bg_counts_per_group[grp] = counts
        print(f"  [{grp}] {int(counts.sum())} total {level}s")
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
    # A4: Between-condition background-aware diff (per-timepoint when --timepoints given)
    # --------------------------------------------------------------
    # Each condition is normalized to its OWN background before comparison.
    # Positive delta_log2_enrichment = more enriched vs background in the
    # headline condition. This is the union pipeline's between-group comparison:
    # the per-group transcript universes (and hence backgrounds) differ, so
    # normalizing to each condition's own background is what makes the contrast
    # valid (a raw Fisher would be confounded by the differing transcript sets).
    if args.between_condition_background_diff == "true":
        if has_timepoints:
            # Per-timepoint variant. Each condition spans several timepoint groups,
            # so slice to one timepoint at a time and feed that timepoint's own
            # per-group background (e.g. BWM_day_0 vs control_day_0), keeping the
            # background matched within each comparison. FDR is applied per
            # (timepoint, site).
            print(f"\n{'='*60}\nANALYSIS A4: PER-TIMEPOINT BACKGROUND-AWARE DIFF\n"
                  f"  NOTE: enrichment-over-background ratio; consensus pools a single set "
                  f"per cell — a no-op\n{'='*60}")
            if args.headline_condition is not None:
                print(f"  Headline condition: {args.headline_condition} "
                      f"(positive delta_log2_enrichment = more enriched vs background in {args.headline_condition})")
            else:
                print("  Headline condition: alphabetical default "
                      "(positive delta_log2_enrichment = more enriched vs background in the first condition)")

            conditions = sorted(set(rep_to_condition.values()))

            # Map (condition, timepoint) -> group so each per-timepoint comparison
            # pulls that timepoint's own per-group background frequencies, e.g.
            #   {("BWM", "day_0"): "BWM_day_0", ("control", "day_0"): "control_day_0", ...}
            cond_tp_to_group = {
                (rep_to_condition[rep], rep_to_timepoint[rep]): grp
                for rep, grp in rep_to_group.items()
            }

            bgdiff_frames = []
            for tp in timepoint_order:
                # Replicates (consensus sets) at this timepoint
                reps_at_tp = {
                    rep: counts for rep, counts in replicate_counts.items()
                    if rep_to_timepoint.get(rep) == tp
                }
                # That timepoint's per-condition background frequencies
                bg_for_tp = {
                    cond: bg_freq_per_group[cond_tp_to_group[(cond, tp)]]
                    for cond in conditions
                }
                df_bgdiff_tp = between_condition_background_diff(
                    reps_at_tp, rep_to_condition, bg_for_tp,
                    feature_col=feature_col, headline_condition=args.headline_condition,
                )
                if not df_bgdiff_tp.empty:
                    # Tag the timepoint as the second column (after "site").
                    df_bgdiff_tp.insert(1, "timepoint", tp)
                bgdiff_frames.append(df_bgdiff_tp)

                n_sig_tp = (df_bgdiff_tp["p_adj"] < 0.05).sum() if not df_bgdiff_tp.empty else 0
                print(f"  [{tp}] {len(df_bgdiff_tp)} tests, {n_sig_tp} significant")

            df_bgdiff = pd.concat(bgdiff_frames, ignore_index=True) if bgdiff_frames else pd.DataFrame()
            bgdiff_path = out_dir / f"per_timepoint_background_diff_{suffix}.csv"
        else:
            # Flat variant. group == condition in this design, so bg_freq_per_group
            # keys directly into the test.
            print(f"\n{'='*60}\nANALYSIS A4: BETWEEN-CONDITION BACKGROUND-AWARE DIFF\n"
                  f"  NOTE: enrichment-over-background ratio; consensus pools sites — interpret cautiously\n{'='*60}")
            if args.headline_condition is not None:
                print(f"  Headline condition: {args.headline_condition} "
                      f"(positive delta_log2_enrichment = more enriched vs background in {args.headline_condition})")
            else:
                print("  Headline condition: alphabetical default "
                      "(positive delta_log2_enrichment = more enriched vs background in the first condition)")
            df_bgdiff = between_condition_background_diff(
                replicate_counts, rep_to_condition, bg_freq_per_group,
                feature_col=feature_col, headline_condition=args.headline_condition,
            )
            for site in sorted(df_bgdiff["site"].unique()) if not df_bgdiff.empty else []:
                site_df = df_bgdiff[df_bgdiff["site"] == site]
                n_sig_s = (site_df["p_adj"] < 0.05).sum()
                print(f"  [{site}] {len(site_df)} tests, {n_sig_s} significant")
            bgdiff_path = out_dir / f"between_condition_background_diff_{suffix}.csv"
        df_bgdiff.to_csv(bgdiff_path, index=False)
        saved_paths.append(bgdiff_path)
    else:
        print(f"\n{'='*60}\nANALYSIS A4: BETWEEN-CONDITION BACKGROUND-AWARE DIFF  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # A7: Between-timepoint background-aware diff (pooled across conditions) — --timepoints only
    # --------------------------------------------------------------
    # Background-aware counterpart that POOLS replicates ACROSS CONDITIONS within
    # each timepoint (ignoring the condition split) and normalizes each timepoint
    # to its OWN count-weighted pooled background. Direction is later-vs-earlier
    # (time_a vs time_b), independent of --headline-condition. One combined CSV
    # across the day-pairs, tagged by a `comparison` column.
    if args.between_timepoint_background_diff == "true":
        if has_timepoints:
            print(f"\n{'='*60}\nANALYSIS A7: BETWEEN-TIMEPOINT BACKGROUND-AWARE DIFF (POOLED ACROSS CONDITIONS)\n"
                  f"  NOTE: enrichment-over-background ratio; consensus pools a single set "
                  f"per cell — a no-op\n{'='*60}")

            # Count-weighted pooled background per timepoint: sum bg_count across
            # the groups (conditions) at the timepoint, then renormalize to
            # frequencies, so the larger-library condition contributes
            # proportionally more — matching how the foreground stall counts are
            # pooled. group "control_day_0" -> timepoint "day_0".
            group_to_timepoint = {grp: grp.split("_", 1)[1] for grp in bg_counts_per_group}
            bg_freq_per_timepoint = {}
            bg_total_per_timepoint = {}
            for tp in timepoint_order:
                pooled_counts = None
                for grp, grp_tp in group_to_timepoint.items():
                    if grp_tp != tp:
                        continue
                    counts = bg_counts_per_group[grp]
                    pooled_counts = counts.copy() if pooled_counts is None else pooled_counts.add(counts, fill_value=0)
                if pooled_counts is None:
                    continue
                total = int(pooled_counts.sum())
                bg_total_per_timepoint[tp] = total
                bg_freq_per_timepoint[tp] = pooled_counts / total if total > 0 else pooled_counts * 0.0

            print("  Pooled background totals per timepoint (count-weighted across conditions):")
            for tp in timepoint_order:
                if tp in bg_total_per_timepoint:
                    print(f"    [{tp}] {bg_total_per_timepoint[tp]} total {level}s")

            timepoint_pairs = build_timepoint_pairs(timepoint_order)
            bgtp_frames = []
            for time_a, time_b, tag in timepoint_pairs:
                print(f"\n--- {time_a} vs {time_b} ---")
                df_bgtp = between_timepoint_background_diff(
                    replicate_counts, rep_to_timepoint, bg_freq_per_timepoint,
                    feature_col=feature_col, time_a=time_a, time_b=time_b,
                )
                if not df_bgtp.empty:
                    # Tag the comparison as the second column (after "site") so the
                    # day-pairs stack into one CSV.
                    df_bgtp.insert(1, "comparison", tag)
                bgtp_frames.append(df_bgtp)

                n_sig = (df_bgtp["p_adj"] < 0.05).sum() if not df_bgtp.empty else 0
                print(f"  [{tag}] {len(df_bgtp)} tests, {n_sig} significant")

            df_bgtp_all = pd.concat(bgtp_frames, ignore_index=True) if bgtp_frames else pd.DataFrame()
            bgtp_path = out_dir / f"between_timepoint_background_diff_{suffix}.csv"
            df_bgtp_all.to_csv(bgtp_path, index=False)
            saved_paths.append(bgtp_path)
        else:
            print(f"\n{'='*60}\nANALYSIS A7: BETWEEN-TIMEPOINT BACKGROUND-AWARE DIFF (POOLED ACROSS CONDITIONS)  "
                  f"[SKIPPED — needs --timepoints with >=2 timepoints]\n{'='*60}")
    else:
        print(f"\n{'='*60}\nANALYSIS A7: BETWEEN-TIMEPOINT BACKGROUND-AWARE DIFF (POOLED ACROSS CONDITIONS)  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # Write summary
    # --------------------------------------------------------------
    if saved_paths:
        print(f"\nSaved:")
        for p in saved_paths:
            print(f"  {p}")
    else:
        print("\nNo analyses selected — nothing written.")
    logging.info(f"All selected {level}-level consensus union enrichment results saved to {out_dir}")


if __name__ == "__main__":
    main()
