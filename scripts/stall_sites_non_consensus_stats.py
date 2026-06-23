#!/usr/bin/env python3
"""
stall_sites_non_consensus_stats.py

Statistical half of the non-consensus stall-site pipeline. Consumes ONE of the
two CSVs emitted by ``stall_sites_non_consensus_call.py``:

  * ``stall_sites_codon.csv`` → codon-level enrichment (alphabet = 61 sense codons)
  * ``stall_sites_aa.csv``    → amino-acid-level enrichment (alphabet = AA_ORDER)

Level is auto-detected from the input columns. The enrichment tests
(within-condition binomial, between-condition Wilcoxon, between-timepoint
Wilcoxon + Fisher, per-timepoint Fisher, the per-timepoint background-aware
diff, and the between-timepoint background-aware diff) are run identically in
both modes — the only things that differ are the feature alphabet and the
background-frequency helper.

The per-timepoint background-aware diff (Analysis 5) is the background-aware
counterpart of the per-timepoint Fisher (Analysis 4): instead of comparing raw
stall-site shares between conditions, it compares each condition's enrichment
over its OWN background, so a shift in the expressed transcriptome between
conditions cannot masquerade as differential stalling. It is run per timepoint
(using each timepoint's own per-group background, e.g. BWM_day_0 vs
control_day_0) because — unlike the flat consensus design where group ==
condition — each condition here spans several timepoint groups. It complements
rather than replaces Fisher; the two agree when the per-(condition,timepoint)
backgrounds match and diverge only when they differ.

The between-timepoint background-aware diff (Analysis 6) is the background-aware
counterpart of the between-timepoint Wilcoxon (Analysis 3a): both POOL replicates
ACROSS CONDITIONS within each timepoint (ignoring the BWM/control split), but
Analysis 6 normalizes each timepoint to its OWN count-weighted pooled background,
so a shift in the expressed transcriptome across the time course cannot
masquerade as differential stalling. It compares every later-vs-earlier day-pair
generated from ``--timepoints`` (for day_0,day_5,day_10 that is d10_vs_d0,
d10_vs_d5, d5_vs_d0) and writes one combined CSV tagged by a ``comparison``
column. Direction is later-vs-earlier (time_a vs time_b), independent of
--headline-condition (the comparison is between timepoints, not conditions).

This script is intentionally ribopy-free: per-group background frequencies
are read from the ``per_group_background_{codon,aa}.csv`` CSVs emitted by
``stall_sites_non_consensus_call.py``. That lets the stats run on a machine
without ribopy / the source ``.ribo`` file.

Run it twice, once per CSV, to get codon-level and AA-level outputs side by
side. Output filenames are suffixed with ``_codon`` or ``_aa`` accordingly.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from ribostall.amino_acids import AA_ORDER, SENSE_CODONS
from ribostall.enrichment import (
    within_condition_enrichment,
    between_condition_wilcoxon,
    between_timepoint_wilcoxon,
    between_timepoint_fisher_within_condition,
    per_timepoint_fisher,
    between_condition_background_diff,
    between_timepoint_background_diff,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run enrichment tests on a per-stall CSV produced by stall_sites_non_consensus_call.py."
    )
    parser.add_argument("--stall-sites", required=True,
                        help="Path to stall_sites_codon.csv or stall_sites_aa.csv")
    parser.add_argument("--groups", required=True,
                        help="Experimental groups, e.g. 'groupA:rep1,rep2;groupB:rep3,rep4'")
    parser.add_argument("--background", required=True,
                        help="Path to per_group_background_{level}.csv written by the call script.")
    parser.add_argument("--out-dir", default="results/stall_sites_non_consensus/analysis",
                        help="Output directory for enrichment CSVs")
    parser.add_argument("--headline-condition", default=None,
                        help="Condition treated as the headline (numerator / direction reference) in "
                             "ALL between-condition tests: the between-condition Wilcoxon (Analysis 2; "
                             "positive log2_FC = higher per-replicate frequency here), the per-timepoint "
                             "Fisher (Analysis 4; positive log2 odds ratio = enriched here), and the "
                             "per-timepoint background-aware diff (Analysis 5; positive "
                             "delta_log2_enrichment = more enriched vs background here). Must match one "
                             "of the two condition labels (e.g. 'BWM'). Default: alphabetical (first "
                             "condition is headline).")
    parser.add_argument("--timepoints", required=True,
                        help="Comma-separated timepoint labels in chronological order (earliest "
                             "first), e.g. 'day_0,day_5,day_10'. Sets the order of the per-timepoint "
                             "analyses (4, 5) and generates the later-vs-earlier comparison pairs "
                             "(Analyses 3, 6). Timepoints are NOT sorted automatically — a string "
                             "sort would place 'day_10' before 'day_5'.")

    # Per-analysis toggles. Each takes the literal value true or false and
    # defaults to true (the analysis runs); pass e.g.
    # --per-timepoint-background-diff false to skip it. The between-timepoint
    # block (Analysis 3) is split into its two sub-tests so each can be skipped
    # independently. A skipped analysis writes no output CSV.
    parser.add_argument("--within-condition", choices=["true", "false"], default="true",
                        help="Analysis 1: within-condition binomial enrichment. "
                             "Default: true (set false to skip).")
    parser.add_argument("--between-condition-wilcoxon", choices=["true", "false"], default="true",
                        help="Analysis 2: between-condition Wilcoxon. "
                             "Default: true (set false to skip).")
    parser.add_argument("--between-timepoint-wilcoxon", choices=["true", "false"], default="true",
                        help="Analysis 3a: between-timepoint Wilcoxon (pooled across conditions). "
                             "Default: true (set false to skip).")
    parser.add_argument("--between-timepoint-fisher", choices=["true", "false"], default="true",
                        help="Analysis 3b: between-timepoint Fisher within each condition. "
                             "Default: true (set false to skip).")
    parser.add_argument("--per-timepoint-fisher", choices=["true", "false"], default="true",
                        help="Analysis 4: per-timepoint Fisher's exact. "
                             "Default: true (set false to skip).")
    parser.add_argument("--per-timepoint-background-diff", choices=["true", "false"], default="true",
                        help="Analysis 5: per-timepoint background-aware diff. "
                             "Default: true (set false to skip).")
    parser.add_argument("--between-timepoint-background-diff", choices=["true", "false"], default="true",
                        help="Analysis 6: between-timepoint background-aware diff (replicates pooled "
                             "across conditions per timepoint). Default: true (set false to skip).")
    return parser.parse_args()


def parse_groups(groups_arg):
    groups = {}
    for block in groups_arg.split(";"):
        name, reps = block.split(":")
        groups[name] = reps.split(",")
    return groups


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


def detect_level(df: pd.DataFrame) -> tuple[str, tuple[str, str, str], list, str]:
    """Return (level, (E_col, P_col, A_col), alphabet, feature_col_name)."""
    if {"E_codon", "P_codon", "A_codon"}.issubset(df.columns):
        return "codon", ("E_codon", "P_codon", "A_codon"), list(SENSE_CODONS), "codon"
    if {"E_aa", "P_aa", "A_aa"}.issubset(df.columns):
        return "aa", ("E_aa", "P_aa", "A_aa"), list(AA_ORDER), "amino_acid"
    raise ValueError(
        "Input CSV must contain either (E_codon,P_codon,A_codon) or (E_aa,P_aa,A_aa) columns."
    )


def build_replicate_counts(df: pd.DataFrame, site_cols, alphabet) -> dict:
    """{rep: {"E": Series, "P": Series, "A": Series}} indexed by ``alphabet``."""
    out = {}
    for rep, sub in df.groupby("replicate"):
        rep_map = {}
        for site_name, col in zip(("E", "P", "A"), site_cols):
            counts = sub[col].value_counts()
            rep_map[site_name] = counts.reindex(alphabet, fill_value=0).astype(int)
        out[rep] = rep_map
    return out


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
    # Groups, condition, timepoint mappings
    # --------------------------------------------------------------
    groups = parse_groups(args.groups)
    # Map reps to groups
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    rep_to_condition = {}
    rep_to_timepoint = {}
    for rep, grp in rep_to_group.items():
        # BWM_day_0 → condition = BWM, timepoint = day_0
        parts = grp.split("_", 1)
        rep_to_condition[rep] = parts[0]
        rep_to_timepoint[rep] = parts[1] if len(parts) > 1 else grp

    # Declared chronological timepoint order (no automatic sorting — a string sort
    # would place "day_10" before "day_5"). Drives both the per-timepoint analyses'
    # output order and the later-vs-earlier comparison pairs (Analyses 3, 6).
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
    # Analysis 1: Within-condition enrichment (binomial)
    # --------------------------------------------------------------
    if args.within_condition == "true":
        print(f"\n{'='*60}\nANALYSIS 1: WITHIN-CONDITION ENRICHMENT (Binomial)\n{'='*60}")
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
        print(f"\n{'='*60}\nANALYSIS 1: WITHIN-CONDITION ENRICHMENT (Binomial)  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # Analysis 2: Between-condition Wilcoxon
    # --------------------------------------------------------------
    if args.between_condition_wilcoxon == "true":
        print(f"\n{'='*60}\nANALYSIS 2: BETWEEN-CONDITION WILCOXON\n{'='*60}")
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
    else:
        print(f"\n{'='*60}\nANALYSIS 2: BETWEEN-CONDITION WILCOXON  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # Analysis 3: Between-timepoint (Wilcoxon pooled across conditions
    #             + Fisher's within each condition)
    # --------------------------------------------------------------
    # The two sub-tests are toggled independently (--between-timepoint-wilcoxon /
    # --between-timepoint-fisher). Within each day-pair, each sub-test is computed
    # and written only if its toggle is on.
    if args.between_timepoint_wilcoxon == "true" or args.between_timepoint_fisher == "true":
        print(f"\n{'='*60}\nANALYSIS 3: BETWEEN-TIMEPOINT\n{'='*60}")

        # All later-vs-earlier day-pairs generated from --timepoints (previously a
        # hard-coded list of three). Each pair runs (a) Wilcoxon pooled across
        # conditions and (b) Fisher's within each condition (pooled replicates).
        # For day_0,day_5,day_10 this reproduces the previous d10_vs_d0, d10_vs_d5,
        # d5_vs_d0 set and order.
        timepoint_pairs = build_timepoint_pairs(timepoint_order)

        for time_a, time_b, tag in timepoint_pairs:
            print(f"\n--- {time_a} vs {time_b} ---")

            # 3a/c/e: Wilcoxon (pooled across conditions, n=4 vs n=4)
            if args.between_timepoint_wilcoxon == "true":
                print(f"  Wilcoxon (pooled across conditions, n=4 vs n=4)")
                df_w_tp = between_timepoint_wilcoxon(
                    replicate_counts, rep_to_timepoint,
                    feature_col=feature_col, time_a=time_a, time_b=time_b,
                )
                n_sig = (df_w_tp["p_adj"] < 0.05).sum() if not df_w_tp.empty else 0
                print(f"    Tests: {len(df_w_tp)}  |  Significant (p_adj<0.05): {n_sig}")
                w_path = out_dir / f"between_timepoint_wilcoxon_{tag}_{suffix}.csv"
                df_w_tp.to_csv(w_path, index=False)
                saved_paths.append(w_path)

            # 3b/d/f: Fisher's within each condition (pool replicates)
            if args.between_timepoint_fisher == "true":
                print(f"  Fisher's exact (within each condition, pooled replicates)")
                print(f"  WARNING: Pooling biological replicates is pseudoreplication.")
                print(f"           P-values are anti-conservative and should be interpreted cautiously.")
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
        print(f"\n{'='*60}\nANALYSIS 3: BETWEEN-TIMEPOINT  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # Analysis 4: Per-timepoint Fisher's
    # --------------------------------------------------------------
    if args.per_timepoint_fisher == "true":
        print(f"\n{'='*60}\nANALYSIS 4: PER-TIMEPOINT FISHER'S EXACT\n"
              f"  NOTE: pooling replicates is pseudoreplication — interpret cautiously\n{'='*60}")
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
        df_fisher.to_csv(fisher_path, index=False)
        saved_paths.append(fisher_path)
    else:
        print(f"\n{'='*60}\nANALYSIS 4: PER-TIMEPOINT FISHER'S EXACT  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # Analysis 5: Per-timepoint background-aware diff (control vs treatment)
    # --------------------------------------------------------------
    # Background-aware counterpart of Analysis 4 (per-timepoint Fisher). For each
    # timepoint it asks whether each unit's enrichment OVER ITS OWN BACKGROUND
    # differs between the two conditions, rather than comparing raw stall-site
    # shares (Fisher). Normalizing each condition to its own background means a
    # shift in the expressed transcriptome between conditions cannot masquerade as
    # differential stalling. Positive delta_log2_enrichment = more enriched vs
    # background in the headline condition.
    #
    # Key design choice: run PER TIMEPOINT, not flat. In the flat consensus design
    # group == condition, so the per-group background keys straight into the test.
    # Here each condition spans several timepoint groups, so we slice to one
    # timepoint at a time and feed that timepoint's own per-group background
    # (e.g. BWM_day_0 vs control_day_0), keeping the background matched within each
    # comparison. FDR is applied per (timepoint, site), mirroring per_timepoint_fisher.
    if args.per_timepoint_background_diff == "true":
        print(f"\n{'='*60}\nANALYSIS 5: PER-TIMEPOINT BACKGROUND-AWARE DIFF\n"
              f"  NOTE: enrichment-over-background ratio; pooling replicates is "
              f"pseudoreplication — interpret cautiously\n{'='*60}")
        if args.headline_condition is not None:
            print(f"  Headline condition: {args.headline_condition} "
                  f"(positive delta_log2_enrichment = more enriched vs background in {args.headline_condition})")
        else:
            print("  Headline condition: alphabetical default "
                  "(positive delta_log2_enrichment = more enriched vs background in the first condition)")

        conditions = sorted(set(rep_to_condition.values()))
        timepoints = timepoint_order

        # Map (condition, timepoint) -> group so each per-timepoint comparison pulls
        # that timepoint's own per-group background frequencies.
        # Will look like:
        #   {("BWM", "day_0"): "BWM_day_0", ("control", "day_0"): "control_day_0", ...}
        cond_tp_to_group = {
            (rep_to_condition[rep], rep_to_timepoint[rep]): grp
            for rep, grp in rep_to_group.items()
        }

        bgdiff_frames = []
        # For each timepoint
        for tp in timepoints:
            # Gather the replicate and their counts for that timepoint
            reps_at_tp = {
                rep: counts for rep, counts in replicate_counts.items()
                if rep_to_timepoint.get(rep) == tp
            }
            # Gets the its background frequencies for the condition at the timepoint
            # In the dataset, there is one background frequency per group
            bg_for_tp = {
                cond: bg_freq_per_group[cond_tp_to_group[(cond, tp)]]
                for cond in conditions
            }
            # Runs the actual test
            df_bgdiff_tp = between_condition_background_diff(
                reps_at_tp, rep_to_condition, bg_for_tp,
                feature_col=feature_col, headline_condition=args.headline_condition,
            )
            if not df_bgdiff_tp.empty:
                # Tag the timepoint as the second column (after "site"), mirroring
                # per_timepoint_fisher's (site, timepoint, ...) layout.
                df_bgdiff_tp.insert(1, "timepoint", tp)
            bgdiff_frames.append(df_bgdiff_tp)

            # Printing purposes only
            n_sig_tp = (df_bgdiff_tp["p_adj"] < 0.05).sum() if not df_bgdiff_tp.empty else 0
            print(f"  [{tp}] {len(df_bgdiff_tp)} tests, {n_sig_tp} significant")

        # Builds the final data frame
        df_bgdiff = pd.concat(bgdiff_frames, ignore_index=True) if bgdiff_frames else pd.DataFrame()
        bgdiff_path = out_dir / f"per_timepoint_background_diff_{suffix}.csv"
        df_bgdiff.to_csv(bgdiff_path, index=False)
        saved_paths.append(bgdiff_path)
    else:
        print(f"\n{'='*60}\nANALYSIS 5: PER-TIMEPOINT BACKGROUND-AWARE DIFF  [SKIPPED]\n{'='*60}")

    # --------------------------------------------------------------
    # Analysis 6: Between-timepoint background-aware diff (pooled across conditions)
    # --------------------------------------------------------------
    # Background-aware counterpart of the between-timepoint Wilcoxon (Analysis 3a):
    # both POOL replicates ACROSS CONDITIONS within each timepoint (ignoring the
    # BWM/control split), but this one normalizes each timepoint to its OWN pooled
    # background (like Analysis 5), so a shift in the expressed transcriptome across
    # the time course cannot masquerade as differential stalling. Direction is
    # later-vs-earlier (time_a vs time_b), independent of --headline-condition (the
    # comparison is between timepoints, not conditions). One combined CSV across the
    # three day-pairs, tagged by a `comparison` column.
    if args.between_timepoint_background_diff == "true":
        print(f"\n{'='*60}\nANALYSIS 6: BETWEEN-TIMEPOINT BACKGROUND-AWARE DIFF (POOLED ACROSS CONDITIONS)\n"
              f"  NOTE: enrichment-over-background ratio; pooling replicates is "
              f"pseudoreplication — interpret cautiously\n{'='*60}")

        timepoints = timepoint_order

        # Count-weighted pooled background per timepoint: sum bg_count across the
        # groups (conditions) at the timepoint, then renormalize to frequencies, so
        # the larger-library condition contributes proportionally more — matching
        # how the foreground stall counts are pooled. group "control_day_0" ->
        # timepoint "day_0".
        # Will look like:
        #   bg_freq_per_timepoint["day_0"] = Series(unit -> pooled bg freq, sums to 1)
        group_to_timepoint = {grp: grp.split("_", 1)[1] for grp in bg_counts_per_group}
        bg_freq_per_timepoint = {}
        bg_total_per_timepoint = {}
        for tp in timepoints:
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
        for tp in timepoints:
            if tp in bg_total_per_timepoint:
                print(f"    [{tp}] {bg_total_per_timepoint[tp]} total {level}s")

        # Same later-vs-earlier day-pairs as Analysis 3, generated from --timepoints.
        timepoint_pairs = build_timepoint_pairs(timepoint_order)

        bgtp_frames = []
        for time_a, time_b, tag in timepoint_pairs:
            print(f"\n--- {time_a} vs {time_b} ---")
            df_bgtp = between_timepoint_background_diff(
                replicate_counts, rep_to_timepoint, bg_freq_per_timepoint,
                feature_col=feature_col, time_a=time_a, time_b=time_b,
            )
            if not df_bgtp.empty:
                # Tag the comparison as the second column (after "site"), mirroring
                # Analysis 5's timepoint tag, so the three day-pairs stack into one CSV.
                df_bgtp.insert(1, "comparison", tag)
            bgtp_frames.append(df_bgtp)

            n_sig = (df_bgtp["p_adj"] < 0.05).sum() if not df_bgtp.empty else 0
            print(f"  [{tag}] {len(df_bgtp)} tests, {n_sig} significant")

        df_bgtp_all = pd.concat(bgtp_frames, ignore_index=True) if bgtp_frames else pd.DataFrame()
        bgtp_path = out_dir / f"between_timepoint_background_diff_{suffix}.csv"
        df_bgtp_all.to_csv(bgtp_path, index=False)
        saved_paths.append(bgtp_path)
    else:
        print(f"\n{'='*60}\nANALYSIS 6: BETWEEN-TIMEPOINT BACKGROUND-AWARE DIFF (POOLED ACROSS CONDITIONS)  [SKIPPED]\n{'='*60}")

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
