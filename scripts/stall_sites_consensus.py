import logging
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import os
import numpy as np
import pandas as pd
from ribopy import Ribo

from ribostall.stall_sites import filter_tx, codonize_counts_cds, call_stalls, consensus_stalls_across_reps, consensus_to_long_df
from ribostall.amino_acids import windows_aa, count_matrix, background_aa_freq, pwm_position_weighted_log2, plot_logo, AA_ORDER, AA_CLASS, CLASS_COLORS
from ribostall.sequence import get_sequence, get_cds_range_lookup

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)

def main():
    parser = argparse.ArgumentParser(
        description="Detect ribosome stall sites"
    )
    parser.add_argument("--pickle", required=True, help="Path to coverage pickle.gz file")
    parser.add_argument("--ribo", required=True, help="Path to ribo file")
    parser.add_argument("--tx_threshold", type=float, default=1.0,
                        help="Minimum reads/nt (in CDS) for filtering transcripts")
    parser.add_argument("--groups", required=True,
                    help="Semicolon-separated group:rep1,rep2 definitions")
    parser.add_argument("--tx_min_reps", type=int, default=2,
                        help="Minimum number of replicates passing threshold for filtering transcripts")
    parser.add_argument("--min_z", type=float, default=1.0,
                        help="Minimum z-score to pass as stall site")
    parser.add_argument("--min_reads", type=int, default=2,
                        help="Minimum number of reads to pass as stall site")
    parser.add_argument("--trim-start", type=int, default=20,
                        help="Exclude first N codons from start (initiation ramp)")
    parser.add_argument("--trim-stop", type=int, default=10,
                        help="Exclude last N codons before stop (termination region)")
    parser.add_argument("--pseudocount", type=float, default=0.5,
                        help="Pseudocount for stall calling")
    parser.add_argument("--stall_min_reps", type=int, default=2,
                        help="Minimum number of replicates that must support a site")
    parser.add_argument("--tol", type=int, default=0,
                        help="Tolerance window for matching sites across reps (same units as indices)")
    parser.add_argument("--min_sep", type=int, default=7,
                        help="Minimum separation between consensus sites; prefer downstream when closer than this")
    parser.add_argument("--out-csv", default="results/stall_sites/motifs/stall_sites.csv", help="Output CSV for stall sites")
    parser.add_argument("--motif", action="store_true", help="Plot motif")
    parser.add_argument("--reference", help="Reference file path")
    parser.add_argument("--flank-left", type=int, default=10, help="Motif")
    parser.add_argument("--flank-right", type=int, default=6, help="Motif")
    parser.add_argument("--psite-offset", type=int, default=0, help="Motif")
    parser.add_argument("--out-png", default="results/stall_sites/motifs/motif.png", help="Motif")
    parser.add_argument("--out-motif-csv", default="results/stall_sites/motifs", help="Output directory for motif PWM CSVs")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Parse experimental groups
    # -------------------------------------------------------------------------
    def parse_groups(groups_arg):
        groups = {}
        for block in groups_arg.split(";"):
            name, reps = block.split(":")
            groups[name] = reps.split(",")
        return groups

    groups = parse_groups(args.groups)
    # --- logging only ---
    logging.info(f"Parsed {len(groups)} groups: {list(groups.keys())}")
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}  # used for display only
    # --- end logging ---

    # -------------------------------------------------------------------------
    # Load coverage data
    # -------------------------------------------------------------------------
    # --- logging only ---
    logging.info(f"Loading coverage from {args.pickle} ...")
    # --- end logging ---
    with gzip.open(args.pickle, "rb") as f:
        cov = pickle.load(f)
    # --- logging only ---
    logging.info(f"Coverage loaded: {len(cov)} experiments, {len(next(iter(cov.values())))} transcripts each")
    # --- end logging ---

    # Filter coverage to only declared replicates
    declared_reps = {rep for reps in groups.values() for rep in reps}
    cov = {exp: tx_dict for exp, tx_dict in cov.items() if exp in declared_reps}
    logging.info(f"Keeping {len(cov)} declared replicates (dropped undeclared samples)")

    # -------------------------------------------------------------------------
    # Load the ribo object
    # -------------------------------------------------------------------------
    # --- logging only ---
    logging.info(f"Loading ribo object from {args.ribo} ...")
    # --- end logging ---
    ribo_object = Ribo(args.ribo, alias=None)
    # --- logging only ---
    logging.info("Ribo object loaded")
    # --- end logging ---

    # -------------------------------------------------------------------------
    # Sanity check: warn if any declared replicate is absent from the coverage
    # -------------------------------------------------------------------------
    missing = [r for rs in groups.values() for r in rs if r not in cov]
    if missing:
        print("Warning: the following replicates are missing from coverage:", ", ".join(missing))

    # -------------------------------------------------------------------------
    # Transcript filtering (per-group, then intersection across all groups)
    # -------------------------------------------------------------------------
    # --- logging only ---
    n_before = len(next(iter(cov.values())))
    print(f"\n{'='*60}")
    print(f"TRANSCRIPT FILTERING (per-group, then intersection)")
    print(f"{'='*60}")
    print(f"Transcripts before filtering: {n_before}")
    print(f"\n  {'Replicate':<25} {'Group':<15} {'Avg cov/tx (reads/nt)':>22} {'SD':>10} {'Total coverage':>16}")
    print(f"  {'-'*25} {'-'*15} {'-'*22} {'-'*10} {'-'*16}")
    for group, reps in groups.items():
        for rep in reps:
            tx_dict = cov[rep]
            means = [np.asarray(v, float).mean() for v in tx_dict.values()]
            avg_cov = np.mean(means) if means else 0.0
            sd_cov = np.std(means) if means else 0.0
            total_cov = sum(np.asarray(v, float).sum() for v in tx_dict.values())
            print(f"  {rep:<25} {group:<15} {avg_cov:>22.4f} {sd_cov:>10.4f} {total_cov:>16,.0f}")
    print()
    # --- end logging ---

    filt_tx_dict = {
        group: filter_tx(cov, reps, min_reps=args.tx_min_reps, threshold=args.tx_threshold,
                         trim_start=args.trim_start, trim_stop=args.trim_stop)
        for group, reps in groups.items()
    }
    # --- logging only ---
    for group, txs in filt_tx_dict.items():
        print(f"  Per-group filter [{group}]: {len(txs)} transcripts  (lost {n_before - len(txs)})")
    # --- end logging ---

    # Intersection across all groups
    filt_tx_set = set.intersection(*(set(v) for v in filt_tx_dict.values())) if filt_tx_dict else set()
    # --- logging only ---
    print(f"  Intersection across all groups: {len(filt_tx_set)} transcripts")
    print(f"{'='*60}\n")
    # --- end logging ---

    # Keep only filtered transcripts in coverage
    cov_filt = {
        exp: {tx: arr for tx, arr in tx_dict.items() if tx in filt_tx_set}
        for exp, tx_dict in cov.items()
    }

    # -------------------------------------------------------------------------
    # Codonize coverage
    # -------------------------------------------------------------------------
    # --- logging only ---
    logging.info("Codonizing coverage arrays ...")
    # --- end logging ---
    codon_cov = {
        exp: {tx: codonize_counts_cds(arr) for tx, arr in tx_dict.items()}
        for exp, tx_dict in cov_filt.items()
    }
    # --- logging only ---
    logging.info("Codonization complete")
    # --- end logging ---

    # -------------------------------------------------------------------------
    # Call stall sites (per replicate)
    # -------------------------------------------------------------------------
    # --- logging only ---
    logging.info(f"Calling stall sites (min_z={args.min_z}, min_reads={args.min_reads}, trim_start={args.trim_start}, trim_stop={args.trim_stop}) ...")
    # --- end logging ---
    stalls = {
        exp: {
            tx: call_stalls(
                arr,
                min_z=args.min_z,
                min_obs=args.min_reads,
                trim_start=args.trim_start,
                trim_stop=args.trim_stop,
                pseudocount=args.pseudocount
            )
            for tx, arr in tx_dict.items()
        }
        for exp, tx_dict in codon_cov.items()
    }
    # --- logging only ---
    logging.info("Stall calling complete")
    print(f"\n{'='*60}")
    print(f"STALL SITE CALLING (per replicate)")
    print(f"{'='*60}")
    rep_total_counts = {exp: sum(len(sites) for sites in tx_stalls.values()) for exp, tx_stalls in stalls.items()}
    for exp, count in rep_total_counts.items():
        grp = rep_to_group.get(exp, "?")
        print(f"  [{exp}] ({grp}): {count} stall sites  ({len(codon_cov[exp])} transcripts)")
    print(f"  Total across all replicates: {sum(rep_total_counts.values())} stall sites")
    print(f"{'='*60}\n")
    # --- end logging ---

    # -------------------------------------------------------------------------
    # Consensus stalls per group
    # -------------------------------------------------------------------------
    # --- logging only ---
    logging.info(f"Computing consensus stalls (min_support={args.stall_min_reps}, tol={args.tol}, min_sep={args.min_sep}) ...")
    # --- end logging ---
    consensus = {
        group: consensus_stalls_across_reps(
            stalls,
            reps,
            min_support=args.stall_min_reps,
            tol=args.tol,
            min_sep=args.min_sep
        )
        for group, reps in groups.items()
    }
    # --- logging only ---
    logging.info("Consensus calling complete")
    print(f"\n{'='*60}")
    print(f"CONSENSUS STALL SITES (per group)")
    print(f"{'='*60}")
    consensus_counts = {group: sum(len(sites) for sites in grp_consensus.values()) for group, grp_consensus in consensus.items()}
    for group, count in consensus_counts.items():
        n_tx = len([tx for tx, sites in consensus[group].items() if sites])
        print(f"  [{group}]: {count} consensus stall sites  ({n_tx} transcripts with stalls)")
    print(f"  Total across all groups: {sum(consensus_counts.values())} consensus stall sites")
    print(f"{'='*60}\n")
    # --- end logging ---

    # -------------------------------------------------------------------------
    # Write consensus stall sites to CSV
    # -------------------------------------------------------------------------
    # --- logging only ---
    print(f"\n{'='*60}")
    print(f"WRITING RESULTS")
    print(f"{'='*60}")
    # --- end logging ---
    df = consensus_to_long_df(consensus)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    # --- logging only ---
    logging.info(f"Saved stall sites to {args.out_csv}")
    print(f"  Saved: {args.out_csv}  ({len(df)} rows)")
    print(f"{'='*60}\n")
    # --- end logging ---

    # Motif
    if args.motif:
        reference_file_path = args.reference

        # --- logging only ---
        print(f"\n{'='*60}")
        print(f"LOADING SEQUENCES")
        print(f"{'='*60}")
        logging.info("Looking up CDS ranges ...")
        # --- end logging ---
        cds_range = get_cds_range_lookup(ribo_object)
        # --- logging only ---
        logging.info(f"CDS ranges loaded for {len(cds_range)} transcripts")
        print(f"  CDS ranges: {len(cds_range)} transcripts")
        logging.info(f"Loading sequences from {reference_file_path} ...")
        # --- end logging ---
        sequence = get_sequence(ribo_object, reference_file_path, alias=None)
        # --- logging only ---
        logging.info(f"Sequences loaded for {len(sequence)} transcripts")
        print(f"  Sequences: {len(sequence)} transcripts")
        print(f"{'='*60}\n")
        print(f"\n{'='*60}")
        print(f"MOTIF ANALYSIS")
        print(f"{'='*60}")
        # --- end logging ---
        def compute_W_for_group(g):
            stalls = consensus[g]
            win = windows_aa(consensus[g], cds_range, sequence,
                        flank_left=args.flank_left, flank_right=args.flank_right, psite_offset_codons=args.psite_offset)
            counts = count_matrix(win, AA_ORDER, flank_left=args.flank_left, flank_right=args.flank_right)
            bg = background_aa_freq(consensus[g].keys(), cds_range, sequence, AA_ORDER)
            return pwm_position_weighted_log2(counts, bg, pseudocount=args.pseudocount)

        # compute all, then unify y-limits for fair visual comparison
        W_by_group = {g: compute_W_for_group(g) for g in groups.keys()}

        # per-position heights (sum over amino acids)
        ymax = max(
            W.loc[:, pos][W.loc[:, pos] > 0].sum()
            for g, W in W_by_group.items()
            for pos in W.columns
        )
        ymin = max(
            abs(W.loc[:, pos][W.loc[:, pos] < 0].sum())
            for g, W in W_by_group.items()
            for pos in W.columns
        )

        # plot side-by-side
        fig, axes = plt.subplots(1, len(groups.keys()), figsize=(5*len(groups.keys()), 5), sharey=True)
        if len(groups.keys()) == 1:
            axes = [axes]

        for ax, g in zip(axes, groups.keys()):
            plt.sca(ax)
            plot_logo(W_by_group[g],
                    title=f"{g.capitalize()}",
                    aa_class=AA_CLASS)
            ax.set_ylim(-ymin, ymax)   # same scale across panels
        
        for ax in axes[1:]:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)

        patches = [mpatches.Patch(color=c, label=cls) for cls, c in CLASS_COLORS.items()]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.17)
        fig.savefig(args.out_png, dpi=600)
        # --- logging only ---
        logging.info(f"Saved image to {args.out_png}")
        print(f"  Saved: {args.out_png}")
        # --- end logging ---

        os.makedirs(args.out_motif_csv, exist_ok=True)
        for g, W in W_by_group.items():
            # Save PWM (AA x position)
            pwm_csv = os.path.join(args.out_motif_csv, f"{g}_pwm_log2_enrichment.csv")
            W.to_csv(pwm_csv)
            # --- logging only ---
            print(f"  Saved: {pwm_csv}")
            # --- end logging ---
        # --- logging only ---
        logging.info(f"Saved PWM CSVs to {args.out_motif_csv}")
        print(f"{'='*60}\n")
        # --- end logging ---

if __name__ == "__main__":
    main()