import logging
import argparse
import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import sys
import datetime
import re
import json, os
import pandas as pd
import numpy as np
from pathlib import Path
import ribopy
from ribopy import Ribo
from functions_stall_sites import filter_tx, codonize_counts_cds, call_stalls, consensus_stalls_across_reps, parse_key, consensus_to_long_df, stalls_to_long_df
from functions_AA import translate_cds_nt_to_aa, windows_aa, count_matrix, background_aa_freq, pwm_position_weighted_log2, plot_logo, CODON2AA, AA_ORDER, AA_CLASS, CLASS_COLORS, epa_triplet_counts, epa_enrichment, epa_pairwise_matrix, plot_top_triplets_multi
from functions import get_sequence, get_cds_range_lookup

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)

def main():
    parser = argparse.ArgumentParser(description="Detect ribosome stall sites")
    parser.add_argument("--pickle", required=True, help="Path to coverage pickle.gz file")
    parser.add_argument("--ribo", required=True, help="Path to ribo file")
    parser.add_argument("--tx_threshold", type=float, default=1.0)
    parser.add_argument("--groups", required=True, help="Semicolon-separated group:rep1,rep2 definitions")
    parser.add_argument("--tx_min_reps", type=int, default=2)
    parser.add_argument("--min_z", type=float, default=1.0)
    parser.add_argument("--min_reads", type=int, default=2)
    parser.add_argument("--trim_edges", type=int, default=10)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument("--stall_min_reps", type=int, default=2)
    parser.add_argument("--tol", type=int, default=0)
    parser.add_argument("--min_sep", type=int, default=7)
    parser.add_argument("--motif", action="store_true")
    parser.add_argument("--reference", help="Reference file path")
    parser.add_argument("--flank-left", type=int, default=10)
    parser.add_argument("--flank-right", type=int, default=6)
    parser.add_argument("--psite-offset", type=int, default=0)
    parser.add_argument("--out-dir", default="../ribostall_results", help="Output directory")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # helper
    def parse_groups(groups_arg):
        groups = {}
        for block in groups_arg.split(";"):
            name, reps = block.split(":")
            groups[name] = reps.split(",")
        return groups

    groups = parse_groups(args.groups)

    # Load coverage + Ribo
    with gzip.open(args.pickle, "rb") as f:
        cov = pickle.load(f)
    ribo_object = Ribo(args.ribo, alias=ribopy.api.alias.apris_human_alias)

    # Filter transcripts
    filt_tx_dict = {
        group: filter_tx(cov, reps, min_reps=args.tx_min_reps, threshold=args.tx_threshold)
        for group, reps in groups.items()
    }
    filt_tx_set = set.intersection(*(set(v) for v in filt_tx_dict.values())) if filt_tx_dict else set()
    cov_filt = {exp: {tx: arr for tx, arr in tx_dict.items() if tx in filt_tx_set}
                for exp, tx_dict in cov.items()}

    # Codonize and call stalls
    codon_cov = {exp: {tx: codonize_counts_cds(arr) for tx, arr in tx_dict.items()}
                 for exp, tx_dict in cov_filt.items()}
    stalls = {exp: {tx: call_stalls(arr, min_z=args.min_z, min_obs=args.min_reads,
                                    trim_edges=args.trim_edges, pseudocount=args.pseudocount)
                    for tx, arr in tx_dict.items()}
              for exp, tx_dict in codon_cov.items()}

    consensus = {
        group: consensus_stalls_across_reps(stalls, reps, min_support=args.stall_min_reps,
                                            tol=args.tol, min_sep=args.min_sep)
        for group, reps in groups.items()
    }

    # Print + summary info
    print(f"Number of filtered transcripts: {len(filt_tx_set)}")
    total_counts = {g: sum(len(idxs) for idxs in stalls.values()) for g, stalls in consensus.items()}
    print(f"Number of total stall sites per group: {total_counts}")

    # Save JSON lines of stall sites
    jsonl_path = out_dir / "stall_sites.jsonl"
    df = consensus_to_long_df(consensus)
    with open(jsonl_path, "w") as f:
        for rec in df.to_dict(orient="records"):
            json.dump(rec, f)
            f.write("\n")
    logging.info(f"Saved JSONL to {jsonl_path}")

    # Map replicate -> group for convenience
    rep_to_group = {rep: g for g, reps in groups.items() for rep in reps}

    # Save per-replicate stall sites as JSONL
    perrep_df = stalls_to_long_df(stalls, rep_to_group=rep_to_group)
    jsonl_reps_path = out_dir / "stall_sites_by_replicate.jsonl"
    with open(jsonl_reps_path, "w") as f:
        for rec in perrep_df.to_dict(orient="records"):
            json.dump(rec, f)
            f.write("\n")
    logging.info(f"Saved per-replicate stall sites JSONL to {jsonl_reps_path}")










    # === Motif ===
    if args.motif:
        reference_file_path = args.reference
        cds_range = get_cds_range_lookup(ribo_object)
        sequence = get_sequence(ribo_object, reference_file_path,
                                alias=ribopy.api.alias.apris_human_alias)

        def compute_matrices(g):
            win = windows_aa(consensus[g], cds_range, sequence,
                             flank_left=args.flank_left,
                             flank_right=args.flank_right,
                             psite_offset_codons=args.psite_offset)
            counts = count_matrix(win, AA_ORDER,
                                  flank_left=args.flank_left,
                                  flank_right=args.flank_right)
            bg = background_aa_freq(consensus[g].keys(), cds_range, sequence, AA_ORDER)
            W = pwm_position_weighted_log2(counts, bg, pseudocount=args.pseudocount)
            return counts, W

        counts_by_group, W_by_group = {}, {}
        for g in groups.keys():
            counts_by_group[g], W_by_group[g] = compute_matrices(g)

        # y-limits unified across all panels
        ymax = max(W.loc[:, pos][W.loc[:, pos] > 0].sum()
                   for W in W_by_group.values() for pos in W.columns)
        ymin = max(abs(W.loc[:, pos][W.loc[:, pos] < 0].sum())
                   for W in W_by_group.values() for pos in W.columns)

        # === Plot logos ===
        fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 5), sharey=True)
        if len(groups) == 1:
            axes = [axes]

        for ax, g in zip(axes, groups.keys()):
            plot_logo(W_by_group[g], title=g.capitalize(), aa_class=AA_CLASS, ax=ax)
            ax.set_ylim(-ymin, ymax)
        for ax in axes[1:]:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)

        patches = [mpatches.Patch(color=c, label=cls) for cls, c in CLASS_COLORS.items()]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches), bbox_to_anchor=(0.5,0.05))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.17)

        out_png = out_dir / "motif_logo.png"
        fig.savefig(out_png, dpi=600)
        logging.info(f"Saved motif image to {out_png}")

        # === Save matrices ===
        motif_pwm_dir = out_dir / "motif_pwm"
        motif_counts_dir = out_dir / "motif_counts"
        motif_pwm_dir.mkdir(exist_ok=True)
        motif_counts_dir.mkdir(exist_ok=True)

        for g in groups.keys():
            W_by_group[g].to_csv(motif_pwm_dir / f"{g}_pwm_log2_enrichment.csv")
            counts_by_group[g].to_csv(motif_counts_dir / f"{g}_raw_counts.csv")
        logging.info(f"Saved PWM and count matrices to {motif_pwm_dir} / {motif_counts_dir}")

        # === EPA TRIPLET ANALYSIS ===
        logging.info("Computing E-P-A triplet enrichment...")

        cds_seq = get_sequence(ribo_object, args.reference, alias=ribopy.api.alias.apris_human_alias)
        offset = args.psite_offset

        epa_counts_by_group = {}
        epa_enrich_by_group = {}

        for g in groups.keys():
            # 1. Compute stall-only EPA counts for this group
            counts_epa_g, countsE_g, countsP_g, countsA_g = epa_triplet_counts(
                consensus_group=consensus[g],
                cds_range=cds_range,
                sequence=sequence,
                psite_offset_codons=args.psite_offset,
                basis="P",
                drop_stop_windows=True
            )

            # 2. Compute background and enrichment
            bg_aa_g = background_aa_freq(
                transcripts=consensus[g].keys(),
                cds_range=cds_range,
                sequence=sequence,
                aa_order=AA_ORDER
            )
            enrich_g = epa_enrichment(counts_epa_g, bg_aa_g, pseudocount=args.pseudocount)

            # 3. Store
            epa_counts_by_group[g] = counts_epa_g
            epa_enrich_by_group[g] = enrich_g

            # 4. Save raw counts and enrichment
            epa_dir = out_dir / "epa_triplets"
            epa_dir.mkdir(exist_ok=True)

            counts_path = epa_dir / f"{g}_epa_counts.csv"
            enrich_path = epa_dir / f"{g}_epa_enrichment.csv"
            counts_epa_g.to_csv(counts_path)
            enrich_g.to_csv(enrich_path)
            logging.info(f"Saved {g} EPA counts → {counts_path}")
            logging.info(f"Saved {g} EPA enrichment → {enrich_path}")

    # === Save summary JSON ===
    summary = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "parameters": {
            "pickle": args.pickle,
            "ribo": args.ribo,
            "reference": args.reference,
            "groups": args.groups,
            "tx_threshold": args.tx_threshold,
            "tx_min_reps": args.tx_min_reps,
            "min_z": args.min_z,
            "min_reads": args.min_reads,
            "trim_edges": args.trim_edges,
            "pseudocount": args.pseudocount,
            "stall_min_reps": args.stall_min_reps,
            "tol": args.tol,
            "min_sep": args.min_sep,
            "flank_left": args.flank_left,
            "flank_right": args.flank_right,
            "psite_offset": args.psite_offset,
        },
        "filtered_transcripts": len(filt_tx_set),
        "total_stall_sites": total_counts,
        "groups": list(groups.keys()),
        "motif_done": args.motif,
        "output_files": {
            "stall_sites_jsonl": str(jsonl_path),
            "motif_logo_png": str(out_png if args.motif else None),
            "motif_pwm_dir": str(motif_pwm_dir if args.motif else None),
            "motif_counts_dir": str(motif_counts_dir if args.motif else None)
        }
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved summary (with parameters and command) to {summary_path}")


if __name__ == "__main__":
    main()