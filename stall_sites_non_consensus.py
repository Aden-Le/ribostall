

import logging
import argparse
import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import re
import json, os
from pathlib import Path
import ribopy
from ribopy import Ribo
from functions_stall_sites import filter_tx, codonize_counts_cds, call_stalls, stalls_to_long_df
from functions_AA import translate_cds_nt_to_aa, windows_aa, count_matrix, background_aa_freq, pwm_position_weighted_log2, plot_logo, CODON2AA, AA_ORDER, AA_CLASS, CLASS_COLORS
from functions import get_sequence, get_cds_range_lookup

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
    parser.add_argument("--trim_edges", type=int, default=10,
                        help="Trim x number of codons after start and before stop")
    parser.add_argument("--pseudocount", type=float, default=0.5,
                        help="Pseudocount for stall calling")
    parser.add_argument("--out-json", default="../ribostall_results/stall_sites.jsonl", help="JSON")
    parser.add_argument("--motif", action="store_true", help="Plot motif")
    parser.add_argument("--reference", help="Reference file path")
    parser.add_argument("--flank-left", type=int, default=10, help="Motif")
    parser.add_argument("--flank-right", type=int, default=6, help="Motif")
    parser.add_argument("--psite-offset", type=int, default=0, help="Motif")
    parser.add_argument("--out-csv", default="../ribostall_results/motif_csv", help="Motif")

    args = parser.parse_args()

    # Rep groups
    def parse_groups(groups_arg):
        groups = {}
        for block in groups_arg.split(";"):
            name, reps = block.split(":")
            groups[name] = reps.split(",")
        return groups
    groups = parse_groups(args.groups)
    logging.info(f"Parsed {len(groups)} groups: {list(groups.keys())}")

    # Load coverage
    logging.info(f"Loading coverage from {args.pickle} ...")
    with gzip.open(args.pickle, "rb") as f:
        cov = pickle.load(f)
    logging.info(f"Coverage loaded: {len(cov)} experiments, {len(next(iter(cov.values())))} transcripts each")

    remove_coverage_list = ["BWM_day0_rep1", "BWM_day5_rep1", "BWM_day10_rep1", "control_day0_rep1", "control_day5_rep1", "control_day10_rep1"]
    removed = [k for k in remove_coverage_list if k in cov]
    for key in removed:
        del cov[key]
    if removed:
        logging.info(f"Removed rep1 experiments: {removed}")

    # Load ribo object (adjust alias to your organism as needed)
    logging.info(f"Loading ribo object from {args.ribo} ...")
    ribo_object = Ribo(args.ribo, alias=None)
    logging.info("Ribo object loaded")

    # (Optional) quick sanity check that all reps exist in coverage
    missing = [r for rs in groups.values() for r in rs if r not in cov]
    if missing:
        print("Warning: the following replicates are missing from coverage:", ", ".join(missing))

    # Filter transcripts
    n_before = len(next(iter(cov.values())))
    print(f"\n{'='*60}")
    print(f"TRANSCRIPT FILTERING")
    print(f"{'='*60}")
    print(f"Transcripts before filtering: {n_before}")

    filt_tx_dict = {
        group: filter_tx(cov, reps, min_reps=args.tx_min_reps, threshold=args.tx_threshold)
        for group, reps in groups.items()
    }
    for group, txs in filt_tx_dict.items():
        print(f"  After per-group filter [{group}]: {len(txs)} transcripts  (lost {n_before - len(txs)})")

    # Intersection across tissues
    filt_tx_set = set.intersection(*(set(v) for v in filt_tx_dict.values())) if filt_tx_dict else set()
    print(f"After intersection across all groups: {len(filt_tx_set)} transcripts  (lost {n_before - len(filt_tx_set)} total)")
    print(f"{'='*60}\n")

    # Keep only filtered transcripts in coverage
    logging.info("Filtering coverage to intersection transcripts ...")
    cov_filt = {
        exp: {tx: arr for tx, arr in tx_dict.items() if tx in filt_tx_set}
        for exp, tx_dict in cov.items()
    }
    logging.info(f"Coverage filtered: {len(next(iter(cov_filt.values())))} transcripts retained per experiment")

    # Codonize counts
    logging.info("Codonizing coverage arrays ...")
    codon_cov = {
        exp: {tx: codonize_counts_cds(arr) for tx, arr in tx_dict.items()}
        for exp, tx_dict in cov_filt.items()
    }
    logging.info("Codonization complete")

    # Identify stall sites per experiment
    logging.info(f"Calling stall sites (min_z={args.min_z}, min_reads={args.min_reads}, trim_edges={args.trim_edges}) ...")
    stalls = {
        exp: {
            tx: call_stalls(
                arr,
                min_z=args.min_z,
                min_obs=args.min_reads,
                trim_edges=args.trim_edges,
                pseudocount=args.pseudocount
            )
            for tx, arr in tx_dict.items()
        }
        for exp, tx_dict in codon_cov.items()
    }
    logging.info("Stall calling complete")

    # Print
    print(f"Number of filtered transcripts: {len(filt_tx_set)}")
    total_counts = {
        exp: sum(len(idxs) for idxs in tx_stalls.values())
        for exp, tx_stalls in stalls.items()
    }
    print(f"Number of total stall sites per experiment: {total_counts}")

    # JSON
    logging.info(f"Converting stalls to long-format dataframe ...")
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    df = stalls_to_long_df(stalls, rep_to_group)
    logging.info(f"Long-format dataframe: {len(df)} rows")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing JSON to {args.out_json} ...")
    with open(args.out_json, "w") as f:
        for rec in df.to_dict(orient="records"):
            json.dump(rec, f)
            f.write("\n")
    logging.info(f"Saved JSON to {args.out_json}")

    # Motif
    if args.motif:
        logging.info("=== MOTIF ANALYSIS ===")
        reference_file_path = args.reference
        logging.info("Looking up CDS ranges ...")
        cds_range = get_cds_range_lookup(ribo_object)
        logging.info(f"CDS ranges loaded for {len(cds_range)} transcripts")
        logging.info(f"Loading sequences from {reference_file_path} ...")
        sequence = get_sequence(ribo_object, reference_file_path, alias = ribopy.api.alias.apris_human_alias)
        logging.info(f"Sequences loaded for {len(sequence)} transcripts")

        def compute_W_for_exp(exp):
            exp_stalls = {tx: [d["index"] for d in site_list] for tx, site_list in stalls[exp].items()}
            n_sites = sum(len(v) for v in exp_stalls.values())
            logging.info(f"  [{exp}] Building AA windows ({n_sites} stall sites) ...")
            win = windows_aa(exp_stalls, cds_range, sequence,
                        flank_left=args.flank_left, flank_right=args.flank_right, psite_offset_codons=args.psite_offset)
            logging.info(f"  [{exp}] {len(win)} windows built; computing count matrix ...")
            counts = count_matrix(win, AA_ORDER, flank_left=args.flank_left, flank_right=args.flank_right)
            logging.info(f"  [{exp}] Computing background AA frequencies ...")
            bg, bg_counts = background_aa_freq(exp_stalls.keys(), cds_range, sequence, AA_ORDER)
            logging.info(f"  [{exp}] Computing PWM ...")
            W = pwm_position_weighted_log2(counts, bg, pseudocount=args.pseudocount)
            logging.info(f"  [{exp}] Done")
            return W, counts, bg, bg_counts

        W_by_exp = {}
        counts_by_exp = {}
        bg_by_exp = {}
        bg_counts_by_exp = {}
        n_exps = len(stalls)
        for i, exp in enumerate(stalls.keys(), 1):
            logging.info(f"Processing experiment {i}/{n_exps}: {exp}")
            W, counts, bg, bg_counts = compute_W_for_exp(exp)
            W_by_exp[exp] = W
            counts_by_exp[exp] = counts
            bg_by_exp[exp] = bg
            bg_counts_by_exp[exp] = bg_counts

        os.makedirs(args.out_csv, exist_ok=True)
        logging.info(f"Saving CSVs to {args.out_csv} ...")
        for exp, W in W_by_exp.items():
            pwm_csv = os.path.join(args.out_csv, f"{exp}_pwm_log2_enrichment.csv")
            W.to_csv(pwm_csv)
            counts_csv = os.path.join(args.out_csv, f"{exp}_counts.csv")
            counts_by_exp[exp].to_csv(counts_csv)
            bg_csv = os.path.join(args.out_csv, f"{exp}_background.csv")
            bg_by_exp[exp].to_csv(bg_csv)
            bg_counts_csv = os.path.join(args.out_csv, f"{exp}_background_counts.csv")
            bg_counts_by_exp[exp].to_csv(bg_counts_csv)
            logging.info(f"  Saved CSVs for {exp}")
        logging.info(f"All CSVs saved to {args.out_csv}")

if __name__ == "__main__":
    main()