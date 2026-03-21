import logging
import argparse
import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import json, os
import pandas as pd
from pathlib import Path
import ribopy
from ribopy import Ribo

# Local modules with the core analysis functions
from functions_stall_sites import filter_tx, codonize_counts_cds, call_stalls, stalls_to_long_df
from functions_AA import (translate_cds_nt_to_aa, windows_aa, count_matrix, background_aa_freq,
                          pwm_position_weighted_log2, plot_logo, epa_triplet_counts,
                          CODON2AA, AA_ORDER, AA_CLASS, CLASS_COLORS)
from functions import get_sequence, get_cds_range_lookup
from functions_enrichment import within_condition_enrichment, between_condition_wilcoxon, per_timepoint_fisher

# =========================
# Logging
# =========================
# Set up logging to print timestamped messages to the console.
# Format: "2025-01-01 12:00:00  INFO  MainProcess  message"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)

def main():
    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
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
    parser.add_argument("--min_z", type=float, default=2.0,
                        help="Minimum z-score to pass as stall site")
    parser.add_argument("--min_reads", type=int, default=5,
                        help="Minimum number of reads to pass as stall site")
    parser.add_argument("--trim-start", type=int, default=20,
                        help="Exclude first N codons from start (initiation ramp)")
    parser.add_argument("--trim-stop", type=int, default=10,
                        help="Exclude last N codons before stop (termination region)")
    parser.add_argument("--pseudocount", type=float, default=0.5,
                        help="Pseudocount for stall calling")
    parser.add_argument("--out-json", default="../ribostall_results/stall_sites.jsonl", help="JSON")
    parser.add_argument("--motif", action="store_true", help="Plot motif")
    parser.add_argument("--reference", help="Reference file path")
    parser.add_argument("--flank-left", type=int, default=10, help="Motif")
    parser.add_argument("--flank-right", type=int, default=6, help="Motif")
    parser.add_argument("--psite-offset", type=int, default=0, help="Motif")
    parser.add_argument("--out-csv", default="../ribostall_results/motif_csv", help="Motif")
    parser.add_argument("--enrichment", action="store_true",
                        help="Run amino acid enrichment analysis at E/P/A sites")
    parser.add_argument("--out-enrichment", default="../ribostall_results/enrichment",
                        help="Output directory for enrichment analysis CSVs")

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Parse experimental groups
    # -------------------------------------------------------------------------
    # Converts the CLI string "groupA:rep1,rep2;groupB:rep3,rep4" into a dict:
    #   {"groupA": ["rep1", "rep2"], "groupB": ["rep3", "rep4"]}
    # This mapping is used throughout to know which experiments belong together.
    def parse_groups(groups_arg):
        groups = {}
        for block in groups_arg.split(";"):
            name, reps = block.split(":")
            groups[name] = reps.split(",")
        return groups
    groups = parse_groups(args.groups)
    logging.info(f"Parsed {len(groups)} groups: {list(groups.keys())}")

    # -------------------------------------------------------------------------
    # Load coverage data
    # -------------------------------------------------------------------------
    # The pickle file holds a nested dict:
    #   cov[experiment_name][transcript_name] = numpy array of per-nt read counts
    # It's gzipped to save disk space, so we open with gzip first.
    logging.info(f"Loading coverage from {args.pickle} ...")
    with gzip.open(args.pickle, "rb") as f:
        cov = pickle.load(f)
    logging.info(f"Coverage loaded: {len(cov)} experiments, {len(next(iter(cov.values())))} transcripts each")

    # -------------------------------------------------------------------------
    # Remove known bad replicates (rep1 experiments)
    # -------------------------------------------------------------------------
    # These specific rep1 samples are excluded from analysis (e.g. due to QC
    # failure or a deliberate design decision to use only rep2+).
    # We build the list first so we can log exactly what was removed.
    remove_coverage_list = ["BWM_day0_rep1", "BWM_day5_rep1", "BWM_day10_rep1", "control_day0_rep1", "control_day5_rep1", "control_day10_rep1"]
    removed = [k for k in remove_coverage_list if k in cov]
    for key in removed:
        del cov[key]
    if removed:
        logging.info(f"Removed rep1 experiments: {removed}")

    # -------------------------------------------------------------------------
    # Load the ribo object
    # -------------------------------------------------------------------------
    # ribopy.Ribo wraps the HDF5 .ribo file and provides easy access to
    # transcript metadata, CDS annotations, and sequences.
    # alias=None means we use raw transcript names without any remapping.
    logging.info(f"Loading ribo object from {args.ribo} ...")
    ribo_object = Ribo(args.ribo, alias=None)
    logging.info("Ribo object loaded")

    # -------------------------------------------------------------------------
    # Sanity check: warn if any declared replicate is absent from the coverage
    # -------------------------------------------------------------------------
    # Flattens all replicates across all groups into one list and checks each
    # against the coverage dict keys.
    missing = [r for rs in groups.values() for r in rs if r not in cov]
    if missing:
        print("Warning: the following replicates are missing from coverage:", ", ".join(missing))

    # -------------------------------------------------------------------------
    # Transcript filtering (per-group, NO intersection across groups)
    # -------------------------------------------------------------------------
    # We only want to analyse transcripts that are reliably expressed.
    # Each group keeps its own filtered transcript set independently.
    # This avoids dropping transcripts uniquely expressed in one condition.
    n_before = len(next(iter(cov.values())))
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}

    print(f"\n{'='*60}")
    print(f"TRANSCRIPT FILTERING (per-group, no intersection)")
    print(f"{'='*60}")
    print(f"Transcripts before filtering: {n_before}")

    # Per-group filter: for each group, call filter_tx which keeps only
    # transcripts that exceed --tx_threshold reads/nt in at least
    # --tx_min_reps replicates within that group.
    # Result: filt_tx_dict[group] = set of transcript names passing that group.
    filt_tx_dict = {
        group: set(filter_tx(cov, reps, min_reps=args.tx_min_reps, threshold=args.tx_threshold))
        for group, reps in groups.items()
    }
    for group, txs in filt_tx_dict.items():
        print(f"  Per-group filter [{group}]: {len(txs)} transcripts  (lost {n_before - len(txs)})")

    # Union of all per-group filtered transcripts (for background AA frequencies)
    filt_tx_union = set.union(*filt_tx_dict.values()) if filt_tx_dict else set()
    print(f"Union across all groups: {len(filt_tx_union)} unique transcripts")
    print(f"{'='*60}\n")

    # Rebuild the coverage dict — each experiment uses its GROUP's filtered transcripts.
    # Structure preserved: cov_filt[exp][tx] = array
    logging.info("Filtering coverage to per-group transcripts ...")
    cov_filt = {}
    for exp, tx_dict in cov.items():
        grp = rep_to_group.get(exp)
        if grp is None:
            continue
        grp_txs = filt_tx_dict[grp]
        cov_filt[exp] = {tx: arr for tx, arr in tx_dict.items() if tx in grp_txs}
    for exp in cov_filt:
        logging.info(f"  [{exp}] {len(cov_filt[exp])} transcripts retained")

    # -------------------------------------------------------------------------
    # Codonize coverage
    # -------------------------------------------------------------------------
    # The raw coverage arrays are nucleotide-resolution (one value per nt).
    # codonize_counts_cds sums every 3 consecutive nt positions into a single
    # codon-level count, restricted to the CDS region.
    # Result: codon_cov[exp][tx] = 1-D array of length (CDS_len / 3)
    logging.info("Codonizing coverage arrays ...")
    codon_cov = {
        exp: {tx: codonize_counts_cds(arr) for tx, arr in tx_dict.items()}
        for exp, tx_dict in cov_filt.items()
    }
    logging.info("Codonization complete")

    # -------------------------------------------------------------------------
    # Call stall sites
    # -------------------------------------------------------------------------
    # For each experiment and transcript, call_stalls identifies codon positions
    # with unusually high ribosome occupancy by computing a z-score across the
    # transcript's codon array. A position is a stall if:
    #   - z-score >= --min_z  (signal is far above the transcript mean)
    #   - raw count >= --min_reads  (enough evidence, not just noise)
    #   - not within --trim_start codons of the start or --trim_stop of the stop
    # Returns a list of dicts, e.g. [{"index": 42, "z": 3.1, "reads": 7}, ...]
    # stalls[exp][tx] = list of stall-site dicts
    logging.info(f"Calling stall sites (min_z={args.min_z}, min_reads={args.min_reads}, trim_start={args.trim_start}, trim_stop={args.trim_stop}) ...")
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
    logging.info("Stall calling complete")

    # Print a quick summary: total number of stall sites detected per experiment.
    print(f"\n{'='*60}")
    print(f"STALL SITE CALLING")
    print(f"{'='*60}")
    total_counts = {
        exp: sum(len(idxs) for idxs in tx_stalls.values())
        for exp, tx_stalls in stalls.items()
    }
    for exp, count in total_counts.items():
        grp = rep_to_group.get(exp, "?")
        print(f"  [{exp}] ({grp}): {count} stall sites  ({len(codon_cov[exp])} transcripts)")
    print(f"  Total across all experiments: {sum(total_counts.values())} stall sites")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Write stall sites to JSON Lines
    # -------------------------------------------------------------------------
    # stalls_to_long_df converts the nested stalls dict into a tidy pandas
    # DataFrame where each row is one stall site, with columns for experiment,
    # group, transcript, codon index, z-score, reads, etc.
    # rep_to_group inverts the groups dict so we can tag each row with its
    # biological group label.
    logging.info(f"Converting stalls to long-format dataframe ...")
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    df = stalls_to_long_df(stalls, rep_to_group)
    logging.info(f"Long-format dataframe: {len(df)} rows")

    # Create the output directory if it doesn't exist, then write one JSON
    # object per line (JSON Lines format) — easy to stream/parse later.
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing JSON to {args.out_json} ...")
    with open(args.out_json, "w") as f:
        for rec in df.to_dict(orient="records"):
            json.dump(rec, f)
            f.write("\n")
    logging.info(f"Saved JSON to {args.out_json}")

    # -------------------------------------------------------------------------
    # Load CDS ranges and sequences (shared by --motif and --enrichment)
    # -------------------------------------------------------------------------
    cds_range = None
    sequence = None
    if args.motif or args.enrichment:
        reference_file_path = args.reference

        print(f"\n{'='*60}")
        print(f"LOADING SEQUENCES")
        print(f"{'='*60}")

        logging.info("Looking up CDS ranges ...")
        cds_range = get_cds_range_lookup(ribo_object)
        logging.info(f"CDS ranges loaded for {len(cds_range)} transcripts")
        print(f"  CDS ranges: {len(cds_range)} transcripts")

        logging.info(f"Loading sequences from {reference_file_path} ...")
        sequence = get_sequence(ribo_object, reference_file_path, alias = ribopy.api.alias.apris_human_alias)
        logging.info(f"Sequences loaded for {len(sequence)} transcripts")
        print(f"  Sequences: {len(sequence)} transcripts")
        print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Optional: amino acid motif analysis
    # -------------------------------------------------------------------------
    if args.motif:

        def compute_W_for_exp(exp):
            """
            For a single experiment, compute the amino acid PWM (position weight
            matrix) over a window centred on each stall site.

            Steps:
            1. Extract the codon indices of all stall sites for this experiment.
            2. Build amino acid windows: for each stall site, translate the CDS
               window of (flank_left + 1 + flank_right) codons around the stall
               codon into amino acids.
            3. Build a count matrix: rows = amino acid, columns = window position.
            4. Compute background amino acid frequencies from all expressed CDS
               sequences (not just stall sites) — this is the expected frequency
               if there were no enrichment.
            5. Compute a log2 odds PWM: log2(observed / expected), which shows
               which amino acids are over- or under-represented at each position.
            """
            # Flatten stalls[exp] from {tx: [{"index": i, ...}, ...]} to
            # {tx: [i, ...]} — just the codon indices, not the full dicts.
            exp_stalls = {tx: [d["index"] for d in site_list] for tx, site_list in stalls[exp].items()}
            n_sites = sum(len(v) for v in exp_stalls.values())

            # windows_aa translates each stall-site window into a list of amino
            # acid strings of length (flank_left + 1 + flank_right).
            # psite_offset_codons shifts the window centre if needed.
            logging.info(f"  [{exp}] Building AA windows ({n_sites} stall sites) ...")
            win = windows_aa(exp_stalls, cds_range, sequence,
                        flank_left=args.flank_left, flank_right=args.flank_right, psite_offset_codons=args.psite_offset)

            # count_matrix tallies how many times each amino acid appears at
            # each position across all windows.
            # Shape: (n_amino_acids, window_width)
            logging.info(f"  [{exp}] {len(win)} windows built; computing count matrix ...")
            counts = count_matrix(win, AA_ORDER, flank_left=args.flank_left, flank_right=args.flank_right)

            # background_aa_freq computes the overall amino acid composition of
            # the CDS sequences for the expressed transcripts. This is the null
            # model: what frequency would we see by chance?
            # bg = Series of frequencies; bg_counts = raw counts.
            logging.info(f"  [{exp}] Computing background AA frequencies ...")
            bg, bg_counts = background_aa_freq(exp_stalls.keys(), cds_range, sequence, AA_ORDER)

            # pwm_position_weighted_log2 computes log2(observed / background)
            # at each position, normalised and smoothed with a pseudocount.
            # W is a DataFrame: rows = amino acids, columns = window positions.
            # Positive values = enrichment; negative = depletion at stall sites.
            logging.info(f"  [{exp}] Computing PWM ...")
            W = pwm_position_weighted_log2(counts, bg, pseudocount=args.pseudocount)
            logging.info(f"  [{exp}] Done")
            return W, counts, bg, bg_counts

        # Run the above for every experiment and store results keyed by exp name.
        W_by_exp = {}          # PWM log2-enrichment matrices
        counts_by_exp = {}     # Raw amino acid count matrices at stall windows
        bg_by_exp = {}         # Background amino acid frequency vectors
        bg_counts_by_exp = {}  # Raw background amino acid counts
        n_exps = len(stalls)
        for i, exp in enumerate(stalls.keys(), 1):
            logging.info(f"Processing experiment {i}/{n_exps}: {exp}")
            W, counts, bg, bg_counts = compute_W_for_exp(exp)
            W_by_exp[exp] = W
            counts_by_exp[exp] = counts
            bg_by_exp[exp] = bg
            bg_counts_by_exp[exp] = bg_counts

        # -------------------------------------------------------------------------
        # Save motif CSVs
        # -------------------------------------------------------------------------
        # For each experiment, write four CSV files to --out-csv:
        #   <exp>_pwm_log2_enrichment.csv  — the PWM log2(obs/bg) matrix
        #   <exp>_counts.csv               — raw AA counts at stall windows
        #   <exp>_background.csv           — background AA frequencies
        #   <exp>_background_counts.csv    — raw background AA counts
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

    # -------------------------------------------------------------------------
    # Optional: amino acid enrichment analysis at E/P/A sites
    # -------------------------------------------------------------------------
    if args.enrichment:
        print(f"\n{'='*60}")
        print(f"EPA EXTRACTION")
        print(f"{'='*60}")

        # Extract E/P/A amino acid counts per replicate
        replicate_counts = {}  # {rep: {"E": Series, "P": Series, "A": Series}}
        for exp in stalls:
            exp_stalls = {tx: [d["index"] for d in site_list] for tx, site_list in stalls[exp].items()}
            n_sites = sum(len(v) for v in exp_stalls.values())
            if n_sites == 0:
                logging.info(f"  [{exp}] No stall sites — skipping")
                continue
            logging.info(f"  [{exp}] Extracting E/P/A amino acids from {n_sites} stall sites ...")
            _, counts_E, counts_P, counts_A = epa_triplet_counts(
                exp_stalls, cds_range, sequence, psite_offset_codons=0
            )
            replicate_counts[exp] = {"E": counts_E, "P": counts_P, "A": counts_A}
            print(f"  [{exp}] {int(counts_E.sum())} valid stall sites with E/P/A extracted")

        print(f"{'='*60}\n")

        # Background AA frequencies from union of all filtered transcripts
        print(f"\n{'='*60}")
        print(f"BACKGROUND AA FREQUENCIES")
        print(f"{'='*60}")
        logging.info(f"Computing background AA frequencies from {len(filt_tx_union)} transcripts ...")
        bg_freq, bg_raw_counts = background_aa_freq(filt_tx_union, cds_range, sequence, AA_ORDER)
        print(f"  Computed from {len(filt_tx_union)} transcripts ({int(bg_raw_counts.sum())} total codons)")
        print(f"{'='*60}\n")

        # Build replicate-to-condition and replicate-to-timepoint mappings
        # Group names expected: "control_day_0", "BWM_day_5", etc.
        rep_to_condition = {}
        rep_to_timepoint = {}
        for rep, grp in rep_to_group.items():
            parts = grp.split("_", 1)
            rep_to_condition[rep] = parts[0]       # "control" or "BWM"
            rep_to_timepoint[rep] = parts[1] if len(parts) > 1 else grp  # "day_0", etc.

        # =====================================================================
        # Analysis 1: Within-condition enrichment (Binomial test)
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"ANALYSIS 1: WITHIN-CONDITION ENRICHMENT (Binomial Test)")
        print(f"{'='*60}")
        logging.info("Running within-condition enrichment analysis ...")
        df_within = within_condition_enrichment(replicate_counts, bg_freq, rep_to_condition)
        n_sig = (df_within["p_adj"] < 0.05).sum() if not df_within.empty else 0
        n_tests = len(df_within)
        print(f"  Tests performed: {n_tests}")
        print(f"  Significant (p_adj < 0.05): {n_sig}")
        if n_sig > 0:
            sig = df_within[df_within["p_adj"] < 0.05][["condition", "site", "amino_acid", "log2_enrichment", "p_adj"]]
            print(f"\n  Significant results:")
            print(sig.to_string(index=False))
        print(f"{'='*60}\n")

        # =====================================================================
        # Analysis 2: Between-condition overall (Wilcoxon rank-sum)
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"ANALYSIS 2: BETWEEN-CONDITION WILCOXON (n=6 vs n=6)")
        print(f"{'='*60}")
        logging.info("Running between-condition Wilcoxon rank-sum ...")
        df_wilcox = between_condition_wilcoxon(replicate_counts, rep_to_condition)
        n_sig = (df_wilcox["p_adj"] < 0.05).sum() if not df_wilcox.empty else 0
        n_tests = len(df_wilcox)
        print(f"  Tests performed: {n_tests}")
        print(f"  Significant (p_adj < 0.05): {n_sig}")
        if n_sig > 0:
            sig = df_wilcox[df_wilcox["p_adj"] < 0.05][["site", "amino_acid", "log2_FC", "p_adj"]]
            print(f"\n  Significant results:")
            print(sig.to_string(index=False))
        print(f"{'='*60}\n")

        # =====================================================================
        # Analysis 3: Per-timepoint between-condition (Fisher's exact)
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"ANALYSIS 3: PER-TIMEPOINT FISHER'S EXACT TEST")
        print(f"  NOTE: Pooling 2 reps is pseudoreplication — interpret cautiously")
        print(f"{'='*60}")
        logging.info("Running per-timepoint Fisher's exact tests ...")
        df_fisher = per_timepoint_fisher(replicate_counts, rep_to_condition, rep_to_timepoint)
        for tp in sorted(df_fisher["timepoint"].unique()) if not df_fisher.empty else []:
            tp_df = df_fisher[df_fisher["timepoint"] == tp]
            n_sig_tp = (tp_df["p_adj"] < 0.05).sum()
            print(f"  [{tp}] {len(tp_df)} tests, {n_sig_tp} significant (p_adj < 0.05)")
            if n_sig_tp > 0:
                sig = tp_df[tp_df["p_adj"] < 0.05][["timepoint", "site", "amino_acid", "odds_ratio", "p_adj"]]
                print(sig.to_string(index=False))
        print(f"{'='*60}\n")

        # =====================================================================
        # Save per-replicate AA frequencies (intermediate output)
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"WRITING RESULTS")
        print(f"{'='*60}")
        os.makedirs(args.out_enrichment, exist_ok=True)

        # Per-replicate frequencies
        freq_rows = []
        for rep, site_counts in replicate_counts.items():
            grp = rep_to_group.get(rep, "")
            cond = rep_to_condition.get(rep, "")
            tp = rep_to_timepoint.get(rep, "")
            for site_name in ("E", "P", "A"):
                counts = site_counts[site_name]
                total = counts.sum()
                for aa in counts.index:
                    freq_rows.append({
                        "replicate": rep,
                        "group": grp,
                        "condition": cond,
                        "timepoint": tp,
                        "site": site_name,
                        "amino_acid": aa,
                        "stall_count": int(counts[aa]),
                        "total_stall_sites": int(total),
                        "stall_freq": counts[aa] / total if total > 0 else 0.0,
                    })
        df_freqs = pd.DataFrame(freq_rows)

        freq_path = os.path.join(args.out_enrichment, "replicate_aa_frequencies.csv")
        df_freqs.to_csv(freq_path, index=False)
        print(f"  Saved: {freq_path}")

        within_path = os.path.join(args.out_enrichment, "within_condition_enrichment.csv")
        df_within.to_csv(within_path, index=False)
        print(f"  Saved: {within_path}")

        wilcox_path = os.path.join(args.out_enrichment, "between_condition_wilcoxon.csv")
        df_wilcox.to_csv(wilcox_path, index=False)
        print(f"  Saved: {wilcox_path}")

        fisher_path = os.path.join(args.out_enrichment, "per_timepoint_fisher.csv")
        df_fisher.to_csv(fisher_path, index=False)
        print(f"  Saved: {fisher_path}")

        print(f"{'='*60}\n")
        logging.info(f"All enrichment results saved to {args.out_enrichment}")

if __name__ == "__main__":
    main()
