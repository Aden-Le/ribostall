import logging
import argparse
import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import os
import pandas as pd
import ribopy
from ribopy import Ribo

# Local modules with the core analysis functions
from functions_folder.functions_stall_sites import filter_tx, codonize_counts_cds, call_stalls, stalls_to_long_df, consensus_stalls_across_reps
from functions_folder.functions_AA import (translate_cds_nt_to_aa, windows_aa, count_matrix, background_aa_freq,
                          pwm_position_weighted_log2, plot_logo, epa_triplet_counts,
                          CODON2AA, AA_ORDER, AA_CLASS, CLASS_COLORS)
from functions_folder.functions import get_sequence, get_cds_range_lookup
from functions_folder.functions_enrichment import within_condition_enrichment, between_condition_wilcoxon, per_timepoint_fisher

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
    parser.add_argument("--reference", help="Reference file path")
    parser.add_argument("--flank-left", type=int, default=10, help="Motif")
    parser.add_argument("--flank-right", type=int, default=6, help="Motif")
    parser.add_argument("--psite-offset", type=int, default=0, help="Motif")
    parser.add_argument("--out-csv", default="stall_sites_results", help="Motif")
    parser.add_argument("--enrichment", action="store_true",
                        help="Run amino acid enrichment analysis at E/P/A sites")
    parser.add_argument("--out-enrichment", default="stall_sites_results",
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
    # Inverses the groups dict to map each replicate to its group, e.g. "rep1" -> "control_day_0".
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}

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

    # NOTE: The coverage pickle may contain rep1 experiments that failed QC.
    # They are excluded automatically because --groups only lists rep2 & rep3.
    # Any experiment not in --groups is never used by the analysis.

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
    # These transcripts represent transcripts that have good coverage in both replicates per group

    # how many transcripts are in the coverage before filtering?
    n_before = len(next(iter(cov.values())))

    print(f"\n{'='*60}")
    print(f"TRANSCRIPT FILTERING (per-group, no intersection)")
    print(f"{'='*60}")
    print(f"Transcripts before filtering: {n_before}")

    # Per-group filter: for each group, call filter_tx which keeps only
    # transcripts that exceed --tx_threshold reads/nt in at least --tx_min_reps replicates within that group.
    # Result: filt_tx_dict[group] = set of transcript names passing that group.
    filt_tx_dict = {
        group: set(filter_tx(cov, reps, min_reps=args.tx_min_reps, threshold=args.tx_threshold))
        for group, reps in groups.items()
    }
    for group, txs in filt_tx_dict.items():
        print(f"  Per-group filter [{group}]: {len(txs)} transcripts  (lost {n_before - len(txs)})")

    # Rebuild the coverage dict — each experiment uses its GROUP's filtered transcripts.
    # Structure preserved: cov_filt[exp][tx] = array
    cov_filt = {}
    for exp, tx_dict in cov.items():
        grp = rep_to_group.get(exp)
        if grp is None:
            continue
        # Transcripts that pass the filter for this experiment's group
        grp_txs = filt_tx_dict[grp]
        # Only keep those transcripts in the coverage dict for this experiment
        cov_filt[exp] = {tx: arr for tx, arr in tx_dict.items() if tx in grp_txs}

    # -------------------------------------------------------------------------
    # Codonize coverage
    # -------------------------------------------------------------------------
    # The raw coverage arrays are nucleotide-resolution (one value per nt).
    # codonize_counts_cds sums every 3 consecutive nt positions into a single codon-level count, restricted to the CDS region.
    # Looks like { "rep1": {"tx1": array([...]), "tx2": array([...]), ...}, "rep2": {...}, ... }
    # Where each array looks like [5, 3, 0, 2, ...] with one value per codon in the CDS.
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

    # Reproducibility quality metric: consensus stall sites across replicates per group.
    print(f"{'='*60}")
    print(f"REPRODUCIBILITY (consensus across replicates)")
    print(f"{'='*60}")
    for grp, reps in groups.items():
        if len(reps) < 2:
            print(f"  [{grp}]: skipped (only {len(reps)} replicate)")
            continue
        consensus = consensus_stalls_across_reps(stalls, reps, min_support=2, tol=0)
        n_consensus = sum(len(idxs) for idxs in consensus.values())
        n_total_union = len(set(
            (tx, d["index"])
            for r in reps
            for tx, site_list in stalls[r].items()
            for d in site_list
        ))
        pct = (n_consensus / n_total_union * 100) if n_total_union else 0
        print(f"  [{grp}] ({len(reps)} reps): "
              f"{n_consensus} consensus / {n_total_union} union stall sites "
              f"({pct:.1f}% reproducible)")
    print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Write stall sites to CSV Lines
    # -------------------------------------------------------------------------
    logging.info(f"Converting stalls to long-format dataframe ...")
    # Inverses the groups dict to map each replicate to its group, e.g. "rep1" -> "control_day_0".
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}

    # stalls_to_long_df converts the nested stalls dict into a tidy pandas
    # DataFrame where each row is one stall site, with columns for experiment,
    # Will look like:
    #   exp       group       transcript  codon_index  z_score  reads
    #   rep1      control_day_0  tx1         42           3
    df = stalls_to_long_df(stalls, rep_to_group)
    logging.info(f"Long-format dataframe: {len(df)} rows")

    os.makedirs(args.out_enrichment, exist_ok=True)
    stalls_csv_path = os.path.join(args.out_enrichment, "stall_sites.csv")
    logging.info(f"Writing stall sites to {stalls_csv_path} ...")
    df.to_csv(stalls_csv_path, index=False)
    logging.info(f"Saved stall sites to {stalls_csv_path}")

    # -------------------------------------------------------------------------
    # Load CDS ranges and sequences (Used in --enrichment)
    # -------------------------------------------------------------------------
    cds_range = None
    sequence = None
    if args.enrichment:
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
    # Optional: amino acid enrichment analysis at E/P/A sites
    # -------------------------------------------------------------------------
    if args.enrichment:
        print(f"\n{'='*60}")
        print(f"EPA EXTRACTION")
        print(f"{'='*60}")

        # Extract E/P/A amino acid counts per replicate
        replicate_counts = {}  # {rep: {"E": Series, "P": Series, "A": Series}}
        for exp in stalls:
            # exp_stalls is a dict: {tx: [index1, index2, ...]} for all stall sites in this experiment
            exp_stalls = {tx: [d["index"] for d in site_list] for tx, site_list in stalls[exp].items()}
            n_sites = sum(len(v) for v in exp_stalls.values())
            if n_sites == 0:
                logging.info(f"  [{exp}] No stall sites — skipping")
                continue
            logging.info(f"  [{exp}] Extracting E/P/A amino acids from {n_sites} stall sites ...")

            # Returns counts of amino acids at E/P/A sites across all stall sites in this experiment.
            _, counts_E, counts_P, counts_A = epa_triplet_counts(
                exp_stalls, cds_range, sequence, psite_offset_codons=0
            )
            # Stores the counts in the replicate_counts dict for later enrichment analysis.
            replicate_counts[exp] = {"E": counts_E, "P": counts_P, "A": counts_A}
            print(f"  [{exp}] {int(counts_E.sum())} valid stall sites with E/P/A extracted")

        print(f"{'='*60}\n")

        # Background AA frequencies per group (each group has its own filtered transcripts)
        print(f"\n{'='*60}")
        print(f"BACKGROUND AA FREQUENCIES (per group)")
        print(f"{'='*60}")
        # Frequency is the proportion of each amino acid across all codons in the filtered transcripts for that group. (Normalized)
        bg_freq_per_group = {}
        # Counts is the total number of codons across all transcripts in that group (Counts)
        bg_counts_per_group = {}
        for grp, grp_txs in filt_tx_dict.items():
            grp_bg_freq, grp_bg_counts = background_aa_freq(grp_txs, cds_range, sequence, AA_ORDER)
            bg_freq_per_group[grp] = grp_bg_freq
            bg_counts_per_group[grp] = grp_bg_counts
            print(f"  [{grp}] {len(grp_txs)} transcripts ({int(grp_bg_counts.sum())} total codons)")
        print(f"{'='*60}\n")

        # Build replicate-to-condition and replicate-to-timepoint mappings
        # Group names expected: "control_day_0", "BWM_day_5", etc.
        # Aligns the experiment to their condition and timepoint for later stratified analyses.
        rep_to_condition = {}
        rep_to_timepoint = {}
        for rep, grp in rep_to_group.items():
            parts = grp.split("_", 1)
            rep_to_condition[rep] = parts[0]       # "control" or "BWM"
            rep_to_timepoint[rep] = parts[1] if len(parts) > 1 else grp  # "day_0", etc.

        # =====================================================================
        # Analysis 1: Within-condition enrichment (Binomial test)
        # =====================================================================
        # replicate_counts: {exp: {"E": Series, "P": Series, "A": Series}, exp2: {...}, ...}
        # bg_freq_per_group: {group: Series of AA frequencies for that group's filtered transcripts}
        # rep_to_condition: {rep: condition} mapping for each experiment
        # rep_to_group: {rep: group} mapping for each experiment
        print(f"\n{'='*60}")
        print(f"ANALYSIS 1: WITHIN-CONDITION ENRICHMENT (Binomial Test)")
        print(f"{'='*60}")

        logging.info("Running within-condition enrichment analysis ...")
        
        # Binomial Test
        df_within = within_condition_enrichment(replicate_counts, bg_freq_per_group, rep_to_condition, rep_to_group)

        # Print summary of results: number of tests performed and how many are significant after multiple testing correction (p_adj < 0.05).
        n_sig = (df_within["p_adj"] < 0.05).sum() if not df_within.empty else 0
        n_tests = len(df_within)
        print(f"  Tests performed: {n_tests}")
        print(f"  Significant (p_adj < 0.05): {n_sig}")
        if n_sig > 0:
            sig = df_within[df_within["p_adj"] < 0.05][["condition", "group", "site", "amino_acid", "log2_enrichment", "p_adj"]]
            print(f"\n  Significant results:")
            print(sig.to_string(index=False))
        print(f"{'='*60}\n")

        # =====================================================================
        # Analysis 2: Between-condition overall (Wilcoxon rank-sum)
        # =====================================================================
        # replicate_counts: {exp: {"E": Series, "P": Series, "A": Series}, exp2: {...}, ...}
        # rep_to_condition: {rep: condition} mapping for each experiment
        print(f"\n{'='*60}")
        print(f"ANALYSIS 2: BETWEEN-CONDITION WILCOXON (n=6 vs n=6)")
        print(f"{'='*60}")

        logging.info("Running between-condition Wilcoxon rank-sum ...")
        df_wilcox = between_condition_wilcoxon(replicate_counts, rep_to_condition)

        # Print summary of results: number of tests performed and how many are significant after multiple testing correction (p_adj < 0.05).
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
        # replicate_counts: {exp: {"E": Series, "P": Series, "A": Series}, exp2: {...}, ...}
        # rep_to_condition: {rep: condition} mapping for each experiment
        # rep_to_timepoint: {rep: timepoint} mapping for each experiment
        print(f"\n{'='*60}")
        print(f"ANALYSIS 3: PER-TIMEPOINT FISHER'S EXACT TEST")
        print(f"  NOTE: Pooling 2 reps is pseudoreplication — interpret cautiously")
        print(f"{'='*60}")

        logging.info("Running per-timepoint Fisher's exact tests ...")
        df_fisher = per_timepoint_fisher(replicate_counts, rep_to_condition, rep_to_timepoint)

        # Print summary of results stratified by timepoint: number of tests and significant results per timepoint.
        for tp in sorted(df_fisher["timepoint"].unique()) if not df_fisher.empty else []:
            tp_df = df_fisher[df_fisher["timepoint"] == tp]
            n_sig_tp = (tp_df["p_adj"] < 0.05).sum()
            print(f"  [{tp}] {len(tp_df)} tests, {n_sig_tp} significant (p_adj < 0.05)")
            if n_sig_tp > 0:
                sig = tp_df[tp_df["p_adj"] < 0.05][["timepoint", "site", "amino_acid", "odds_ratio", "p_adj"]]
                print(sig.to_string(index=False))
        print(f"{'='*60}\n")

        # =====================================================================
        # Save per-replicate AA frequencies (intermediate output) | Just a bunch of saving
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

        # Per-replicate E/P/A raw count CSVs (wide: AA × site, one file per rep)
        input_dir = os.path.join(args.out_enrichment, "input_data")
        os.makedirs(input_dir, exist_ok=True)
        for rep, site_counts in replicate_counts.items():
            rep_df = pd.DataFrame({
                site_name: site_counts[site_name]
                for site_name in ("E", "P", "A")
            })
            rep_df.index.name = "amino_acid"
            rep_path = os.path.join(input_dir, f"{rep}_epa_counts.csv")
            rep_df.to_csv(rep_path)
        print(f"  Saved: per-replicate E/P/A count CSVs to {input_dir}")

        # Per-group background AA frequencies
        bg_rows = []
        for grp in bg_freq_per_group:
            grp_bg_freq = bg_freq_per_group[grp]
            grp_bg_counts = bg_counts_per_group[grp]
            for aa in grp_bg_freq.index:
                bg_rows.append({
                    "group": grp,
                    "amino_acid": aa,
                    "bg_count": int(grp_bg_counts[aa]),
                    "bg_freq": grp_bg_freq[aa],
                })
        df_bg_groups = pd.DataFrame(bg_rows)
        bg_group_path = os.path.join(input_dir, "per_group_background_aa.csv")
        df_bg_groups.to_csv(bg_group_path, index=False)
        print(f"  Saved: {bg_group_path}")

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
