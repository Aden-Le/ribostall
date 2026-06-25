import logging
import argparse
import gzip
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from ribopy import Ribo

from ribostall.stall_sites import (
    filter_tx,
    codonize_counts_cds,
    call_stalls,
    consensus_stalls_across_reps,
    consensus_to_long_df,
)
from ribostall.amino_acids import (
    AA_ORDER,
    SENSE_CODONS,
    STOP_CODONS,
    annotate_stalls_epa,
    background_aa_freq,
    background_codon_freq,
)
from ribostall.sequence import get_sequence, get_cds_range_lookup
from ribostall.enrichment import plot_coverage_density

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Detect ribosome stall sites with cross-replicate consensus (INTERSECTION variant: "
                    "every group is restricted to the transcripts that pass filtering in ALL groups, so "
                    "all conditions share one transcript universe) and emit stats-ready E/P/A CSVs"
    )
    parser.add_argument("--pickle", required=True, help="Path to coverage pickle.gz file")
    parser.add_argument("--ribo", required=True, help="Path to ribo file")
    parser.add_argument("--reference", required=True,
                        help="Reference FASTA file used to look up CDS sequences for E/P/A annotation")
    parser.add_argument("--groups", required=True,
                        help="Experimental groups, e.g. 'groupA:rep1,rep2;groupB:rep3,rep4'")
    parser.add_argument("--tx_threshold", type=float, default=1.0,
                        help="Minimum reads/nt (in CDS) for filtering transcripts")
    parser.add_argument("--tx_min_reps_per_group", required=True,
                        help="Per-group minimum replicates passing --tx_threshold for transcript "
                             "filtering; must name every declared group, e.g. 'control:2;treatment:1'.")
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
    parser.add_argument("--stall_min_reps_per_group", required=True,
                        help="Per-group minimum supporting replicates; must name every "
                             "declared group, e.g. 'control:2;treatment:1'.")
    parser.add_argument("--tol", type=int, default=0,
                        help="Tolerance window for matching sites across reps (same units as indices)")
    parser.add_argument("--min_sep", type=int, default=7,
                        help="Minimum separation between consensus sites; prefer downstream when closer than this")
    parser.add_argument("--psite-offset", type=int, default=0,
                        help="Codon offset applied to each stall index before deriving E/P/A sites")
    parser.add_argument("--basis", choices=("P", "A"), default="P",
                        help="Register for E/P/A offsets (P: E=-1,P=0,A=+1; A: E=-2,P=-1,A=0)")
    parser.add_argument("--drop-stop-codons", choices=("True", "False"), default="True",
                        help="Drop stall windows whose E/P/A site hits a stop codon "
                             "(TAA/TAG/TGA) from the output CSVs. Default: True; pass "
                             "'--drop-stop-codons False' to keep them.")
    parser.add_argument("--out-dir", default="results/stall_sites_consensus_intersection/raw",
                        help="Output directory for stall-site CSVs")

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
    rep_to_group = {rep: grp for grp, reps in groups.items() for rep in reps}
    # --- end logging ---

    # -------------------------------------------------------------------------
    # Per-group min_support: every declared group must be named explicitly
    # (there is no global fallback).
    # -------------------------------------------------------------------------
    def parse_min_support(arg):
        mapping = {}
        for block in arg.split(";"):
            name, val = block.split(":")
            mapping[name.strip()] = int(val)
        return mapping

    min_support_by_group = parse_min_support(args.stall_min_reps_per_group)
    unknown = set(min_support_by_group) - set(groups)
    if unknown:
        raise ValueError(
            f"--stall_min_reps_per_group names unknown group(s) {sorted(unknown)}; "
            f"declared groups are {sorted(groups)}"
        )
    missing = set(groups) - set(min_support_by_group)
    if missing:
        raise ValueError(
            f"--stall_min_reps_per_group must name every declared group; "
            f"missing {sorted(missing)} (declared groups are {sorted(groups)})"
        )
    # Warn if any group's support exceeds its replicate count (which would make
    # every site fail the support gate → empty consensus for that group).
    for group, reps in groups.items():
        if min_support_by_group[group] > len(reps):
            logging.warning(
                f"min_support={min_support_by_group[group]} for group '{group}' exceeds its "
                f"{len(reps)} replicate(s); no site can reach that support and the group's "
                f"consensus will be empty."
            )
    logging.info(f"Per-group min_support: {min_support_by_group}")

    # -------------------------------------------------------------------------
    # Per-group tx_min_reps: like --stall_min_reps_per_group, every declared
    # group must be named explicitly (there is no global fallback).
    # -------------------------------------------------------------------------
    tx_min_reps_by_group = parse_min_support(args.tx_min_reps_per_group)
    unknown = set(tx_min_reps_by_group) - set(groups)
    if unknown:
        raise ValueError(
            f"--tx_min_reps_per_group names unknown group(s) {sorted(unknown)}; "
            f"declared groups are {sorted(groups)}"
        )
    missing = set(groups) - set(tx_min_reps_by_group)
    if missing:
        raise ValueError(
            f"--tx_min_reps_per_group must name every declared group; "
            f"missing {sorted(missing)} (declared groups are {sorted(groups)})"
        )
    # Warn if any group's tx_min_reps exceeds its replicate count (which would
    # make every transcript fail the filter → empty tx set for that group).
    for group, reps in groups.items():
        if tx_min_reps_by_group[group] > len(reps):
            logging.warning(
                f"tx_min_reps={tx_min_reps_by_group[group]} for group '{group}' exceeds its "
                f"{len(reps)} replicate(s); no transcript can reach that support and the "
                f"group's filtered transcript set will be empty."
            )
    logging.info(f"Per-group tx_min_reps: {tx_min_reps_by_group}")

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
    # Load the ribo object + reference sequences (needed for E/P/A annotation)
    # -------------------------------------------------------------------------
    # --- logging only ---
    logging.info(f"Loading ribo object from {args.ribo} ...")
    # --- end logging ---
    ribo_object = Ribo(args.ribo, alias=None)
    # --- logging only ---
    logging.info("Ribo object loaded")
    print(f"\n{'='*60}\nLOADING SEQUENCES\n{'='*60}")
    logging.info("Looking up CDS ranges ...")
    # --- end logging ---
    cds_range = get_cds_range_lookup(ribo_object)
    # --- logging only ---
    logging.info(f"CDS ranges loaded for {len(cds_range)} transcripts")
    print(f"  CDS ranges: {len(cds_range)} transcripts")
    logging.info(f"Loading sequences from {args.reference} ...")
    # --- end logging ---
    sequence = get_sequence(ribo_object, args.reference, alias=False)
    # --- logging only ---
    logging.info(f"Sequences loaded for {len(sequence)} transcripts")
    print(f"  Sequences: {len(sequence)} transcripts")
    print(f"{'='*60}\n")
    # --- end logging ---

    # -------------------------------------------------------------------------
    # Sanity check: warn if any declared replicate is absent from the coverage
    # -------------------------------------------------------------------------
    missing = [r for rs in groups.values() for r in rs if r not in cov]
    if missing:
        print("Warning: the following replicates are missing from coverage:", ", ".join(missing))

    # -------------------------------------------------------------------------
    # Transcript filtering (per-group, then cross-group intersection)
    # -------------------------------------------------------------------------
    # --- logging only ---
    n_before = len(next(iter(cov.values())))
    print(f"\n{'='*60}")
    print(f"TRANSCRIPT FILTERING (per-group, then cross-group intersection)")
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

    # Coverage density plot: KDE of per-transcript body coverage per replicate
    # (mirrors stall_sites_non_consensus.py). The body window matches the
    # one used by filter_tx / call_stalls. Written into --out-dir.
    os.makedirs(args.out_dir, exist_ok=True)
    plot_coverage_density(cov, groups, args.out_dir,
                          trim_start=args.trim_start, trim_stop=args.trim_stop)
    logging.info(f"Saved coverage density plot to {args.out_dir}/coverage_density.png")

    # Per-group transcript universe — first filter each group independently,
    # then (this is the intersection variant) collapse to the common core.
    filt_tx_dict = {
        group: filter_tx(cov, reps, min_reps=tx_min_reps_by_group[group], threshold=args.tx_threshold,
                         trim_start=args.trim_start, trim_stop=args.trim_stop)
        for group, reps in groups.items()
    }
    # --- logging only ---
    for group, txs in filt_tx_dict.items():
        print(f"  Per-group filter [{group}]: {len(txs)} transcripts  (lost {n_before - len(txs)})")
    # --- end logging ---

    # INTERSECTION: restrict every group to the transcripts that passed filtering
    # in ALL groups, so all conditions are called on one shared transcript
    # universe. This makes the between-condition stall counts apples-to-apples
    # and (because the background is computed from the same transcript set for
    # every group below) makes the per-group backgrounds identical.
    common_txs = (
        sorted(set.intersection(*(set(txs) for txs in filt_tx_dict.values())))
        if filt_tx_dict else []
    )
    if not common_txs:
        logging.warning(
            "Cross-group transcript intersection is empty; no stall sites will be called."
        )
    filt_tx_dict = {group: list(common_txs) for group in filt_tx_dict}
    # --- logging only ---
    print(f"  Cross-group intersection: {len(common_txs)} transcripts shared by all "
          f"{len(groups)} group(s)  (each group restricted to this common set)")
    print(f"{'='*60}\n")
    # --- end logging ---

    # Keep only each group's filtered transcripts in its replicates' coverage
    cov_filt = {}
    for exp, tx_dict in cov.items():
        grp = rep_to_group.get(exp)
        if grp is None:
            continue
        grp_txs = filt_tx_dict[grp]
        cov_filt[exp] = {tx: arr for tx, arr in tx_dict.items() if tx in grp_txs}

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
    logging.info(f"Computing consensus stalls (min_support per group={min_support_by_group}, tol={args.tol}, min_sep={args.min_sep}) ...")
    # --- end logging ---
    consensus = {
        group: consensus_stalls_across_reps(
            stalls,
            reps,
            min_support=min_support_by_group[group],
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
    # Long dataframe of consensus stalls; consensus has one set per group, so
    # treat each group as its own "replicate" for the downstream stats script.
    # -------------------------------------------------------------------------
    logging.info("Converting consensus stalls to long-format dataframe ...")
    df = consensus_to_long_df(consensus)
    df["replicate"] = df["group"]
    logging.info(f"Long-format dataframe: {len(df)} rows")

    # -------------------------------------------------------------------------
    # Annotate each stall with E/P/A codon + AA
    # -------------------------------------------------------------------------
    logging.info("Annotating stalls with E/P/A codons and amino acids ...")
    df_codon, df_aa = annotate_stalls_epa(
        df, cds_range, sequence,
        psite_offset_codons=args.psite_offset,
        basis=args.basis,
        drop_invalid=True,
    )
    logging.info(
        f"Annotation complete: {len(df_codon)} codon rows, {len(df_aa)} AA rows "
        f"(dropped {len(df) - len(df_codon)} rows where E/P/A fell outside the CDS or hit an unknown codon)"
    )

    # -------------------------------------------------------------------------
    # Optionally drop stall windows whose E/P/A hits a stop codon
    # -------------------------------------------------------------------------
    if args.drop_stop_codons == "True":
        n_before_stop = len(df_codon)
        codon_keep = ~df_codon[["E_codon", "P_codon", "A_codon"]].isin(STOP_CODONS).any(axis=1)
        aa_keep = ~df_aa[["E_aa", "P_aa", "A_aa"]].isin(["*"]).any(axis=1)
        df_codon = df_codon.loc[codon_keep].reset_index(drop=True)
        df_aa = df_aa.loc[aa_keep].reset_index(drop=True)
        logging.info(
            f"--drop-stop-codons: dropped {n_before_stop - len(df_codon)} stall windows "
            f"containing a stop codon; {len(df_codon)} codon rows / {len(df_aa)} AA rows remain"
        )

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    codon_path = os.path.join(args.out_dir, "stall_sites_codon.csv")
    aa_path = os.path.join(args.out_dir, "stall_sites_aa.csv")
    df_codon.to_csv(codon_path, index=False)
    df_aa.to_csv(aa_path, index=False)
    logging.info(f"Saved codon-annotated stall sites to {codon_path}")
    logging.info(f"Saved AA-annotated stall sites to {aa_path}")

    # -------------------------------------------------------------------------
    # Per-group background frequencies (codon + AA) for the stats script
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\nBACKGROUND FREQUENCIES (per group)\n{'='*60}")
    for level, helper, alphabet, feature_col in (
        ("codon", background_codon_freq, SENSE_CODONS, "codon"),
        ("aa", background_aa_freq, AA_ORDER, "amino_acid"),
    ):
        rows = []
        for grp, grp_txs in filt_tx_dict.items():
            bg_freq, bg_counts = helper(
                grp_txs, cds_range, sequence, alphabet,
                trim_start=args.trim_start, trim_stop=args.trim_stop,
            )
            print(f"  [{grp}] ({level}) {len(grp_txs)} transcripts, {int(bg_counts.sum())} total {level}s")
            for feat in bg_freq.index:
                rows.append({
                    "group": grp,
                    feature_col: feat,
                    "bg_count": int(bg_counts[feat]),
                    "bg_freq": float(bg_freq[feat]),
                })
        bg_path = os.path.join(args.out_dir, f"per_group_background_{level}.csv")
        pd.DataFrame(rows).to_csv(bg_path, index=False)
        logging.info(f"Saved per-group {level} backgrounds to {bg_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
