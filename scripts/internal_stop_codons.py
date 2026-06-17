#!/usr/bin/env python3
"""
internal_stop_codons.py

Diagnostic: find transcripts whose CDS carries an in-frame stop codon *inside*
the coding body — i.e. a stop codon that survives after the terminal stop is
trimmed off.

Every CDS ends in a stop codon, which is expected. This script trims the CDS
body with ``--trim-start`` / ``--trim-stop`` (the terminal stop is the last
codon, so the default ``--trim-stop 1`` removes it) and then reports every
in-frame stop codon (TAA / TAG / TGA) that remains. Those internal stops are
candidates for translational recoding — TGA -> selenocysteine, TAG ->
pyrrolysine — i.e. codons that are usually stops but are in rare cases read
through as amino acids. (They can also flag annotation problems.)

For each internal stop the script also pulls the ribosome P-site read count at
that codon from the coverage pickle, per replicate, as evidence that the stop
is actually occupied / translated through rather than purely terminating. The
coverage is indexed exactly as in ``global_codon_occ.py`` (one-to-one nt mapping
to the CDS sequence).

Outputs (both written to ``--out-dir``):
  * ``internal_stop_codons_long.csv``          — one row per internal-stop occurrence
  * ``internal_stop_codons_by_transcript.csv`` — one row per transcript (summary)

This is a read-only diagnostic; it does not feed the occupancy / stall-site
pipelines.
"""

import argparse
import gzip
import logging
import os
import pickle
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import ribopy
from ribopy import Ribo

from ribostall.sequence import get_cds_range_lookup, get_sequence
from ribostall.amino_acids import STOP_CODONS
from ribostall.global_occupancy import iter_trimmed_codons, parse_groups


# -------------------------------------------------------------------------
# Logging (console + dated log file; format shared across the repo)
# -------------------------------------------------------------------------
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(
    _LOG_DIR,
    f"{datetime.now().strftime('%Y%m%d')}_{os.path.splitext(os.path.basename(__file__))[0]}_log.txt",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE),
        logging.StreamHandler(),
    ],
)

# Recoding identity of each stop codon when it is read through as sense.
# TAA has no documented recoding; it is still reported, just left unannotated.
STOP_RECODING = {
    "TGA": "Sec (selenocysteine)",
    "TAG": "Pyl (pyrrolysine)",
    "TAA": "",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnostic: report transcripts whose CDS body contains in-frame "
                    "stop codons after the terminal stop is trimmed off."
    )
    p.add_argument("--ribo", required=True, help="Path to .ribo file")
    p.add_argument("--pickle", required=True,
                   help="Path to gzipped pickle with coverage dict: {rep: {tx: np.ndarray (CDS coverage)}}")
    p.add_argument("--reference", required=True,
                   help="Path to reference fasta used by get_sequence()")
    p.add_argument("--out-dir", default="results/diagnostics/internal_stops",
                   help="Output directory (default: results/diagnostics/internal_stops)")
    p.add_argument("--trim-start", type=int, default=0,
                   help="Exclude the first N codons of the CDS before scanning (default: 0)")
    p.add_argument("--trim-stop", type=int, default=1,
                   help="Exclude the last N codons of the CDS before scanning. The terminal "
                        "stop codon is the last codon, so the default of 1 trims it off and any "
                        "in-frame stop remaining in the body is reported as internal. Pass 0 to "
                        "keep the terminal stop (then every transcript is reported).")
    p.add_argument("--use-human-alias", action="store_true",
                   help="Use ribopy.api.alias.apris_human_alias when opening the Ribo file")
    p.add_argument("--groups",
                   help="Semicolon-separated group:rep1,rep2 definitions, "
                        "e.g. 'groupA:rep1,rep2;groupB:rep3,rep4' "
                        "(used to filter coverage_dict to declared replicates only)")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load inputs (Ribo object + coverage pickle)
    # ------------------------------------------------------------------
    logging.info(f"Loading ribo object from {args.ribo} ...")
    if args.use_human_alias:
        ribo_object = Ribo(args.ribo, alias=ribopy.api.alias.apris_human_alias)
    else:
        ribo_object = Ribo(args.ribo)
    logging.info("Ribo object loaded")

    logging.info(f"Loading coverage from {args.pickle} ...")
    # Nested dict: coverage_dict[replicate][transcript] = np.array of per-nt CDS reads
    # (P-site offset coverage, same indexing as the CDS sequence string).
    # Example:
    #   coverage_dict["bwm_day_0_rep1"]["ZK1098.10"] = np.array([0, 0, 3, 5, 2, 1, ...])
    with gzip.open(args.pickle, "rb") as f:
        coverage_dict = pickle.load(f)
    logging.info(f"Coverage loaded: {len(coverage_dict)} replicates")

    if args.groups:
        # Keep only the replicates declared in --groups (mirrors global_codon_occ.py).
        declared_reps = {r for reps in parse_groups(args.groups).values() for r in reps}
        coverage_dict = {k: v for k, v in coverage_dict.items() if k in declared_reps}
        logging.info(f"Filtered to {len(coverage_dict)} replicates matching --groups")

    replicates = sorted(coverage_dict.keys())

    # Get CDS ranges and sequences (keyed identically to the coverage pickle).
    cds_range = get_cds_range_lookup(ribo_object)
    sequence = get_sequence(ribo_object, args.reference, alias=bool(args.use_human_alias))

    stop_codons = set(STOP_CODONS)

    # ------------------------------------------------------------------
    # 1. Scan every CDS for in-frame internal stop codons (sequence only)
    # ------------------------------------------------------------------
    # Key design choice: "internal" is defined purely by the trim window.
    # iter_trimmed_codons walks the CDS in-frame after dropping the first
    # --trim-start and last --trim-stop codons; with the default --trim-stop 1
    # the terminal stop is excluded, so any stop codon it still yields sits in
    # the coding body. This keeps "trim off the stop" and "find the leftover
    # stops" as one consistent operation rather than special-casing the last
    # codon.
    #
    # internal_stops[tx] = list of (codon_index, nt_index, stop_codon)
    logging.info("Scanning CDS sequences for in-frame internal stop codons ...")
    internal_stops = OrderedDict()
    n_scanned = 0
    for tx in cds_range:
        if tx not in sequence:
            continue
        start, stop = cds_range[tx]
        cds_seq = sequence[tx][start:stop]
        n_scanned += 1

        hits = []
        for codon, nt_idx in iter_trimmed_codons(cds_seq, args.trim_start, args.trim_stop):
            if codon.upper() in stop_codons:
                hits.append((nt_idx // 3, nt_idx, codon.upper()))
        if hits:
            internal_stops[tx] = hits

    n_with = len(internal_stops)
    n_occ = sum(len(v) for v in internal_stops.values())
    logging.info(
        f"Scanned {n_scanned} transcripts; {n_with} carry >=1 internal stop "
        f"({n_occ} occurrences total)"
    )

    # ------------------------------------------------------------------
    # 2. Pull P-site read counts at each internal stop, per replicate
    # ------------------------------------------------------------------
    # For a ribosome whose P-site sits on the stop codon at CDS nt index i, the
    # coverage window cov[i:i+3] holds its read count (same model as the
    # occupancy pipeline). Non-zero reads here are evidence the stop codon is
    # occupied / translated through rather than purely terminating.
    def reads_at(cov, nt_idx):
        # Guard a coverage array that is shorter than the CDS (length mismatch)
        # or a missing transcript by returning zero reads.
        if cov is None or nt_idx + 3 > len(cov):
            return 0.0
        return float(cov[nt_idx:nt_idx + 3].sum())

    logging.info("Pulling P-site read counts at internal stops from coverage ...")

    # Will look like (internal_stop_codons_long.csv):
    #   transcript  cds_len_nt  n_codons  codon_index  nt_index  frac_position  codons_to_end  stop_codon  recoding               bwm_day_0_rep1_reads  total_reads
    #   ZK1098.10   1428        476       142          426       0.2983         333            TGA         Sec (selenocysteine)   18.0                  57.0
    #   F45E4.2     2106        702       655          1965      0.9330         46             TGA         Sec (selenocysteine)   4.0                   11.0
    long_rows = []
    summary_rows = []
    n_tx_in_cov = 0
    for tx, hits in internal_stops.items():
        # Will pull the coverage for each replicate and check if the transcript is covered.
        cov_by_rep = {rep: coverage_dict[rep].get(tx) for rep in replicates}
        if any(cov_by_rep[rep] is not None for rep in replicates):
            n_tx_in_cov += 1

        start, stop = cds_range[tx]
        cds_seq = sequence[tx][start:stop]
        n_codons_full = len(cds_seq) // 3

        per_tx_rep_reads = defaultdict(float)
        for codon_index, nt_idx, stop_codon in hits:
            row = OrderedDict()
            row["transcript"] = tx
            row["cds_len_nt"] = len(cds_seq)
            row["n_codons"] = n_codons_full
            row["codon_index"] = codon_index
            row["nt_index"] = nt_idx
            row["frac_position"] = round(codon_index / n_codons_full, 4) if n_codons_full else 0.0
            row["codons_to_end"] = n_codons_full - 1 - codon_index
            row["stop_codon"] = stop_codon
            row["recoding"] = STOP_RECODING.get(stop_codon, "")
            total = 0.0
            for rep in replicates:
                r = reads_at(cov_by_rep[rep], nt_idx)
                row[f"{rep}_reads"] = r
                per_tx_rep_reads[rep] += r
                total += r
            row["total_reads"] = total
            long_rows.append(row)

        # One summary row per transcript.
        srow = OrderedDict()
        srow["transcript"] = tx
        srow["cds_len_nt"] = len(cds_seq)
        srow["n_codons"] = n_codons_full
        srow["n_internal_stops"] = len(hits)
        srow["internal_stop_codon_indices"] = ";".join(str(h[0]) for h in hits)
        srow["internal_stop_codons"] = ";".join(h[2] for h in hits)
        s_total = 0.0
        for rep in replicates:
            srow[f"{rep}_reads"] = per_tx_rep_reads[rep]
            s_total += per_tx_rep_reads[rep]
        srow["total_reads"] = s_total
        summary_rows.append(srow)

    logging.info(
        f"{n_tx_in_cov}/{n_with} internal-stop transcripts had coverage in >=1 replicate"
    )

    # ------------------------------------------------------------------
    # 3. Sort and write outputs
    # ------------------------------------------------------------------
    # Sort the most-covered internal stops first — those are the strongest
    # read-through / recoding candidates and the first thing to inspect.
    df_long = pd.DataFrame(long_rows)
    df_summary = pd.DataFrame(summary_rows)
    if not df_long.empty:
        df_long = df_long.sort_values(
            ["total_reads", "transcript", "codon_index"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    if not df_summary.empty:
        df_summary = df_summary.sort_values(
            ["total_reads", "transcript"], ascending=[False, True]
        ).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "internal_stop_codons_long.csv"
    summary_path = out_dir / "internal_stop_codons_by_transcript.csv"
    df_long.to_csv(long_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    logging.info(f"Saved {long_path}")
    logging.info(f"Saved {summary_path}")

    # ------------------------------------------------------------------
    # Scan summary (console)
    # ------------------------------------------------------------------
    by_codon = defaultdict(int)
    for hits in internal_stops.values():
        for _, _, sc in hits:
            by_codon[sc] += 1

    print(f"\n{'='*60}")
    print("INTERNAL STOP CODON SCAN")
    print(f"{'='*60}")
    print(f"Transcripts scanned:                 {n_scanned}")
    print(f"Transcripts with internal stop(s):   {n_with}")
    print(f"Internal stop occurrences:           {n_occ}")
    print(f"  with coverage in >=1 replicate:    {n_tx_in_cov} transcripts")
    print(f"Trim window: first {args.trim_start} / last {args.trim_stop} codons removed before scanning")
    print("By stop codon:")
    for sc in sorted(STOP_RECODING):
        tag = STOP_RECODING[sc]
        suffix = f"  ({tag})" if tag else ""
        print(f"  {sc}: {by_codon.get(sc, 0)}{suffix}")
    print(f"{'='*60}\n")

    logging.info("Done.")


if __name__ == "__main__":
    main()
