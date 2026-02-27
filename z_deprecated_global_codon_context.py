#!/usr/bin/env python3
import argparse
import gzip
import pickle
from collections import defaultdict
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import ribopy
from ribopy import Ribo

# --- your helpers (already in your repo) ---
from functions import get_cds_range_lookup, get_sequence

# --- constants ---
CODON2AA = {
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
    "AAT":"N","AAC":"N","GAT":"D","GAC":"D","TGT":"C","TGC":"C",
    "GAA":"E","GAG":"E","CAA":"Q","CAG":"Q","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
    "CAT":"H","CAC":"H","ATT":"I","ATC":"I","ATA":"I",
    "TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "AAA":"K","AAG":"K","ATG":"M","TTT":"F","TTC":"F",
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","AGT":"S","AGC":"S",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T","TGG":"W",
    "TAT":"Y","TAC":"Y","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TAA":"*","TAG":"*","TGA":"*"
}
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")  # no stop
ALL_CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)

# --- arg parsing ---
def parse_args():
    p = argparse.ArgumentParser(
        description="Coverage-weighted codon/AA context around read-bearing codons (CDS-only coverage)."
    )
    p.add_argument("--ribo", required=True, help="Path to .ribo")
    p.add_argument("--pickle", required=True,
                   help="gzipped pickle: {exp: {transcript: np.ndarray (CDS-only per-nt coverage)}}")
    p.add_argument("--reference", required=True, help="Reference FASTA/2bit used by get_sequence()")
    p.add_argument("--out-prefix", default="context", help="Prefix for outputs (default: context)")


    # trimming & context window
    p.add_argument("--trim-start", type=int, default=0, help="Exclude first N codons of CDS")
    p.add_argument("--trim-stop", type=int, default=0, help="Exclude last N codons of CDS")
    p.add_argument("--flank-left", type=int, default=10, help="Context size upstream of anchor (codons)")
    p.add_argument("--flank-right", type=int, default=10, help="Context size downstream of anchor (codons)")
    p.add_argument("--min-anchor-reads", type=float, default=0.0,
                   help="Ignore anchors with summed reads ≤ threshold")

    # grouping & options
    p.add_argument(
        "--groups",
        type=str,
        required=True,
        help=("Group experiments for aggregation, e.g. "
              "\"kidney:kidney_rep1,kidney_rep2,kidney_rep3;"
              "liver:liver_rep1,liver_rep2,liver_rep3;"
              "lung:lung_rep1,lung_rep2,lung_rep3\"")
    )
    p.add_argument("--collapse-aa", action="store_true", help="Collapse codons to amino acids")
    p.add_argument("--use-human-alias", action="store_true",
                   help="Use ribopy.api.alias.apris_human_alias when opening Ribo")
    p.add_argument("--threads", type=int, default=4, help="Workers for parallel background build")

    return p.parse_args()

def parse_groups(groups_str: str):
    groups = {}
    for grp in groups_str.split(";"):
        grp = grp.strip()
        if not grp:
            continue
        if ":" not in grp:
            raise ValueError(f"Invalid group spec: '{grp}' (missing ':')")
        name, reps_csv = grp.split(":", 1)
        name = name.strip()
        reps = [r.strip() for r in reps_csv.split(",") if r.strip()]
        if not name or not reps:
            raise ValueError(f"Invalid group spec: '{grp}'")
        groups[name] = reps
    if not groups:
        raise ValueError("No valid groups parsed from --groups")
    return groups

# --- helpers ---
def iter_trimmed_codons(seq_nt: str, trim_start: int, trim_stop: int):
    """
    Yield (codon_str, local_cidx, cds_nt_start) over the TRIMMED CDS.
      - local_cidx: 0..(n_trimmed_codons-1) within trimmed window
      - cds_nt_start: nt index within the CDS (for slicing CDS-only coverage)
    """
    n = len(seq_nt)
    start_nt = trim_start * 3
    end_nt = n - trim_stop * 3
    if start_nt >= end_nt:
        return
    # only full codons
    last_full = end_nt - ((end_nt - start_nt) % 3)
    local = 0
    for i in range(start_nt, last_full, 3):
        codon = seq_nt[i:i+3].upper().replace("U", "T")
        if len(codon) == 3:
            yield codon, local, i
            local += 1



# Map codon->AA; define CODON2AA and AA_ORDER earlier in your file.

def _labels_from_cds_fast(seq_nt, trim_start, trim_stop, codon_to_index, aa_mode=False, aa_to_index=None):
    """
    Convert a CDS nt string into a trimmed array of integer labels per codon.
    Returns np.ndarray[int] (len = n_trimmed_codons), -1 for 'skip'.
    If aa_mode=True, collapse codons to AA before indexing with aa_to_index.
    """
    n = len(seq_nt)
    start_nt = trim_start * 3
    end_nt = n - trim_stop * 3
    if start_nt >= end_nt:
        return np.empty(0, dtype=np.int32)

    last_full = end_nt - ((end_nt - start_nt) % 3)
    n_cod = (last_full - start_nt) // 3
    if n_cod <= 0:
        return np.empty(0, dtype=np.int32)

    out = np.full(n_cod, -1, dtype=np.int32)
    j = 0
    for i in range(start_nt, last_full, 3):
        codon = seq_nt[i:i+3].upper().replace("U", "T")
        if aa_mode:
            aa = CODON2AA.get(codon, "X")
            if aa_to_index is not None and aa in aa_to_index:
                out[j] = aa_to_index[aa]
        else:
            if codon in codon_to_index:
                out[j] = codon_to_index[codon]
        j += 1
    return out

def _count_offsets_for_labels(labels, n_rows, offsets):
    """
    Vectorized per-offset counting. labels = int array (>=0 valid, -1 skip).
    Returns np.ndarray (n_rows x n_offsets).
    """
    N = labels.shape[0]
    mat = np.zeros((n_rows, len(offsets)), dtype=np.float64)
    if N == 0:
        return mat

    for k, off in enumerate(offsets):
        if off < 0:
            a0, a1 = -off, N
            b0, b1 = 0, N + off
        elif off > 0:
            a0, a1 = 0, N - off
            b0, b1 = off, N
        else:
            a0, a1 = 0, N
            b0, b1 = 0, N

        if a1 - a0 <= 0:
            continue

        neigh = labels[b0:b1]
        neigh = neigh[neigh >= 0]
        if neigh.size == 0:
            continue

        counts = np.bincount(neigh, minlength=n_rows).astype(np.float64)
        mat[:, k] += counts
    return mat

def _bg_worker(task):
    """
    Worker is top-level (picklable). Computes background chunk.
    task = (chunk_tx, cds_range, sequence, trim_start, trim_stop, offsets,
            row_labels, aa_mode, codon_to_index, aa_to_index)
    """
    (chunk, cds_range, sequence, trim_start, trim_stop, offsets,
     row_labels, aa_mode, codon_to_index, aa_to_index) = task

    n_rows = len(row_labels)
    acc = np.zeros((n_rows, len(offsets)), dtype=np.float64)

    for tx in chunk:
        if tx not in cds_range or tx not in sequence:
            continue
        s, e = cds_range[tx]
        if e <= s:
            continue
        cds_nt = sequence[tx][s:e]
        if not cds_nt:
            continue

        labels = _labels_from_cds_fast(
            cds_nt, trim_start, trim_stop,
            codon_to_index=codon_to_index,
            aa_mode=aa_mode,
            aa_to_index=aa_to_index
        )
        if labels.size == 0:
            continue
        acc += _count_offsets_for_labels(labels, n_rows, offsets)

    return acc

def build_background_parallel(cds_range, sequence, trim_start, trim_stop,
                              offsets, row_labels, collapse_aa, threads,
                              allowed_codons=None, AA_ORDER=None):
    """
    Parallel background builder with top-level worker.
    - collapse_aa: bool (True => AA mode)
    - allowed_codons: set/list of codons to keep (codon mode)
    - AA_ORDER: list of amino acids (AA mode)
    """
    from math import ceil
    import multiprocessing as mp

    # Build index maps at parent process
    if collapse_aa:
        aa_to_index = {aa: i for i, aa in enumerate(AA_ORDER)}
        codon_to_index = {}  # unused
    else:
        codon_list = sorted(list(allowed_codons))
        codon_to_index = {c: i for i, c in enumerate(codon_list)}
        aa_to_index = None

    txs = list(cds_range.keys())
    if not txs:
        return pd.DataFrame(0.0, index=row_labels, columns=offsets)

    threads = max(1, int(threads))
    chunk_size = max(1, ceil(len(txs) / threads))
    chunks = [txs[i:i+chunk_size] for i in range(0, len(txs), chunk_size)]

    # Build tasks
    tasks = [
        (chunk, cds_range, sequence, trim_start, trim_stop, offsets,
         row_labels, bool(collapse_aa), codon_to_index, aa_to_index)
        for chunk in chunks
    ]

    # macOS uses 'spawn' — safe if everything is top-level
    with mp.Pool(processes=threads) as pool:
        parts = pool.map(_bg_worker, tasks)

    total = np.sum(parts, axis=0) if parts else np.zeros((len(row_labels), len(offsets)), dtype=np.float64)
    bg = pd.DataFrame(total, index=row_labels, columns=offsets)
    return bg



def main():
    args = parse_args()

    # Ribo object
    if args.use_human_alias:
        ribo = Ribo(args.ribo, alias=ribopy.api.alias.apris_human_alias)
    else:
        ribo = Ribo(args.ribo)

    # Load coverage dict
    with gzip.open(args.pickle, "rb") as f:
        coverage_dict = pickle.load(f)  # {exp: {tx: np.ndarray (CDS-only)}}

    groups = parse_groups(args.groups)
    all_exps = set(coverage_dict.keys())
    for gname, reps in groups.items():
        missing = [r for r in reps if r not in all_exps]
        if missing:
            print(f"[warn] group '{gname}': missing experiments not in pickle: {missing}")

    # Reference helpers
    cds_range = get_cds_range_lookup(ribo)
    # If your get_sequence expects an alias object, adapt below accordingly
    sequence = get_sequence(ribo, args.reference, alias=bool(args.use_human_alias))

    # Offsets and label mapping
    offsets = list(range(-args.flank_left, 0)) + [0] + list(range(1, args.flank_right + 1))

    if args.collapse_aa:
        row_labels = AA_ORDER
        def label_of(codon: str):
            aa = CODON2AA.get(codon, "X")
            return aa if aa in AA_ORDER else None
    else:
        row_labels = [c for c in ALL_CODONS if CODON2AA.get(c, "*") != "*"]  # drop stops
        def label_of(codon: str):
            return codon if CODON2AA.get(codon, "*") != "*" else None

    # ---------- build background once (sequence-only) ----------
    # logging.info("Building background")
    # bg = pd.DataFrame(0.0, index=row_labels, columns=offsets)

    # for tx in cds_range:
    #     if tx not in sequence:
    #         continue
    #     start, stop = cds_range[tx]
    #     if stop <= start:
    #         continue
    #     cds_nt = sequence[tx][start:stop]
    #     if not cds_nt:
    #         continue

    #     codons = []
    #     for codon, local_idx, cds_nt_start in iter_trimmed_codons(cds_nt, args.trim_start, args.trim_stop):
    #         codons.append((codon, local_idx, cds_nt_start))
    #     if not codons:
    #         continue

    #     n_trim = len(codons)
    #     for codon, local_idx, _ in codons:
    #         tgt = label_of(codon)
    #         if tgt is None:
    #             continue
    #         for off in offsets:
    #             j = local_idx + off
    #             if 0 <= j < n_trim:
    #                 neigh = label_of(codons[j][0])
    #                 if neigh is not None:
    #                     bg.at[neigh, off] += 1.0

    allowed_codons = set([c for c in row_labels]) if not args.collapse_aa else None

    logging.info("Building background (parallel)")
    bg = build_background_parallel(
        cds_range=cds_range,
        sequence=sequence,
        trim_start=args.trim_start,
        trim_stop=args.trim_stop,
        offsets=offsets,
        row_labels=row_labels,
        collapse_aa=args.collapse_aa,
        threads=args.threads if hasattr(args, "threads") else 4,
        allowed_codons=allowed_codons,
        AA_ORDER=AA_ORDER
    )
    print(bg)
    # # Precompute background probabilities (per-offset)
    # bg_prob = bg.div(bg.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)

    # # ---------- per-group coverage-weighted context ----------
    # out_prefix = Path(args.out_prefix)
    # out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # for gname, reps in groups.items():
    #     print(f"[info] processing group: {gname}")

    #     # Sum coverage for this group only
    #     cov_sum_by_tx: dict[str, np.ndarray] = {}
    #     for exp in reps:
    #         if exp not in coverage_dict:
    #             continue
    #         for tx, cov in coverage_dict[exp].items():
    #             if cov is None:
    #                 continue
    #             if tx not in cov_sum_by_tx:
    #                 cov_sum_by_tx[tx] = np.array(cov, dtype=float, copy=True)
    #             else:
    #                 if len(cov_sum_by_tx[tx]) == len(cov):
    #                     cov_sum_by_tx[tx] += cov  # CDS-only arrays, same length expected

    #     # Context accumulator for this group
    #     context = pd.DataFrame(0.0, index=row_labels, columns=offsets)

    #     for tx, cov in cov_sum_by_tx.items():
    #         if tx not in cds_range or tx not in sequence:
    #             continue
    #         start, stop = cds_range[tx]
    #         if stop <= start:
    #             continue
    #         cds_nt = sequence[tx][start:stop]
    #         if not cds_nt:
    #             continue

    #         # Build trimmed codon list: (codon, local_idx, cds_nt_start)
    #         codons = []
    #         for codon, local_idx, cds_nt_start in iter_trimmed_codons(cds_nt, args.trim_start, args.trim_stop):
    #             codons.append((codon, local_idx, cds_nt_start))
    #         if not codons:
    #             continue

    #         n_trim = len(codons)

    #         # Anchor weights = sum of reads over codon's 3 nts (coverage is CDS-only)
    #         anchor_w = np.zeros(n_trim, dtype=float)
    #         cov_len = len(cov)
    #         for _, local_idx, cds_nt_start in codons:
    #             s, e = cds_nt_start, cds_nt_start + 3
    #             if e <= cov_len:
    #                 anchor_w[local_idx] = float(cov[s:e].sum())

    #         # Accumulate context weighted by anchor reads
    #         for local_idx in range(n_trim):
    #             w = anchor_w[local_idx]
    #             if w <= args.min_anchor_reads:
    #                 continue
    #             for off in offsets:
    #                 j = local_idx + off
    #                 if 0 <= j < n_trim:
    #                     lab = label_of(codons[j][0])
    #                     if lab is not None:
    #                         context.at[lab, off] += w

    #     # Save outputs for this group
    #     context_path = out_prefix.with_suffix(f".{gname}.context.csv")
    #     context.to_csv(context_path)

    #     context_prob = context.div(context.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    #     context_prob_path = out_prefix.with_suffix(f".{gname}.context_prob.csv")
    #     context_prob.to_csv(context_prob_path)

    #     eps = 1e-9
    #     oe = (context_prob + eps) / (bg_prob + eps)
    #     oe_path = out_prefix.with_suffix(f".{gname}.OE.csv")
    #     oe.to_csv(oe_path)

    #     print(f"  wrote: {context_path.name}, {context_prob_path.name}, {oe_path.name}")

    # # Write background once
    # bg_path = out_prefix.with_suffix(".background.csv")
    # bg_prob_path = out_prefix.with_suffix(".background_prob.csv")
    # bg.to_csv(bg_path)
    # bg_prob.to_csv(bg_prob_path)
    # print(f"[info] wrote background: {bg_path.name}, {bg_prob_path.name}")
    # print("Done.")

if __name__ == "__main__":
    main()