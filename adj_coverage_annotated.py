#!/usr/bin/env python3
import argparse
import gzip
import logging
import multiprocessing as mp
import pickle
import sys
import time
from typing import Dict, Tuple, Iterable, List

import numpy as np
import ribopy
from ribopy import Ribo

# --- import your helpers ---
from functions import get_cds_range_lookup, get_offset, apris_human_alias

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)

# =========================
# Worker globals
# =========================
_RIBO = None
_CDS_RANGE: Dict[str, Tuple[int, int]] = {}
_TRANSCRIPTS: List[str] = []

def _safe_window(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Return arr[lo:hi] with zero-padding if the window runs off the array."""
    # lo/hi are signed Python ints
    print(f"[_safe_window] Creating window from {lo} to {hi} on array of length {len(arr)}")
    n = hi - lo
    if n <= 0:
        return np.zeros(0, dtype=arr.dtype)
    left = max(0, -lo)
    right = max(0, hi - int(len(arr)))
    lo_c, hi_c = max(0, lo), min(int(len(arr)), hi)
    core = arr[lo_c:hi_c]
    if left or right:
        return np.concatenate(
            (np.zeros(left, dtype=arr.dtype), core, np.zeros(right, dtype=arr.dtype))
        )
    return core

def _init_worker(ribo_path: str, use_alias: bool,
                 cds_range: Dict[str, Tuple[int, int]],
                 transcripts: List[str]) -> None:
    """Open the .ribo once per worker and stash shared metadata."""
    print(f"[_init_worker] Initializing worker with ribo file: {ribo_path}, use_alias={use_alias}")
    global _RIBO, _CDS_RANGE, _TRANSCRIPTS
    if use_alias:
        _RIBO = Ribo(ribo_path, alias=ribopy.api.alias.apris_human_alias)
        _TRANSCRIPTS = [apris_human_alias(t) for t in _RIBO.transcript_names]
    else:
        _RIBO = Ribo(ribo_path)
        _TRANSCRIPTS = list(transcripts)
    print(f"[_init_worker] Loaded {len(_TRANSCRIPTS)} transcripts")
    # cast CDS ranges to signed ints defensively
    _CDS_RANGE = {t: (int(s), int(e)) for t, (s, e) in cds_range.items()}

# Creates output array for each transcript sized to its CDS window, then iteratively adds coverage for each read length and experiment
# Takes the stop codon - start codon position to get length and make array of that size
def _preallocate_output(transcripts: Iterable[str]) -> Dict[str, np.ndarray]:
    """Pre-allocate a zero array per transcript sized to its CDS window."""
    # Transcript then array of 0s of size length
    print(f"[_preallocate_output] Allocating output arrays for {len(list(transcripts))} transcripts")
    out: Dict[str, np.ndarray] = {}
    for t in transcripts:
        start, stop = _CDS_RANGE[t]
        win = max(0, int(stop) - int(start))
        out[t] = np.zeros(win, dtype=np.int64)
    return out

def _add_length_into_out(exp: str, L: int, ps: int,
                         out: Dict[str, np.ndarray],
                         batch: Iterable[str] = None) -> None:
    """Read coverage for a single length L once, then accumulate into 'out'."""
    print(f"[_add_length_into_out] Reading coverage for experiment={exp}, length={L}, P-site offset={ps}")
    # Assuming this is the coverage of the 5' end of reads
    cov_all = _RIBO.get_coverage(experiment=exp, range_lower=int(L), range_upper=int(L))
    to_iter = _TRANSCRIPTS if batch is None else batch
    ps_i = int(ps)
    test_num = 0
    for t in to_iter:
        if test_num == 5:
            exit()
        test_num += 1
        # raw is an array of 0's for each transcript
        raw = cov_all.get(t)
        print(f"[_add_length_into_out] Raw coverage for transcript {t}: {raw}")
        if raw is None:
            continue
        start, stop = _CDS_RANGE[t]
        start_i, stop_i = int(start), int(stop)
        if stop_i <= start_i:
            continue
        lo = start_i - ps_i
        hi = stop_i - ps_i
        seg = _safe_window(raw, lo, hi)

        print(f"[_add_length_into_out] Seg for transcript {t}: {seg}")
        
        # Write raw and seg to file for inspection (no truncation)
        with open("coverage_debug.txt", "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Experiment: {exp}, Length: {L}, Transcript: {t}\n")
            f.write(f"P-site offset: {ps_i}, Window: [{lo}, {hi}]\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"RAW COVERAGE (5' end aligned, length={len(raw)}):\n")
            f.write(f"{raw}\n\n")
            
            f.write(f"EXTRACTED SEGMENT (P-site aligned, length={len(seg)}):\n")
            f.write(f"{seg}\n\n")
        
        # Defensive: enforce exact length match
        if seg.shape[0] != out[t].shape[0]:
            fix = np.zeros_like(out[t])
            m = min(fix.shape[0], seg.shape[0])
            fix[:m] = seg[:m]
            seg = fix
        # In-place add
        out[t] += seg.astype(np.int64, copy=False)

def _process_experiment(exp: str, min_len: int, max_len: int,
                        offsets: Dict[int, int],
                        batch_size: int = 0) -> Tuple[str, Dict[str, np.ndarray]]:
    """
    Compute CDS-aligned, length-summed coverage for an experiment by:
      - loading each read length once (bulk per-length I/O),
      - applying that length's P-site offset,
      - accumulating into per-transcript arrays.
    Optional batching of transcripts to bound memory.
    """
    t0 = time.time()
    print(f"[_process_experiment] Processing {exp}: lengths {min_len}..{max_len}, batch_size={batch_size}")

    out = _preallocate_output(_TRANSCRIPTS)
    lengths = list(range(int(min_len), int(max_len) + 1))
    print(f"[_process_experiment] Processing {len(lengths)} read lengths")

    # Process lengths; one HDF5 call per L
    if batch_size and batch_size > 0:
        print(f"[_process_experiment] Using batched mode with batch_size={batch_size}")
        for L in lengths:
            ps = int(offsets.get(L, 0))
            for i in range(0, len(_TRANSCRIPTS), batch_size):
                batch = _TRANSCRIPTS[i:i + batch_size]
                _add_length_into_out(exp, L, ps, out, batch=batch)
    else:
        print(f"[_process_experiment] Processing all transcripts at once per length")
        for L in lengths:
            ps = int(offsets.get(L, 0))
            _add_length_into_out(exp, L, ps, out, batch=None)

    dt = time.time() - t0
    print(f"[_process_experiment] Completed {exp} in {dt:.2f}s")
    logging.info(f"[{exp}] done in {dt:.2f}s")
    return exp, out

def parse_args():
    p = argparse.ArgumentParser(
        description="CDS-aligned coverage via bulk per-length I/O, parallelized by experiment."
    )
    p.add_argument("--ribo", required=True, help="Path to .ribo file")
    p.add_argument("--min-len", type=int, required=True, help="Minimum read length (inclusive)")
    p.add_argument("--max-len", type=int, required=True, help="Maximum read length (inclusive)")
    p.add_argument("--site-type", help="Site type for offset")
    p.add_argument("--search-window",
                    nargs=2,
                    type=int,
                    metavar=("LO", "HI"),
                    help="Position window relative to landmark (e.g., --search-window -60 -30)"
                )
    p.add_argument("--return-site", required=True, help="Adjust coverage to P- or A-site")
    p.add_argument("--alias", action="store_true",
                   help="Use apris_human_alias (set if your .ribo uses mouse/human aliasing)")
    p.add_argument("--procs", type=int, default=1, help="Number of parallel worker processes (experiments run in parallel)")
    p.add_argument("--batch-size", type=int, default=0,
                   help="Optional transcript batch size to bound memory (0 = process all transcripts at once per length)")
    p.add_argument("--out", default="coverage_bulk_perlen_perexp.pkl.gz",
                   help="Output pickle.gz path")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"[main] Starting adj_coverage analysis")
    print(f"[main] Input: {args.ribo}")
    print(f"[main] Output: {args.out}")

    # Single handle to gather metadata and compute offsets up front
    print(f"[main] Loading .ribo file...")
    ribo0 = Ribo(args.ribo, alias=ribopy.api.alias.apris_human_alias) if args.alias else Ribo(args.ribo)
    experiments = list(ribo0.experiments)
    transcripts = list(ribo0.transcript_names)

    print(f"[main] Found {len(experiments)} experiments: {experiments}")
    print(f"[main] Found {len(transcripts)} transcripts")

    # CDS ranges (dict: transcript -> (start, stop)), cast to signed ints
    print(f"[main] Computing CDS ranges...")
    cds_range = get_cds_range_lookup(ribo0)
    cds_range = {t: (int(s), int(e)) for t, (s, e) in cds_range.items()}

    # Example Output:
    # CDS range for Y110A7A.10.1|cdna|chromosome:WBcel235:I:5107833:5110183:1|gene:WBGene00000001.1|gene_biotype:protein_coding|transcript_biotype:protein_coding|gene_symbol:aap-1: start=11, stop=1577
    # for transcript, (start, stop) in list(cds_range.items())[:5]:  # print first 5 for sanity check
    #    print(f"[main] CDS range for {transcript}: start={start}, stop={stop}")


    # Precompute per-experiment offsets (dict of dict: exp -> {L -> offset})
    print(f"[main] Computing P-site offsets for each experiment...")
    # Set experiment offset dictionar
    # Runs each experiment though get_offiset to compute offsets for each read length, then stores in exp_offsets
    exp_offsets: Dict[str, Dict[int, int]] = {}
    for exp in experiments:
        od = get_offset(ribo0, exp, args.min_len, args.max_len, args.site_type, args.search_window, args.return_site)  # user-provided
        print(f"[main] Computed offsets for {exp}: {len(od)} lengths")
        logging.info(F"Experiment {exp} offsets: {od}")
        # cast keys/values to plain ints to avoid uint overflows later
        exp_offsets[exp] = {int(L): int(o) for L, o in od.items()}

    all_coverage_dict: Dict[str, Dict[str, np.ndarray]] = {}

    # Error Handling, not Important
    if len(experiments) == 0:
        print(f"[main] ERROR: No experiments found in the .ribo file.")
        logging.warning("No experiments found in the .ribo file.")
        with gzip.open(args.out, "wb") as f:
            pickle.dump(all_coverage_dict, f)
        return 0

    # Parallelize by experiment
    print(f"[main] Starting parallel processing with {args.procs} workers...")
    with mp.Pool(
        processes=int(args.procs),
        initializer=_init_worker,
        initargs=(args.ribo, bool(args.alias), cds_range, transcripts),
    ) as pool:
        tasks = [
            (exp, int(args.min_len), int(args.max_len), exp_offsets[exp], int(args.batch_size))
            for exp in experiments
        ]
        print(f"[main] Submitting {len(tasks)} experiments for processing")
        # starmap keeps order; imap_unordered also fine if you want streaming
        for exp, out_dict in pool.starmap(_process_experiment, tasks):
            all_coverage_dict[exp] = out_dict
            print(f"[main] Received results for {exp}")

    # Save
    print(f"[main] Saving results to {args.out}...")
    with gzip.open(args.out, "wb") as f:
        pickle.dump(all_coverage_dict, f)
    print(f"[main] DONE! Saved coverage to {args.out}")
    logging.info(f"Saved coverage to {args.out}")

    return 0

if __name__ == "__main__":
    sys.exit(main())