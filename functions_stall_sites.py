import numpy as np
import pandas as pd
import bisect
from typing import Dict, List, Iterable, Any

def filter_tx(cov_by_exp: dict, reps: list[str], min_reps: int = 2, threshold: float = 1.0):
    """
    Keep transcripts where at least `min_reps` of the given replicates
    have mean coverage per-nt > threshold.
    """
    # Only keeps transcripts present in all replicates
    common = set.intersection(*(set(cov_by_exp[r].keys()) for r in reps)) # Takes all the replicates and finds the common keys (transcripts) across them
    keep = []
    for tx in common:
        # Calculates the mean coverage for this transcript in each replicate and counts how many replicates pass the threshold
        # True value = 1, False value = 0, so summing gives the count of replicates that pass the threshold
        n_pass = sum(np.asarray(cov_by_exp[r][tx], float).mean() > threshold for r in reps) 
        # If the count of replicates that pass the threshold is at least min_reps, keep this transcript
        if n_pass >= min_reps:
            keep.append(tx)
    # Returns a list of transcripts that have sufficient coverage in at least min_reps replicates
    return keep

def codonize_counts_cds(x_nt: np.ndarray, frame: int = 0):
    """
    Codon-sum a CDS-only nt-coverage vector.
    x_nt: 1D array of per-nt counts covering ONLY the CDS (5'UTR and 3'UTR removed).
    frame: which nt within the codon the indexing should start from (0, 1, or 2).
           Use 0 if x_nt[0] corresponds to the first nt of the start codon, etc.

    Returns:
      x_codon: float array of length = number of full codons
      map_codon_to_nt: list of (lo, hi) nt index slices in the original x_nt
    """
    # Confirm 1D input
    assert x_nt.ndim == 1, "x_nt must be 1D"
    # Normalize frame to [0, 2]
    frame = int(frame) % 3
    # Align start by frame and trim tail to multiple of 3
    start = frame
    # Example: if len(x_nt) = 99 and frame = 0, usable_len = 99; if frame = 1, usable_len = 98; if frame = 2, usable_len = 97. 
    # Then we trim to the nearest multiple of 3. which gives us 99, 96, and 96 respectively. This ensures that we only consider full codons starting 
    # from the specified frame.
    usable_len = ((len(x_nt) - start) // 3) * 3
    # Return nothing if no full codons are available after trimming
    if usable_len <= 0:
        return np.zeros(0, dtype=float), []
    # Gives stop
    stop = start + usable_len
    # Slices CDS into usable array
    cds_slice = x_nt[start:stop]                # length is multiple of 3
    # Reshapes the array into a 2D array where each row corresponds to a codon (3 nts). Then sums across the columns to get the total coverage 
    # for each codon. The result is a 1D array where each element corresponds to the total coverage for a codon.
    x3 = cds_slice.reshape(-1, 3)               # (codons, 3)
    # 1d Array where each element is the sum of the 3 nts in the corresponding codon. This gives us the codon-level coverage.
    x_codon = x3.sum(axis=1).astype(float)
    return x_codon

def global_z_log(x: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    """Transcript-wide z on log2(x+pc)."""
    # Ensure numeric array
    x = np.asarray(x, dtype=float)
    # The psuedocount is added to avoid taking log of zero, which would be undefined. By adding a small pseudocount, we ensure that all values are positive
    # Applies a log2 transformation to the input array for each value
    v = np.log2(x + pseudocount)
    # Calculates the standard deviation
    sd = v.std()
    if sd == 0:
        return np.zeros_like(v)
    # Computes a Z score for each codon
    return (v - v.mean()) / (sd + 1e-12)

def call_stalls(
    x_codon: np.ndarray,
    min_z: float = 1.0,
    min_obs: int = 2,
    trim_edges: int = 10,
    pseudocount: float = 0.5,
):
    """
    Keep any codon with global z >= min_z and obs >= min_obs (no local filters).
    Returns list of dicts: {index, obs, z}.
    """
    # Ensure that the array is numeric and 1D
    # Each array is from one replicate's transcript 
    # It is a codon level coverage array
    x = np.asarray(x_codon, dtype=float)
    # Gets the length
    n = x.size
    # If empty return empty list
    if n == 0:
        return []
    # Calculates the Z score for each codon
    z = global_z_log(x, pseudocount=pseudocount)
    # Trim edges by "trim_edges" (int) codons on each side. This is to avoid edge effects where coverage 
    # might be artificially low or high due to the start and stop codons. The "lo" variable represents the 
    # index of the first codon to consider after trimming the 5' end, and "hi" represents the index of the last codon 
    # to consider before trimming the 3' end. If "hi" is less than or equal to "lo", it means that there are no codons left to consider after trimming, 
    # and an empty list is returned.
    lo, hi = trim_edges, n - trim_edges
    if hi <= lo:
        return []

    # Goes through each codon and checks if it meets the criteria for being a stall site. The conditions are:
    # z >= min_z: the Z-score for the codon must be greater than or equal to the specified minimum Z-score threshold.
    # x >= min_obs: the observed coverage for the codon must be greater than or equal to the specified minimum observed coverage threshold.
    # np.arange(n) >= lo: the index of the codon must be greater than or equal to the lower bound defined by "lo" (trimming the 5' end).
    # np.arange(n) < hi: the index of the codon must be less than the upper bound defined by "hi" (trimming the 3' end).
    cand = np.flatnonzero((z >= min_z) & 
                          (x >= min_obs) & 
                          (np.arange(n) >= lo) & 
                          (np.arange(n) < hi))
    #cand is an array of indices where the conditions are met
    # [2, 3, 5] codon indices that are called as stalls

    # Returns a list of dictionaries, where each dictionary corresponds to a codon that meets the criteria for being a stall site. Each dictionary contains the following
    # keys:
    # "index": the index of the codon in the original array (the position of the codon within the transcript).
    # "obs": the observed coverage for that codon (the value from the x array at the corresponding index).
    # "z": the Z-score for that codon (the value from the z array at the corresponding index).
    # Example: {'index': 2, 'obs': 10.0, 'z': 1.8}
    return [dict(index=int(i), obs=float(x[i]), z=float(z[i])) for i in cand]

def _indices_from_stalls(stalls):
    return sorted({int(d["index"]) for d in stalls})

import numpy as np
import bisect
from typing import Dict, List

def consensus_stalls_across_reps(
    stalls_by_exp: Dict[str, dict],
    reps: List[str],
    *,
    min_support: int = 2,
    tol: int = 0,
    min_sep: int = 0,
    conflict_resolution: str = "keep_both"  # "keep_both" removes downstream preference | confict resolution is predefined here
):
    """
    Compute consensus stall indices per transcript across replicate experiments,
    allowing a tolerance window. When two candidates are closer than `min_sep`,
    resolve with `conflict_resolution`:
        - "keep_both": keep both close sites (ignores min_sep)
        - "downstream": keep downstream (your original behavior)
        - "upstream": keep upstream
        - "merge_median": replace close pair with median index
        - "drop_both": remove both close sites
    """

    def _indices_from_stalls(stalls_for_tx: Any) -> List[int]:
        # If empty return nothing
        if not stalls_for_tx:
            return []
        if isinstance(stalls_for_tx, dict) and "indices" in stalls_for_tx:
            return sorted(set(int(i) for i in stalls_for_tx["indices"]))
        if isinstance(stalls_for_tx, Iterable) and not isinstance(stalls_for_tx, (str, bytes, dict)):
            it = iter(stalls_for_tx)
            first = next(it, None)
            if first is None:
                return []
            if isinstance(first, dict):
                candidate_keys = ["idx", "index", "pos", "position", "codon", "codon_idx", "codon_index"]
                key = next((k for k in candidate_keys if k in first), None)
                if key is None:
                    raise TypeError(
                        f"Don't know which key holds the stall index in dicts. "
                        f"Expected one of {candidate_keys}, got keys: {list(first.keys())}"
                    )
                idxs = [int(d[key]) for d in stalls_for_tx if d is not None]
                return sorted(set(idxs))
            if isinstance(first, (list, tuple)):
                return sorted(set(int(x[0]) for x in stalls_for_tx))
            if isinstance(first, (int, np.integer)):
                return sorted(set(int(x) for x in stalls_for_tx))
            if isinstance(first, str):
                return sorted(set(int(x) for x in stalls_for_tx))
        if isinstance(stalls_for_tx, (int, np.integer)):
            return [int(stalls_for_tx)]
        raise TypeError(f"Unsupported stalls_for_tx shape/type: {type(stalls_for_tx)}")

    # Find common transcripts across replicates that have stall sites for each condition
    common_txs = set.intersection(*(set(stalls_by_exp[r].keys()) for r in reps))
    out: Dict[str, List[int]] = {}

    # For each common trnascript in a condition
    for tx in common_txs:
        # Run each transcript for each replicate through indices_from_stalls
        # Example input {"tx1": [{"index": 2, "obs": 10.0, "z": 1.8}, {"index": 5, "obs": 8.0, "z": 1.5}]}
        # Returns a dictionary of the form {rep1: [2, 5], rep2: [3, 5], rep3: [2, 4]} where the lists are the stall indices for that transcript in each replicate. 
        # This allows us to compare the stall sites across replicates for the same transcript.
        per_rep_idx = {r: _indices_from_stalls(stalls_by_exp[r][tx]) for r in reps}
        # Makes set of unique stall sites across all replicates
        candidates = sorted(set().union(*per_rep_idx.values()))
        consensus: List[int] = []

        # For each stall site
        for c in candidates:
            support = 0
            supporting_hits: List[int] = []
            for r in reps:
                # Goes through each replicate's list of stall sites
                arr = per_rep_idx[r] 
                # Find where c - tol will fit in the sorted list of stall sites for this replicate. 
                # This gives us the index of the first stall site that is greater than or equal to c - tol. 
                # We then check if this stall site is within c + tol. If it is, we consider it a supporting hit for candidate c.

                # Moves to the lower bound (c - tol)
                j = bisect.bisect_left(arr, c - tol)
                hit = None
                while j < len(arr) and arr[j] <= c + tol:
                    hit = arr[j]
                    break
                if hit is not None:
                    support += 1
                    supporting_hits.append(hit)

            if support >= min_support:
                rep_idx = int(np.median(supporting_hits)) if (tol > 0 and supporting_hits) else int(c)

                if not consensus:
                    consensus.append(rep_idx)
                    continue

                # handle close-by conflicts
                if rep_idx - consensus[-1] < min_sep:
                    if conflict_resolution == "downstream":
                        consensus[-1] = rep_idx
                    elif conflict_resolution == "upstream":
                        pass  # keep previous; skip new
                    elif conflict_resolution == "merge_median":
                        consensus[-1] = int(np.round(np.median([consensus[-1], rep_idx])))
                    elif conflict_resolution == "drop_both":
                        consensus.pop()  # drop previous and skip new
                    elif conflict_resolution == "keep_both":
                        consensus.append(rep_idx)  # ignore min_sep
                    else:
                        raise ValueError(f"Unknown conflict_resolution: {conflict_resolution}")
                else:
                    consensus.append(rep_idx)

        out[tx] = sorted(set(consensus))

    return out


def parse_key(k: str):
    s = str(k)
    parts = s.split("|")
    tx_id  = parts[0] if len(parts) > 0 else None
    gene   = parts[5] if len(parts) > 5 else None  # based on your key pattern
    return s, tx_id, gene

def consensus_to_long_df(consensus: dict) -> pd.DataFrame:
    rows = []
    for grp, tx_map in consensus.items():
        for tx_key, positions in tx_map.items():
            tx_str, tx_id, gene = parse_key(tx_key)
            for p in positions:
                rows.append({
                    "group": grp,
                    "transcript": tx_str,
                    "tx_id": tx_id,
                    "gene": gene,
                    "pos_codon": int(p),
                })
    return pd.DataFrame(rows).sort_values(["group","gene","tx_id","pos_codon"])


def stalls_to_long_df(stalls_by_exp: dict, rep_to_group: Dict[str, str] | None = None) -> pd.DataFrame:
    """
    Flatten per-replicate stall calls into a long dataframe.

    Columns:
      group, replicate, transcript, tx_id, gene, pos_codon, obs (optional), z (optional)
    """
    rows = []
    for rep, tx_map in stalls_by_exp.items():
        grp = rep_to_group.get(rep) if rep_to_group else None
        for tx_key, stall_list in tx_map.items():
            tx_str, tx_id, gene = parse_key(tx_key)

            # Common case: list of dicts from call_stalls()
            if isinstance(stall_list, list) and stall_list and isinstance(stall_list[0], dict):
                # accept any alias for the index key
                idx_keys = ["index", "idx", "pos", "position", "codon", "codon_idx", "codon_index"]
                for d in stall_list:
                    if d is None:
                        continue
                    # find the index
                    idx_key = next((k for k in idx_keys if k in d), None)
                    if idx_key is None:
                        continue
                    rows.append({
                        "group": grp,
                        "replicate": rep,
                        "transcript": tx_str,
                        "tx_id": tx_id,
                        "gene": gene,
                        "pos_codon": int(d[idx_key]),
                        "obs": float(d.get("obs")) if "obs" in d else None,
                        "z": float(d.get("z")) if "z" in d else None,
                    })
            else:
                # Fallback: use indices only
                for p in _indices_from_stalls(stall_list):
                    rows.append({
                        "group": grp,
                        "replicate": rep,
                        "transcript": tx_str,
                        "tx_id": tx_id,
                        "gene": gene,
                        "pos_codon": int(p),
                        "obs": None,
                        "z": None,
                    })

    cols = ["group","replicate","gene","tx_id","transcript","pos_codon","obs","z"]
    return pd.DataFrame(rows)[cols].sort_values(["group","replicate","gene","tx_id","pos_codon"])