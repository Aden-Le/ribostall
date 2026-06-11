import numpy as np
import pandas as pd
import bisect
from typing import Dict, List

def filter_tx(cov_by_exp: dict, reps: list[str], min_reps: int = 2, threshold: float = 1.0,
              trim_start: int = 0, trim_stop: int = 0):
    """
    Keep transcripts where at least `min_reps` of the given replicates
    have mean coverage per-nt > threshold over the elongation body
    (CDS with the first `trim_start` and last `trim_stop` codons removed,
    matching the window used by `call_stalls`).
    """
    common = set.intersection(*(set(cov_by_exp[r].keys()) for r in reps))
    trim_start_nt = trim_start * 3
    trim_stop_nt = trim_stop * 3
    keep = []
    for tx in common:
        n_pass = 0
        for r in reps:
            arr = np.asarray(cov_by_exp[r][tx], float)
            body = arr[trim_start_nt : len(arr) - trim_stop_nt] if len(arr) > trim_start_nt + trim_stop_nt else arr[:0]
            if body.size and body.mean() > threshold:
                n_pass += 1
        if n_pass >= min_reps:
            keep.append(tx)
    return keep

def codonize_counts_cds(x_nt: np.ndarray, frame: int = 0):
    """
    Codon-sum a CDS-only nt-coverage vector.
    x_nt: 1D array of per-nt counts covering ONLY the CDS (5'UTR and 3'UTR removed).
    frame: which nt within the codon the indexing should start from (0, 1, or 2).
           Use 0 if x_nt[0] corresponds to the first nt of the start codon, etc.

    Returns:
      x_codon: float array of length = number of full codons
    """
    assert x_nt.ndim == 1, "x_nt must be 1D"
    # Frame is generally 0
    frame = int(frame) % 3
    # Align start by frame and trim tail to multiple of 3
    start = frame
    # Floor division, so rounds down to last full codon
    usable_len = ((len(x_nt) - start) // 3) * 3
    if usable_len <= 0:
        return np.zeros(0, dtype=float)
    stop = start + usable_len
    cds_slice = x_nt[start:stop]                # length is multiple of 3
    # Reshapes into a 2D array where each row is a codon (3 nts), then sums across columns to get codon counts
    x3 = cds_slice.reshape(-1, 3)               # (codons, 3)
    # Sums across each row to get total counts per codon, resulting in a 1D array of codon counts
    x_codon = x3.sum(axis=1).astype(float)
    return x_codon

def global_z_log(x: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    """Transcript-wide z on log2(x+pc)."""
    x = np.asarray(x, dtype=float)
    v = np.log2(x + pseudocount)
    sd = v.std()
    if sd == 0:
        return np.zeros_like(v)
    return (v - v.mean()) / (sd + 1e-12)

def call_stalls(
    x_codon: np.ndarray,
    min_z: float = 1.0,
    min_obs: int = 2,
    trim_start: int = 10,
    trim_stop: int = 10,
    pseudocount: float = 0.5,
    trim_edges: int | None = None,
):
    """
    Keep any codon with global z >= min_z and obs >= min_obs (no local filters).

    The initiation ramp and termination region are excluded BEFORE the z-score
    is computed, so the null distribution (mean, sd) reflects only the
    elongation body of the CDS. Returned `index` values refer to positions in
    the original (untrimmed) x_codon array.

    trim_start: exclude first N codons (initiation ramp).
    trim_stop:  exclude last N codons (termination region).
    trim_edges: legacy parameter — if provided, sets both trim_start and trim_stop.
    Returns list of dicts: {index, obs, z}.
    """
    if trim_edges is not None:
        trim_start = trim_edges
        trim_stop = trim_edges
    x = np.asarray(x_codon, dtype=float)
    n = x.size
    if n == 0:
        return []
    lo, hi = trim_start, n - trim_stop
    if hi <= lo:
        return []
    # Trim first, then z-score: keeps initiation / termination peaks out of
    # the null so mid-CDS stalls aren't measured against an inflated std.
    x_body = x[lo:hi]
    z_body = global_z_log(x_body, pseudocount=pseudocount)
    # Return indices where z >= min_z and obs >= min_obs, adjusting index back to original x_codon coordinates
    cand = np.flatnonzero((z_body >= min_z) & (x_body >= min_obs))
    # Each candidate is returned as a dict with the original index (adjusted for trimming), observed count, and z-score.
    return [dict(index=int(i + lo), obs=float(x_body[i]), z=float(z_body[i])) for i in cand]

def consensus_stalls_across_reps(
    stalls_by_exp: Dict[str, dict],
    reps: List[str],
    *,
    min_support: int = 2,
    tol: int = 0,
    min_sep: int = 0,
    conflict_resolution: str = "keep_both"  # "keep_both" removes downstream preference
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

    def _indices_from_stalls(stalls_for_tx: List[dict]) -> List[int]:
        # call_stalls always yields a (possibly empty) list of dicts each with
        # an "index" key; pull out the sorted, de-duplicated indices.
        if not stalls_for_tx:
            return []
        return sorted({int(d["index"]) for d in stalls_for_tx if d is not None})

    # Only takes transcripts that are common across all replicates, ALL the reps don't have to have a stall
    # Just be present in all replicates
    common_txs = set.intersection(*(set(stalls_by_exp[r].keys()) for r in reps))
    out: Dict[str, List[int]] = {}

    for tx in common_txs:
        # The stalls indices for this transcript across all replicates
        per_rep_idx = {r: _indices_from_stalls(stalls_by_exp[r][tx]) for r in reps}
        # The union of all indices from all replicates for this transcript
        candidates = sorted(set().union(*per_rep_idx.values()))
        consensus: List[int] = []

        for c in candidates:
            support = 0
            supporting_hits: List[int] = []
            for r in reps:
                arr = per_rep_idx[r]
                j = bisect.bisect_left(arr, c - tol)
                hit = None
                # If the value is not past the last element of the array and less than or equal to the tagert plus tolerance
                if j < len(arr) and arr[j] <= c + tol:
                    hit = arr[j]
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

    `stalls_by_exp` is the nested structure produced by `call_stalls`, shaped
    {replicate_name: {transcript_key: [stall_dict, ...]}}. Each stall_dict has
    keys {"index", "obs", "z"}; the position key is also looked up under a few
    aliases (e.g. "idx", "pos", "codon_idx") for upstream flexibility.

    Columns (always present, in this order):
      group, replicate, gene, tx_id, transcript, pos_codon, obs, z
    """
    rows = []

    for rep, tx_map in stalls_by_exp.items():
        # Get Group for Replicate
        grp = rep_to_group.get(rep) if rep_to_group else None

        # For each transcript in this replicate, extract stall information and append to rows
        for tx_key, stall_list in tx_map.items():
            tx_str, tx_id, gene = parse_key(tx_key)

            #-----------------------------------
            # Ignore this, just error safety net
            if not isinstance(stall_list, list) or (stall_list and not isinstance(stall_list[0], dict)):
                raise TypeError(
                    f"stalls_to_long_df expected list[dict] for rep={rep!r}, "
                    f"tx={tx_key!r}; got {type(stall_list).__name__}"
                    + (f" of {type(stall_list[0]).__name__}" if stall_list else "")
                )
            #-----------------------------------

            # Upstream callers may use different names for the codon position;
            # try each known alias and use the first one present in the dict.
            idx_keys = ["index", "idx", "pos", "position", "codon", "codon_idx", "codon_index"]
            # If stall_list is empty, skip it
            for d in stall_list:
                if d is None:
                    continue
                idx_key = next((k for k in idx_keys if k in d), None)


                #-----------------------------------
                # Ignore this, just error safety net
                if idx_key is None:
                    raise KeyError(
                        f"stall dict for rep={rep!r}, tx={tx_key!r} has no "
                        f"recognisable position key. Expected one of {idx_keys}, "
                        f"got keys: {list(d.keys())}"
                    )
                #-----------------------------------
                
                
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

    cols = ["group","replicate","gene","tx_id","transcript","pos_codon","obs","z"]
    return pd.DataFrame(rows)[cols].sort_values(["group","replicate","gene","tx_id","pos_codon"])