# global_context.py
from __future__ import annotations

import logging

import argparse, os, gzip, pickle, re
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Literal
import matplotlib.patches as mpatches
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ribopy
from ribopy import Ribo
from functions import get_cds_range_lookup, get_sequence  # your helpers

# -------------------------------
# Constants & symbol maps
# -------------------------------
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_CLASS = {
    "D":"acidic","E":"acidic",
    "K":"basic","R":"basic","H":"basic",
    "A":"hydrophobic","V":"hydrophobic","I":"hydrophobic","L":"hydrophobic",
    "M":"hydrophobic","F":"hydrophobic","W":"hydrophobic","Y":"hydrophobic",
    "C":"polar","N":"polar","Q":"polar","S":"polar","T":"polar",
    "G":"neutral","P":"neutral"
}
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
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "TGG":"W","TAT":"Y","TAC":"Y","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TAA":"*","TAG":"*","TGA":"*",
}

def lex64() -> List[str]:
    bases = ["T","C","A","G"]
    return [a+b+c for a in bases for b in bases for c in bases]

# -------------------------------
# Parsing helpers
# -------------------------------
def parse_groups(s: str) -> Dict[str, List[str]]:
    """
    "kidney:rep1,rep2;lung:repA,repB" -> {"kidney":[rep1,rep2], "lung":[repA,repB]}
    """
    groups: Dict[str, List[str]] = {}
    for block in filter(None, [x.strip() for x in s.split(";")]):
        name, reps = block.split(":")
        groups[name.strip()] = [r.strip() for r in reps.split(",") if r.strip()]
    return groups

def isoform_from_header(header: str) -> str | None:
    parts = str(header).split("|")
    for tok in parts:
        if re.match(r".+-\d+$", tok):
            return tok
    return None

def get_cds_seq(full_seq_by_header: Dict[str,str], cds_range: Dict[str,Tuple[int,int]]) -> Dict[str,str]:
    out = {}
    for header, seq in full_seq_by_header.items():
        iso = isoform_from_header(header)
        if not iso or iso not in cds_range:
            continue
        start0, end0 = map(int, cds_range[iso])
        cds = seq[start0:end0].upper()
        r = len(cds) % 3
        if r != 0 and (end0 + (3-r)) <= len(seq):
            cds = seq[start0:end0 + (3-r)].upper()
        out[header] = cds
    return out

# -------------------------------
# Coverage/codonization helpers
# -------------------------------
def codonize(seq: str) -> List[str]:
    L = len(seq) - (len(seq) % 3)
    return [seq[i:i+3] for i in range(0, L, 3)]

def codonize_counts_cds(x_nt: np.ndarray, frame: int = 0) -> np.ndarray:
    assert x_nt.ndim == 1
    frame = int(frame) % 3
    start = frame
    usable_len = ((len(x_nt) - start) // 3) * 3
    if usable_len <= 0:
        return np.zeros(0, dtype=float)
    stop = start + usable_len
    x3 = x_nt[start:stop].reshape(-1, 3)
    return x3.sum(axis=1).astype(float)

def translate_codons_to_aa_idx(codons: List[str], aa_to_idx: Dict[str,int]) -> np.ndarray:
    out = np.full(len(codons), -1, dtype=np.int32)
    for i, c in enumerate(codons):
        aa = CODON2AA.get(c, "X")
        out[i] = aa_to_idx.get(aa, -1)
    return out

def codons_to_idx(codons: List[str], codon_to_idx: Dict[str,int]) -> np.ndarray:
    out = np.full(len(codons), -1, dtype=np.int32)
    for i, c in enumerate(codons):
        out[i] = codon_to_idx.get(c, -1)
    return out

# -------------------------------
# Context (vectorized)
# -------------------------------
def context_from_codon_cov_fast(
    codon_cov_rep: Dict[str, np.ndarray],
    cds_seq: Dict[str, str],
    *,
    flank_left: int = 10,
    flank_right: int = 10,
    mode: Literal["AA","codon"] = "AA",
    min_cov: float = 0.0,
    drop_stop_windows: bool = True,
    aa_order: List[str] = AA_ORDER,
    codon_order: List[str] | None = None,
    anchor_offset_codons: int = 0, 
) -> pd.DataFrame:
    """
    Vectorized global context weighted by coverage.
    Column 0 corresponds to (anchor codon index + anchor_offset_codons).

    Examples:
      - Coverage is A-site, you want P-site at 0: anchor_offset_codons = -1
      - Coverage is P-site, you want A-site at 0: anchor_offset_codons = +1
      - Disome: coverage is trailing P-site, you want leading P-site at 0 and spacing ~+10: anchor_offset_codons = +10
    """
    assert mode in ("AA","codon")
    pos_cols = np.arange(-flank_left, flank_right + 1, dtype=np.int32)
    W = pos_cols.size

    if mode == "AA":
        row_syms = list(aa_order)
        sym_to_idx = {a:i for i,a in enumerate(row_syms)}
    else:
        row_syms = codon_order if codon_order is not None else lex64()
        sym_to_idx = {c:i for i,c in enumerate(row_syms)}

    counts = np.zeros((len(row_syms), W), dtype=np.float64)

    for header, cov in codon_cov_rep.items():
        cds = cds_seq.get(header)
        if not cds:
            continue

        cods = codonize(cds)
        n = len(cods)
        if n == 0 or len(cov) != n:
            continue

        if mode == "AA":
            aa_idx = translate_codons_to_aa_idx(cods, sym_to_idx)  # -1 for unknown/*
            stop_mask = (aa_idx == sym_to_idx.get("*", -999)) if "*" in sym_to_idx else np.zeros(n, bool)
            sym_idx_arr = aa_idx
        else:
            sym_idx_arr = codons_to_idx(cods, sym_to_idx)
            stop_mask = np.zeros(n, bool)

        # anchors where coverage > threshold
        anchors = np.flatnonzero(cov > min_cov)
        if anchors.size == 0:
            continue

        # shift anchors so that column 0 = desired site
        anchors0 = anchors + int(anchor_offset_codons)

        # keep only anchors whose whole window is inside the CDS after the shift
        in_bounds = (anchors0 - flank_left >= 0) & (anchors0 + flank_right < n)
        if not np.any(in_bounds):
            continue
        anchors     = anchors[in_bounds]      # weights come from original coverage at unshifted anchors
        anchors0    = anchors0[in_bounds]     # shifted anchors define the window indices

        # (M, W) indices into transcript positions
        win_idx = anchors0[:, None] + pos_cols[None, :]  # relative to shifted anchor
        M = anchors0.size

        # optionally drop windows that contain a stop (AA mode)
        if drop_stop_windows and stop_mask.any():
            has_stop = np.any(stop_mask[win_idx], axis=1)
            if np.any(has_stop):
                keep = ~has_stop
                if not np.any(keep):
                    continue
                win_idx  = win_idx[keep]
                anchors  = anchors[keep]   # keep weights aligned

        # gather symbol ids and weights
        sym_ids = sym_idx_arr[win_idx]             # (M, W), -1 for unknown
        w = cov[anchors].astype(np.float64)         # weights from original coverage

        # accumulate by column
        for j in range(W):
            sj = sym_ids[:, j]
            mask = sj >= 0
            if not np.any(mask):
                continue
            np.add.at(counts[:, j], sj[mask], w[mask])

    return pd.DataFrame(counts, index=row_syms, columns=pos_cols.tolist())

# -------------------------------
# Background & PWM & plotting
# -------------------------------
def background_freq_fast(
    transcripts: Iterable[str],
    cds_range: Dict[str, Tuple[int,int]],
    sequence: Dict[str,str],
    *,
    mode: Literal["AA","codon"] = "AA",
    aa_order: List[str] = AA_ORDER,
    codon_order: List[str] | None = None,
    pseudocount: float = 1e-6
) -> pd.Series:
    """
    Uniform over *all callable CDS codons* of the provided transcripts.
    """
    if mode == "AA":
        row_syms = list(aa_order)
        sym_to_idx = {a:i for i,a in enumerate(row_syms)}
        acc = np.zeros(len(row_syms), dtype=np.float64)
    else:
        row_syms = codon_order if codon_order is not None else lex64()
        sym_to_idx = {c:i for i,c in enumerate(row_syms)}
        acc = np.zeros(len(row_syms), dtype=np.float64)

    for tx in transcripts:
        # allow either full header or isoform key
        key = tx if tx in cds_range else (isoform_from_header(tx) or tx)
        if key not in cds_range or tx not in sequence:
            continue
        start, stop = map(int, cds_range[key])
        cds_nt = sequence[tx][start:stop].upper()
        r = len(cds_nt) % 3
        if r != 0:
            cds_nt = cds_nt + "N"*(3-r)  # pad safely if needed

        cods = codonize(cds_nt)
        if mode == "AA":
            for c in cods:
                aa = CODON2AA.get(c, "X")
                idx = sym_to_idx.get(aa, None)
                if idx is not None:
                    acc[idx] += 1
        else:
            for c in cods:
                idx = sym_to_idx.get(c, None)
                if idx is not None:
                    acc[idx] += 1

    bg = (acc + pseudocount)
    bg = bg / bg.sum()
    return pd.Series(bg, index=row_syms, dtype=float)

def pwm_position_weighted_log2(counts_pos: pd.DataFrame, bg_freq: pd.Series, pseudocount: float = 0.5) -> pd.DataFrame:
    nonempty = counts_pos.sum(axis=0) > 0
    counts = counts_pos.loc[:, nonempty].copy()
    probs = (counts + pseudocount).div((counts + pseudocount).sum(axis=0), axis=1)
    lo = np.log2(probs.div(bg_freq, axis=0))
    return probs * lo

def plot_logo(weight_mat: pd.DataFrame, title: str = "", aa_class: Dict[str,str] | None = None, ax=None, ylim: float | None = None, basis: str = "P"):
    import logomaker
    df = weight_mat.T.copy()
    CLASS_COLORS = {
        "acidic":"#D62728","basic":"#1F77B4","hydrophobic":"#2CA02C","polar":"#9467BD","neutral":"#7F7F7F",
    }
    if aa_class is None:
        aa_class = {aa:"neutral" for aa in df.columns}
    color_scheme = {aa: CLASS_COLORS.get(aa_class.get(aa,"neutral"), "#000000") for aa in df.columns}
    if ax is None:
        ax = plt.gca()

    site_pos = {"P": { "E":-1, "P":0, "A":+1 }, "A": { "E":-2, "P":-1, "A":0 }}.get(basis.upper(), {})
    for pos in site_pos.values():
        if pos in df.index:
            ax.axvspan(pos-0.5, pos+0.5, color="lightgray", alpha=0.3, zorder=0)

    logomaker.Logo(df, color_scheme=color_scheme, shade_below=False, fade_below=False, flip_below=True, vpad=0.02, baseline_width=0, ax=ax)
    ax.set_title(title, pad=6)
    ax.set_xlabel(f"Relative position (codons; 0 = {basis.upper()}-site)" if basis in ("P","A") else "Relative position (codons)")
    ax.set_ylabel("p · log2(p / bg)")
    if ylim is not None: ax.set_ylim(-ylim, ylim)
    return ax

# -------------------------------
# IO & main
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Global codon/AA context from pooled replicate coverage, with group support.")
    p.add_argument("--ribo", required=True, help="Path to .ribo")
    p.add_argument("--pickle", required=True, help="Gzipped pickle: {exp: {tx_header: np.ndarray (CDS-only nt coverage)}}")
    p.add_argument("--reference", required=True, help="Reference FASTA/2bit path for get_sequence()")
    p.add_argument("--use-human-alias", action="store_true", help="Use ribopy.api.alias.apris_human_alias when opening the Ribo file")
    p.add_argument("--groups", required=True, help='e.g., "kidney:kidney_rep1,kidney_rep2;kidney2:kidney_rep3"')
    p.add_argument("--mode", default="AA", choices=["AA","codon"])
    p.add_argument("--flank-left", type=int, default=20)
    p.add_argument("--flank-right", type=int, default=20)
    p.add_argument("--offset", type=int, default=0, help="Anchor offset in codons (e.g., -1 if coverage is A-site and you want P=0)")
    p.add_argument("--min-cov", type=float, default=0.0, help="Ignore anchors with coverage <= this")
    p.add_argument("--keep-stop", action="store_true", help="Keep windows containing stop codons (default is drop)")
    p.add_argument("--tx-threshold", type=float, default=0.0, help="Min average reads-per-codon per transcript after pooling (e.g., 0.3)")
    p.add_argument("--outdir", default="../global_context_out")
    p.add_argument("--basis", default="P", choices=["P","A"], help="Label 0 as P-site or A-site in the plot")
    return p.parse_args()

    

# ... keep everything above ...

def main():
    args = parse_args()
    # use the top-level parse_groups you already defined
    groups = parse_groups(args.groups)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    if args.use_human_alias:
        ribo_object = Ribo(args.ribo, alias=ribopy.api.alias.apris_human_alias)
        alias_flag = True
    else:
        ribo_object = Ribo(args.ribo)
        alias_flag = False

    with gzip.open(args.pickle, "rb") as f:
        coverage_dict = pickle.load(f)

    cds_range = get_cds_range_lookup(ribo_object)
    sequence = get_sequence(ribo_object, args.reference, alias=alias_flag)
    cds_seq = get_cds_seq(sequence, cds_range)

    # Background (you can keep this, or switch to coverage-matched per group later)
    bg = background_freq_fast(
        transcripts=ribo_object.transcript_names,
        cds_range=cds_range,
        sequence=sequence,
        mode=args.mode,
        aa_order=AA_ORDER
    )

    # Convert nt -> codon coverage for every replicate
    codon_cov: Dict[str, Dict[str, np.ndarray]] = {
        rep: {tx: codonize_counts_cds(arr) for tx, arr in tx_dict.items()}
        for rep, tx_dict in coverage_dict.items()
    }

    print(codon_cov)

    row_index = AA_ORDER if args.mode == "AA" else lex64()
    pos_cols  = list(range(-args.flank_left, args.flank_right + 1))
    group2counts = {}

    for gname, reps in groups.items():
        mats = []
        for rep in reps:
            if rep not in codon_cov:
                print(f"[WARN] missing {rep}")
                continue
            c = context_from_codon_cov_fast(
                codon_cov_rep=codon_cov[rep],
                cds_seq=cds_seq,
                flank_left=args.flank_left,
                flank_right=args.flank_right,
                mode=args.mode,
                aa_order=AA_ORDER,
                codon_order=lex64() if args.mode=="codon" else None,
                anchor_offset_codons=args.offset,
                min_cov=args.min_cov,
                drop_stop_windows=not args.keep_stop,   # <-- honor CLI
            )
            c = c.reindex(index=row_index, columns=pos_cols).fillna(0.0)
            mats.append(c)

        if not mats:
            print(f"[WARN] no replicates found for {gname}")
            continue

        counts_group = reduce(lambda a, b: a.add(b, fill_value=0.0), mats)
        group2counts[gname] = counts_group
        counts_group.to_csv(outdir / f"{gname}_{args.mode.lower()}_counts.csv")

    W_by_group = {}
    for gname, counts_group in group2counts.items():
        W_by_group[gname] = pwm_position_weighted_log2(counts_group, bg_freq=bg, pseudocount=0.5)

    def pos_height(W): return W.clip(lower=0).sum(axis=0).max() if not W.empty else 0.0
    def neg_height(W): return (-W.clip(upper=0)).sum(axis=0).max() if not W.empty else 0.0
    ymax = max((pos_height(W) for W in W_by_group.values()), default=0.0)
    ymin = max((neg_height(W) for W in W_by_group.values()), default=0.0)
    Y = max(ymax, ymin)

    # plot side-by-side
    names = list(groups.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5*len(names), 4), sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, g in zip(axes, names):
        W = W_by_group[g]
        if W.empty:
            ax.text(0.5, 0.5, f"{g}\n(no data)", ha="center", va="center")
            ax.set_axis_off()
            continue
        plot_logo(
            W,
            title=f"{g.capitalize()}",
            aa_class=AA_CLASS if args.mode=="AA" else None,  # <-- use args.mode
            ax=ax,
            ylim=Y,
            basis=args.basis                                  # <-- use args.basis
        )

    CLASS_COLORS = {"acidic":"#D62728","basic":"#1F77B4","hydrophobic":"#2CA02C","polar":"#9467BD","neutral":"#7F7F7F"}
    patches = [mpatches.Patch(color=c, label=cls) for cls, c in CLASS_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches))

    panel_png = outdir / "global_context.png"
    plt.tight_layout()
    fig.savefig(panel_png, dpi=600)
    print(f"[OK] Saved image to {panel_png}")             # <-- fixed log target



if __name__ == "__main__":
    main()
