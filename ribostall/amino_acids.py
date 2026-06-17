import os

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict
from pathlib import Path
import logomaker
import matplotlib.pyplot as plt

# --- mappings ---
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

AA_CLASS = {
    "D":"acidic","E":"acidic",
    "K":"basic","R":"basic","H":"basic",
    "A":"hydrophobic","V":"hydrophobic","I":"hydrophobic","L":"hydrophobic",
    "M":"hydrophobic","F":"hydrophobic","W":"hydrophobic","Y":"hydrophobic",
    "C":"polar","N":"polar","Q":"polar","S":"polar","T":"polar",
    "G":"neutral","P":"neutral"
}

AA_ORDER = [a for a in "ACDEFGHIKLMNPQRSTVWY"]  # no stop

# Sense codons (61 total, stops excluded), alphabetical.
SENSE_CODONS = sorted(c for c, aa in CODON2AA.items() if aa != "*")

# Stop codons (TAA/TAG/TGA), alphabetical. Used by the --drop-stop-codons flag.
STOP_CODONS = sorted(c for c, aa in CODON2AA.items() if aa == "*")

CLASS_COLORS = {
    "acidic":"#D62728",      # red
    "basic":"#1F77B4",       # blue
    "hydrophobic":"#2CA02C", # green
    "polar":"#9467BD",       # purple
    "neutral":"#7F7F7F",     # gray
}

def translate_cds_nt_to_aa(cds_nt: str) -> str:
    cds_nt = cds_nt.upper().replace("U","T")
    aas = []
    for i in range(0, len(cds_nt) - (len(cds_nt) % 3), 3):
        cod = cds_nt[i:i+3]
        aas.append(CODON2AA.get(cod, "X"))  # X for unknown/ambiguous
    return "".join(aas)

def windows_aa(consensus_group: dict, cds_range: dict, sequence: dict,
               flank_left=10, flank_right=10, psite_offset_codons=0):
    """
    Build AA windows of size (flank_left + 1 + flank_right) around each stall
    aligned at the P-site index (i + psite_offset_codons).
    Returns: list of AA lists, each length = window_len.
    """
    print(f"Building AA windows with flank_left={flank_left}, flank_right={flank_right}, psite_offset_codons={psite_offset_codons}")
    W = flank_left + 1 + flank_right
    win_list = []
    for tx, idx_list in consensus_group.items():
        # harmonize key for cds_range
        key = tx if tx in cds_range else tx.split("|")[4]
        start, stop = cds_range[key]
        cds_nt = sequence[tx][start:stop]
        aa_seq = translate_cds_nt_to_aa(cds_nt)
        Lcod = len(aa_seq)
        for i in idx_list:
            center = i + psite_offset_codons  # codon index for P-site
            lo = center - flank_left
            hi = center + flank_right + 1
            if lo < 0 or hi > Lcod:
                continue
            win = list(aa_seq[lo:hi])
            if "*" in win:    # optional: drop windows containing stop
                continue
            if "X" in win:    # drop ambiguous
                continue
            win_list.append(win)
    return win_list  # list of lists

def count_matrix(win_list, aa_order=AA_ORDER, flank_left=10, flank_right=10):
    """rows=AA, cols=relative positions (-flank_left..+flank_right)."""
    if not win_list:
        return pd.DataFrame(0, index=aa_order,
                            columns=list(range(-flank_left, flank_right+1)))
    W = flank_left + 1 + flank_right
    # ensure all windows consistent
    for w in win_list:
        assert len(w) == W, f"Window length {len(w)} != {W}"
    cols = list(range(-flank_left, flank_right + 1))
    counts = pd.DataFrame(0, index=aa_order, columns=cols, dtype=int)
    for win in win_list:
        for j, aa in enumerate(win):
            pos = j - flank_left  # -L..0..+R
            if aa in counts.index:
                counts.loc[aa, pos] += 1
    return counts

def background_aa_freq(transcripts: dict, cds_range: dict, sequence: dict,
                       aa_order=AA_ORDER, *, trim_start: int = 0, trim_stop: int = 0):
    """
    Background AA frequency across callable CDS of the same transcripts.
    Returns a Series over aa_order that sums to 1.

    ``trim_start`` / ``trim_stop`` drop the first / last N codons of each CDS
    so the background mirrors the elongation body used by ``call_stalls``.
    """
    bg_counts = Counter()
    for tx in transcripts:
        # Gets the transript key for cds range
        key = tx if tx in cds_range else tx.split("|")[4] if "|" in tx else tx
        if key not in cds_range or tx not in sequence:
            continue  # skip missing entries gracefully
        start, stop = cds_range[key]
        cds_nt = sequence[tx][start:stop]
        # Gets the amino acid sequence for the CDS (X for unknown, * for stop)
        aa_seq = translate_cds_nt_to_aa(cds_nt)
        lo = trim_start
        hi = len(aa_seq) - trim_stop
        if hi <= lo:
            continue
        for aa in aa_seq[lo:hi]:
            if aa in aa_order:
                bg_counts[aa] += 1
    # Series of the counts, aligned to aa_order (missing AAs get 0)
    bg = pd.Series({aa: bg_counts.get(aa, 0) for aa in aa_order}, dtype=float)
    bg_counts_csv = bg.copy()
    # pseudocount to avoid zeros, Normalied
    bg = (bg + 1e-6) / (bg.sum() + 1e-6 * len(bg))
    return bg, bg_counts_csv

def pwm_position_weighted_log2(counts_pos, bg_freq, pseudocount=0.5):
    counts = counts_pos.copy()
    # columns with any counts
    has = counts.sum(axis=0) > 0
    # compute probs only where data exists
    probs = pd.DataFrame(0.0, index=counts.index, columns=counts.columns)
    tmp = (counts.loc[:, has] + pseudocount)
    tmp = tmp.div(tmp.sum(axis=0), axis=1)
    probs.loc[:, has] = tmp

    # align bg to AA index
    bg = bg_freq.reindex(counts.index)
    # avoid division by zero (shouldn't happen with your bg)
    bg = bg.clip(lower=1e-12)

    lo = np.log2(probs.div(bg, axis=0))  # log ratio
    W = probs * lo                       # p * log2(p/bg)
    return W

def plot_logo(weight_mat, title: str = "", aa_class=None, ax=None, ylim=None):
    df = weight_mat.T.copy()  # index=positions, columns=AAs

    # Color map by AA class
    CLASS_COLORS = {
        "acidic":      "#D62728",  # red
        "basic":       "#1F77B4",  # blue
        "hydrophobic": "#2CA02C",  # green
        "polar":       "#9467BD",  # purple
        "neutral":     "#7F7F7F",  # gray
    }
    if aa_class is None:
        aa_class = {aa: "neutral" for aa in df.columns}
    color_scheme = {
        aa: CLASS_COLORS.get(aa_class.get(aa, "neutral"), "#000000")
        for aa in df.columns
    }

    if ax is None:
        ax = plt.gca()

    # Highlight E/P/A site spans (P-site basis: -1, 0, +1)
    for pos in (-1, 0, 1):
        if pos in df.index:
            ax.axvspan(pos - 0.5, pos + 0.5, color="lightgray", alpha=0.3, zorder=0)

    # Build logo with class-based colors for pos & neg values
    logomaker.Logo(
        df,
        color_scheme=color_scheme,
        shade_below=False,
        fade_below=False,
        flip_below=True,
        vpad=0.02,
        baseline_width=0,
        ax=ax,
    )

    # --- Styling to match your newer function ---
    ax.set_title(title, pad=3, fontsize=14)
    ax.set_ylabel("p · log₂(p / bg)", fontsize=12)
    ax.set_xlabel("")  # remove x-axis label entirely
    if ylim is not None:
        ax.set_ylim(-ylim, ylim)

    # Remove x-axis ticks and bottom spine; keep a minimal y-axis
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=10, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)

    ax.margins(x=0)
    return ax

def epa_triplet_counts(
    consensus_group: dict[str, list[int]],   # {tx: [stall_codons]}
    cds_range: dict,                         # {tx_or_alias: (start,stop)}
    sequence: dict,                          # {tx: genomic nt sequence}
    *,                                       # Everything After * must be name = value, else default
    psite_offset_codons: int = 0,
    basis: str = "P",                        # "P": E=-1,P=0,A=+1 ; "A": E=-2,P=-1,A=0
    drop_stop_windows: bool = True,
    aa_order = AA_ORDER
):
    """
    Unweighted EPA triplet counts at *stall codons only*.
    Returns:
      - counts_epa: Series indexed by MultiIndex(E,P,A) over aa_order (full 20^3)
      - counts_E, counts_P, counts_A: per-site marginals (Series over aa_order)
    """
    assert basis in ("P","A")
    e_rel, p_rel, a_rel = (-1,0,+1) if basis=="P" else (-2,-1,0)
    AA_SET = set(aa_order)

    triplet_counter = Counter()
    E_counter = Counter(); P_counter = Counter(); A_counter = Counter()

    for tx, idx_list in consensus_group.items():
        # harmonize key for cds_range
        key = tx if tx in cds_range else (tx.split("|")[4] if "|" in tx else tx)
        if key not in cds_range or tx not in sequence:
            continue
        start, stop = cds_range[key]
        cds_nt = sequence[tx][start:stop]
        # translate to AA string (X for unknown, * for stop)
        aa_seq = translate_cds_nt_to_aa(cds_nt)
        L = len(aa_seq)
        if L < 3: 
            continue
        
        # i is the P-site codon index
        for i in idx_list:
            center = i + psite_offset_codons
            # Offsets for E/P/A sites based on chosen basis
            e_i, p_i, a_i = center + e_rel, center + p_rel, center + a_rel
            if e_i < 0 or a_i >= L or p_i < 0 or p_i >= L:
                continue
            eaa, paa, aaa = aa_seq[e_i], aa_seq[p_i], aa_seq[a_i]
            if drop_stop_windows and ("*" in (eaa,paa,aaa) or "X" in (eaa,paa,aaa)):
                continue
            # Counts the amino acid for each site
            if (eaa in AA_SET) and (paa in AA_SET) and (aaa in AA_SET):
                triplet_counter[(eaa, paa, aaa)] += 1
                E_counter[eaa] += 1
                P_counter[paa] += 1
                A_counter[aaa] += 1

    # materialize full 20^3 index with zeros
    mi = pd.MultiIndex.from_product([aa_order, aa_order, aa_order], names=["E","P","A"])
    counts_epa = pd.Series(0, index=mi, dtype=int)
    if triplet_counter:
        counts_epa.update(pd.Series(triplet_counter))

    # marginals as Series over aa_order
    counts_E = pd.Series([E_counter.get(a,0) for a in aa_order], index=aa_order, name="E")
    counts_P = pd.Series([P_counter.get(a,0) for a in aa_order], index=aa_order, name="P")
    counts_A = pd.Series([A_counter.get(a,0) for a in aa_order], index=aa_order, name="A")

    return counts_epa, counts_E, counts_P, counts_A

def annotate_stalls_epa(
    df: pd.DataFrame,
    cds_range: dict,
    sequence: dict,
    *,
    psite_offset_codons: int = 0,
    basis: str = "P",
    drop_invalid: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Annotate each per-stall row with its E/P/A codon and E/P/A amino acid.

    Parameters
    ----------
    df : long-format dataframe from ``stalls_to_long_df`` with at minimum a
        ``transcript`` column and a ``pos_codon`` column (codon index within CDS).
    cds_range, sequence : lookups produced by ``ribostall.sequence``.
    psite_offset_codons, basis : same semantics as ``epa_triplet_counts``.
    drop_invalid : if True, drop rows that cannot be annotated — where any of
        E/P/A falls outside the CDS, the transcript is unresolvable, or a codon
        is ambiguous/unknown (``X``). Stop codons (``*``) are NOT dropped here;
        stop removal is owned by the callers' ``--drop-stop-codons`` flag so it
        happens exactly once and consistently across the pipeline.

    Returns
    -------
    (df_codon, df_aa) : two dataframes with the same rows (post filtering),
        the first with added columns ``E_codon, P_codon, A_codon`` and the
        second with ``E_aa, P_aa, A_aa``.
    """
    assert basis in ("P", "A")
    e_rel, p_rel, a_rel = (-1, 0, +1) if basis == "P" else (-2, -1, 0)

    # Cache per-transcript codon lists so we don't re-split the same CDS for
    # every stall row on that transcript.
    codon_cache: dict[str, list[str]] = {}

    def _get_codons(tx: str):
        # Returns a list of codons in order for the given transcript
        cached = codon_cache.get(tx)
        if cached is not None:
            return cached
        key = tx if tx in cds_range else (tx.split("|")[4] if "|" in tx else tx)
        if key not in cds_range or tx not in sequence:
            codon_cache[tx] = None
            return None
        start, stop = cds_range[key]
        cds_nt = sequence[tx][start:stop].upper().replace("U", "T")
        n_codons = len(cds_nt) // 3
        codons = [cds_nt[3 * i : 3 * i + 3] for i in range(n_codons)]
        codon_cache[tx] = codons
        return codons

    e_cod_out, p_cod_out, a_cod_out = [], [], []
    e_aa_out, p_aa_out, a_aa_out = [], [], []
    keep = []
    
    # for each stall site, get the transcript and codon position, compute E/P/A codon and AA based on offsets,
    for row in df.itertuples(index=False):
        tx = getattr(row, "transcript")
        i = int(getattr(row, "pos_codon")) + psite_offset_codons
        codons = _get_codons(tx)
        if codons is None:
            keep.append(False)
            e_cod_out.append(None); p_cod_out.append(None); a_cod_out.append(None)
            e_aa_out.append(None); p_aa_out.append(None); a_aa_out.append(None)
            continue
        L = len(codons)
        ei, pi, ai = i + e_rel, i + p_rel, i + a_rel
        if ei < 0 or ai >= L or pi < 0 or pi >= L:
            keep.append(False)
            e_cod_out.append(None); p_cod_out.append(None); a_cod_out.append(None)
            e_aa_out.append(None); p_aa_out.append(None); a_aa_out.append(None)
            continue
        e_cod, p_cod, a_cod = codons[ei], codons[pi], codons[ai]
        e_aa = CODON2AA.get(e_cod, "X")
        p_aa = CODON2AA.get(p_cod, "X")
        a_aa = CODON2AA.get(a_cod, "X")
        # Only ambiguous/unknown (X) codons are "invalid" here; stop codons (*)
        # are kept and removed downstream via --drop-stop-codons.
        invalid = any(aa == "X" for aa in (e_aa, p_aa, a_aa))
        keep.append(not invalid)
        e_cod_out.append(e_cod); p_cod_out.append(p_cod); a_cod_out.append(a_cod)
        e_aa_out.append(e_aa); p_aa_out.append(p_aa); a_aa_out.append(a_aa)

    base = df.copy()
    base["E_codon"] = e_cod_out
    base["P_codon"] = p_cod_out
    base["A_codon"] = a_cod_out
    base["E_aa"] = e_aa_out
    base["P_aa"] = p_aa_out
    base["A_aa"] = a_aa_out
    if drop_invalid:
        base = base.loc[pd.Series(keep, index=base.index)].reset_index(drop=True)

    codon_cols = [c for c in base.columns if c not in ("E_aa", "P_aa", "A_aa")]
    aa_cols = [c for c in base.columns if c not in ("E_codon", "P_codon", "A_codon")]
    return base[codon_cols].copy(), base[aa_cols].copy()


def background_codon_freq(transcripts, cds_range: dict, sequence: dict,
                          codon_order=SENSE_CODONS, *,
                          trim_start: int = 0, trim_stop: int = 0):
    """
    Background codon-usage frequency across callable CDS of ``transcripts``.
    Mirrors ``background_aa_freq`` but at codon granularity.

    ``trim_start`` / ``trim_stop`` drop the first / last N codons of each CDS
    so the background mirrors the elongation body used by ``call_stalls``.

    Returns (bg_freq_series, bg_counts_series) both indexed by ``codon_order``.
    """
    bg_counts = Counter()
    codon_set = set(codon_order)
    for tx in transcripts:
        key = tx if tx in cds_range else (tx.split("|")[4] if "|" in tx else tx)
        if key not in cds_range or tx not in sequence:
            continue
        start, stop = cds_range[key]
        cds_nt = sequence[tx][start:stop].upper().replace("U", "T")
        n_codons = len(cds_nt) // 3
        lo = trim_start
        hi = n_codons - trim_stop
        if hi <= lo:
            continue
        for c_idx in range(lo, hi):
            cod = cds_nt[3 * c_idx : 3 * c_idx + 3]
            if cod in codon_set:
                bg_counts[cod] += 1
    counts = pd.Series({c: bg_counts.get(c, 0) for c in codon_order}, dtype=float)
    bg = (counts + 1e-6) / (counts.sum() + 1e-6 * len(counts))
    return bg, counts


def epa_enrichment(counts_epa: pd.Series, bg_aa_freq: pd.Series, pseudocount: float = 0.5) -> pd.Series:
    """
    W(E,P,A) = p(EPA) * log2( p(EPA) / (q_E q_P q_A) ), using stall-only triplet counts.
    """
    # background over AA_ORDER
    bg = bg_aa_freq.reindex(AA_ORDER).fillna(1e-6)
    qE = counts_epa.index.get_level_values("E").map(bg)
    qP = counts_epa.index.get_level_values("P").map(bg)
    qA = counts_epa.index.get_level_values("A").map(bg)
    q_trip = np.clip(qE.values * qP.values * qA.values, 1e-12, None)

    C = counts_epa.astype(float).values
    p = (C + pseudocount) / (C.sum() + pseudocount * (20**3))
    W = p * np.log2(p / q_trip)
    return pd.Series(W, index=counts_epa.index)


def epa_pairwise_matrix(counts_epa: pd.Series, pair: str = "EP") -> pd.DataFrame:
    """
    Collapse the third site to get a 20x20 matrix for heatmaps.
    pair ∈ {"EP","PA","EA"}.
    """
    assert pair in ("EP", "PA", "EA")
    s = counts_epa
    if pair == "EP":
        mat = s.groupby(level=["E", "P"]).sum().unstack("P")
    elif pair == "PA":
        mat = s.groupby(level=["P", "A"]).sum().unstack("A")
    else:  # "EA"
        mat = s.groupby(level=["E", "A"]).sum().unstack("A")
    return mat.reindex(index=AA_ORDER, columns=AA_ORDER).fillna(0.0)


def plot_top_triplets_multi(
    epa_enrich_by_group: dict[str, pd.Series],
    groups=("kidney","liver","lung"),
    N: int = 25,
    *,
    title: str | None = None,
    ylim: tuple[float,float] | None = None,
    fontsize: int = 11,
    legend: bool = True,
    out_csv: str | None = None,
):
    """
    Grouped bar plot of top-N EPA triplets by mean enrichment across selected groups.
    epa_enrich_by_group[g] must be a Series indexed by (E,P,A) with enrichment values.

    Returns (fig, ax, pivot) where pivot is a Triplet x Group table of enrichments.
    """
    # Collect and standardize into one DataFrame
    parts = {}
    for g in groups:
        s = epa_enrich_by_group.get(g)
        if s is None: 
            continue
        # Ensure the Series has 3 levels (E,P,A)
        if s.index.nlevels != 3:
            raise ValueError(f"{g}: enrichment Series must be indexed by (E,P,A)")
        s = s.rename("enrich")
        parts[g] = s

    if not parts:
        raise ValueError("No matching groups found in epa_enrich_by_group.")

    df = pd.concat(parts, names=["Group"]).reset_index()  # columns: Group,E,P,A,enrich (or unnamed for E,P,A)

    # Normalize level names to E,P,A if unnamed (0,1,2)
    aa_cols = [c for c in df.columns if c not in ("Group","enrich")]
    if len(aa_cols) != 3:
        raise ValueError(f"Expected 3 AA index levels, got columns {aa_cols}")
    rename_map = {}
    # If they aren't already named E,P,A, force them
    if set(aa_cols) != {"E","P","A"}:
        rename_map = {aa_cols[0]:"E", aa_cols[1]:"P", aa_cols[2]:"A"}
        df = df.rename(columns=rename_map)

    # Build compact triplet labels
    df["Triplet"] = df[["E","P","A"]].agg(''.join, axis=1)

    # Pick top-N by mean enrichment across chosen groups
    top_triplets = (
        df.groupby("Triplet")["enrich"]
          .mean()
          .nlargest(N)
          .index
    )
    df_top = df[df["Triplet"].isin(top_triplets)]

    # Pivot to Triplet x Group
    pivot = (
        df_top.pivot_table(index="Triplet", columns="Group", values="enrich", fill_value=0.0)
    )
    # Order rows/cols
    pivot = pivot.loc[top_triplets, [g for g in groups if g in pivot.columns]]

    # === Plot ===
    fig, ax = plt.subplots(figsize=(max(8, 0.35*len(pivot)*len(pivot.columns)), 3.0))
    x = range(len(pivot))
    ncols = max(1, len(pivot.columns))
    width = 0.8 / ncols

    for i, g in enumerate(pivot.columns):
        x_off = [xi + (i - (ncols-1)/2)*width for xi in x]
        ax.bar(x_off, pivot[g].values, width=width, label=g)

    ax.set_xticks(list(x), pivot.index, rotation=90, fontsize=fontsize-1)
    ax.set_ylabel(r"$p \cdot \log_2\!\left(p / (q_E q_P q_A)\right)$", fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize+1, pad=4)

    # y-lims unified or auto
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Styling: grid + minimal spines
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    for spine in ("top","right"):
        ax.spines[spine].set_visible(False)

    if legend:
        ax.legend(frameon=False, fontsize=fontsize)

    fig.tight_layout()

    # Optional: save the table
    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pivot.to_csv(out_path)

    return fig, ax, pivot
