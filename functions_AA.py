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
                       aa_order=AA_ORDER):
    """
    Background AA frequency across callable CDS of the same transcripts.
    Returns a Series over aa_order that sums to 1.
    """
    bg_counts = Counter()
    for tx in transcripts:
        # robust alias fallback
        key = tx if tx in cds_range else tx.split("|")[4] if "|" in tx else tx
        if key not in cds_range or tx not in sequence:
            continue  # skip missing entries gracefully
        start, stop = cds_range[key]
        cds_nt = sequence[tx][start:stop]
        aa_seq = translate_cds_nt_to_aa(cds_nt)  # assumes in-frame, stop excluded
        for aa in aa_seq:
            if aa in aa_order:
                bg_counts[aa] += 1
    bg = pd.Series({aa: bg_counts.get(aa, 0) for aa in aa_order}, dtype=float)
    # pseudocount to avoid zeros
    bg = (bg + 1e-6) / (bg.sum() + 1e-6 * len(bg))
    return bg

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
    *,
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

        for i in idx_list:
            center = i + psite_offset_codons
            e_i, p_i, a_i = center + e_rel, center + p_rel, center + a_rel
            if e_i < 0 or a_i >= L or p_i < 0 or p_i >= L:
                continue
            eaa, paa, aaa = aa_seq[e_i], aa_seq[p_i], aa_seq[a_i]
            if drop_stop_windows and ("*" in (eaa,paa,aaa) or "X" in (eaa,paa,aaa)):
                continue
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
