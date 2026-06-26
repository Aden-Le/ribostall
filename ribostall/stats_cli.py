"""
Shared CLI / input-parsing helpers for the stats scripts.

These are the genuinely cross-script helpers used by more than one stats entry
point — the stall scripts (``stall_sites_consensus_union_stats.py``,
``stall_sites_consensus_intersection_stats.py``, and
``stall_sites_non_consensus_stats.py``) and the occupancy script
(``global_codon_occ_stats.py``). Collecting them here keeps a single source of
truth, so a fix to (say) the timepoint pairing logic lands in every consumer at
once.

  * ``parse_groups``            — CLI ``--groups`` string → {group: [rep, ...]}
  * ``parse_timepoints``        — CLI ``--timepoints`` string → ordered [tp, ...]
  * ``timepoint_token``         — 'day_10' → 'd10' short tag for filenames
  * ``build_timepoint_pairs``   — chronological [tp, ...] → later-vs-earlier pairs
  * ``build_rep_to_timepoint``  — {rep: group} → {rep: timepoint_label}
  * ``validate_timepoints``     — declared-vs-data timepoint cross-check
  * ``detect_level``            — codon vs amino-acid level from the input columns
  * ``build_replicate_counts``  — per-replicate E/P/A counts indexed by alphabet

Package rule: this module has no ``print`` and no ``sys.exit`` — user-facing
output and process exit belong in the top-level scripts. ``detect_level`` and
``validate_timepoints`` raise ``ValueError`` so the calling script can decide
how to surface each error.
"""

import pandas as pd

from ribostall.amino_acids import AA_ORDER, SENSE_CODONS


def parse_groups(groups_arg: str) -> dict:
    """Parse CLI group string into dict: {group_name: [rep1, rep2, ...]}."""
    groups = {}
    for block in groups_arg.split(";"):
        name, reps = block.split(":")
        groups[name] = reps.split(",")
    return groups


def parse_timepoints(timepoints_arg):
    """['day_0', 'day_5', 'day_10'] from 'day_0,day_5,day_10' (order preserved)."""
    return [t.strip() for t in timepoints_arg.split(",") if t.strip()]


def timepoint_token(label):
    """'day_10' -> 'd10' (legacy short tag); any other label passes through unchanged."""
    return "d" + label[len("day_"):] if label.startswith("day_") else label


def build_timepoint_pairs(timepoint_order):
    """All later-vs-earlier (time_a, time_b, tag) pairs from a chronological list.

    For ``['day_0', 'day_5', 'day_10']`` this yields, in order,
    ``('day_10', 'day_0', 'd10_vs_d0')``, ``('day_10', 'day_5', 'd10_vs_d5')``,
    ``('day_5', 'day_0', 'd5_vs_d0')`` — the same three pairs (and order) the
    script used to hard-code. ``time_a`` is the later timepoint (direction is
    later-vs-earlier).
    """
    pairs = []
    for j in range(len(timepoint_order) - 1, 0, -1):   # later: latest index down to 1
        for i in range(j):                              # earlier: 0 .. j-1
            time_a, time_b = timepoint_order[j], timepoint_order[i]
            tag = f"{timepoint_token(time_a)}_vs_{timepoint_token(time_b)}"
            pairs.append((time_a, time_b, tag))
    return pairs


def build_rep_to_timepoint(rep_to_group: dict) -> dict:
    """{rep: timepoint_label} derived from group names ('BWM_day_0' → 'day_0').

    A flat group name with no underscore (e.g. 'BWM') maps to the group name
    itself.  The caller should pass the result to ``validate_timepoints`` before
    using it in analyses.
    """
    out = {}
    for rep, grp in rep_to_group.items():
        parts = grp.split("_", 1)
        out[rep] = parts[1] if len(parts) > 1 else grp
    return out


def validate_timepoints(timepoint_order: list, rep_to_timepoint: dict) -> set:
    """Cross-check declared timepoints against those inferred from group names.

    Raises ``ValueError`` if a declared timepoint is absent from the data so
    the calling script can surface it via ``sys.exit``.  Returns the set of
    undeclared timepoints (present in the data but absent from
    ``timepoint_order``); the caller should ``logging.warning`` these as
    excluded from the analyses.
    """
    present = set(rep_to_timepoint.values())
    missing = [tp for tp in timepoint_order if tp not in present]
    if missing:
        raise ValueError(
            f"--timepoints lists {missing}, not found among the --groups "
            f"timepoints {sorted(present)}"
        )
    return present - set(timepoint_order)


def detect_level(df: pd.DataFrame) -> tuple[str, tuple[str, str, str], list, str]:
    """Return (level, (E_col, P_col, A_col), alphabet, feature_col_name)."""
    if {"E_codon", "P_codon", "A_codon"}.issubset(df.columns):
        return "codon", ("E_codon", "P_codon", "A_codon"), list(SENSE_CODONS), "codon"
    if {"E_aa", "P_aa", "A_aa"}.issubset(df.columns):
        return "aa", ("E_aa", "P_aa", "A_aa"), list(AA_ORDER), "amino_acid"
    raise ValueError(
        "Input CSV must contain either (E_codon,P_codon,A_codon) or (E_aa,P_aa,A_aa) columns."
    )


def build_replicate_counts(df: pd.DataFrame, site_cols, alphabet) -> dict:
    """{rep: {"E": Series, "P": Series, "A": Series}} indexed by ``alphabet``."""
    out = {}
    for rep, sub in df.groupby("replicate"):
        rep_map = {}
        for site_name, col in zip(("E", "P", "A"), site_cols):
            counts = sub[col].value_counts()
            rep_map[site_name] = counts.reindex(alphabet, fill_value=0).astype(int)
        out[rep] = rep_map
    return out
