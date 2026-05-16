"""
Insert a single-letter `aa` column after the `codon` column in every Top-hits
markdown table inside an Olive codon .qmd. Idempotent: if a table's header row
already contains `aa` immediately after `codon`, the table is left alone.

Side effects also handled:
  * Strips inline AA annotations from codon cells (e.g. `AAG (K, lysine)` ->
    `AAG`, `TGT (Cys)` -> `TGT`).
  * Updates the YAML `tbl-colwidths` field (inserts a small width for the new
    `aa` column right after the existing `codon` width).
  * Appends a one-sentence note to the `## Top hits` intro paragraph mentioning
    the new column (only once per file).

This is a one-shot maintenance helper used to action the
`[ ] For Codon Tables can we put the amino acid counterpart`
checklist item in `results/stall_sites/enrichment/olive_reports/_MANUAL_REVIEW.md`.

Usage:
    python scripts/_for_claude_add_aa_col_to_codon_qmds.py <qmd_path> [<qmd_path> ...]
"""

import argparse
import re
import sys
from pathlib import Path

CODON_TO_AA = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

INLINE_AA_RE = re.compile(r"^([ACGT]{3})\s*\([^)]+\)\s*$")
INTRO_NOTE = " The `aa` column is the single-letter amino acid translation of each codon."


def _split_row(line: str) -> list[str] | None:
    if not line.startswith("|") or not line.rstrip().endswith("|"):
        return None
    inner = line.rstrip()[1:-1]
    return [cell.strip() for cell in inner.split("|")]


def _join_row(cells: list[str], widths: list[int]) -> str:
    padded = [cell.ljust(width) for cell, width in zip(cells, widths)]
    return "| " + " | ".join(padded) + " |"


def _is_separator(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", c) for c in cells)


def _clean_codon_cell(cell: str) -> str:
    m = INLINE_AA_RE.match(cell)
    if m:
        return m.group(1)
    return cell


def transform_codon_tables(text: str) -> tuple[str, dict]:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    stats = {"tables_modified": 0, "tables_skipped_already_done": 0}

    while i < n:
        line = lines[i]
        cells = _split_row(line)
        # Detect a table header that starts with `codon` as the first column.
        if cells and len(cells) >= 2 and cells[0].lower() == "codon" and i + 1 < n:
            sep_cells = _split_row(lines[i + 1])
            if sep_cells and len(sep_cells) == len(cells) and _is_separator(sep_cells):
                # If second column is already `aa`, skip.
                if len(cells) >= 2 and cells[1].lower() == "aa":
                    stats["tables_skipped_already_done"] += 1
                    out.append(line)
                    i += 1
                    continue
                # Collect the table body.
                body_start = i + 2
                j = body_start
                data_rows: list[list[str]] = []
                while j < n:
                    row_cells = _split_row(lines[j])
                    if row_cells is None or len(row_cells) != len(cells):
                        break
                    data_rows.append(row_cells)
                    j += 1

                # Build new header and rows with an inserted `aa` column.
                new_header = [cells[0], "aa"] + cells[1:]
                new_separator = ["---"] * len(new_header)
                new_data: list[list[str]] = []
                for row in data_rows:
                    codon_cell = _clean_codon_cell(row[0])
                    aa = CODON_TO_AA.get(codon_cell.upper(), "?")
                    new_row = [codon_cell, aa] + row[1:]
                    new_data.append(new_row)

                # Column widths: max content width per column.
                all_rows = [new_header, new_separator] + new_data
                widths = [
                    max(len(all_rows[r][c]) for r in range(len(all_rows)))
                    for c in range(len(new_header))
                ]
                # Separator gets dashes at least as wide as the column.
                new_separator = ["-" * w for w in widths]
                all_rows[1] = new_separator

                for row in all_rows:
                    out.append(_join_row(row, widths))

                stats["tables_modified"] += 1
                i = j
                continue

        out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), stats


def update_yaml_tbl_colwidths(text: str) -> tuple[str, bool]:
    """Insert an aa-column width (small) right after the codon-column width.

    Heuristic: the first entry in tbl-colwidths corresponds to `codon`; insert
    a small width (8) at position 1 if not already done. We detect "already
    done" by checking whether the array length already matches the new table
    width by re-parsing the file's largest codon table - but simpler: stamp a
    marker comment so we don't double-insert.
    """
    pattern = re.compile(r"^tbl-colwidths:\s*\[([^\]]+)\](\s*#\s*aa-col-added)?\s*$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return text, False
    if m.group(2):  # already stamped
        return text, False
    parts = [p.strip() for p in m.group(1).split(",")]
    if not parts:
        return text, False
    new_parts = [parts[0], "8"] + parts[1:]
    new_line = f"tbl-colwidths: [{', '.join(new_parts)}]  # aa-col-added"
    return text[: m.start()] + new_line + text[m.end() :], True


def add_intro_note(text: str) -> tuple[str, bool]:
    """Append a one-sentence aa-column note to the `## Top hits` intro paragraph.

    Idempotent via INTRO_NOTE marker.
    """
    if INTRO_NOTE.strip() in text:
        return text, False
    # Match `## Top hits` followed by a blank line then the intro paragraph
    # (one or more non-blank lines), ending at the next blank line.
    pat = re.compile(
        r"(## Top hits\s*\n\s*\n)((?:[^\n]+\n)+?)(\n)",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return text, False
    intro = m.group(2).rstrip("\n")
    new_intro = intro + INTRO_NOTE + "\n"
    return text[: m.start()] + m.group(1) + new_intro + m.group(3) + text[m.end() :], True


def process_file(path: Path) -> dict:
    original = path.read_text(encoding="utf-8")
    text, stats = transform_codon_tables(original)
    text, yaml_changed = update_yaml_tbl_colwidths(text)
    text, intro_changed = add_intro_note(text)
    if text != original:
        path.write_text(text, encoding="utf-8")
    return {
        **stats,
        "yaml_updated": yaml_changed,
        "intro_updated": intro_changed,
        "file_changed": text != original,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("qmds", nargs="+", type=Path, help="Codon .qmd file(s) to edit in place.")
    args = ap.parse_args()

    failures = 0
    for qmd in args.qmds:
        if not qmd.exists():
            print(f"[SKIP] {qmd} does not exist", file=sys.stderr)
            failures += 1
            continue
        result = process_file(qmd)
        print(f"[OK]   {qmd}: {result}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
