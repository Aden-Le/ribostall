"""
Assemble the `C_elegan_upload/` drag-and-drop hand-off package.

For each of the first 8 reports in
`results/stall_sites/enrichment/olive_reports/_MANUAL_REVIEW.md` (all stall_sites
Wilcoxon reports), create `C_elegan_upload/stall_sites/<stem>/` and copy in:

  * the report PDF      (olive_reports/<stem>.pdf)
  * its plot PDFs       (every image referenced by olive_reports/<stem>.qmd,
                         with the `.png` extension rewritten to `.pdf` —
                         this captures the composite + the 3 individual site plots)
  * the source CSV      (enrichment/analysis_stats/<stem>.csv)

`C_elegan_upload/global_occupancy/` is created empty: no Olive reports exist for
that pipeline yet (tracked as NOT READY in C_elegan_upload/INDEX.md).

Idempotent: safe to re-run; existing copies are overwritten. Originals under
`results/` are never moved or modified — this only copies. Exits non-zero and
prints the offending paths if any expected source file is missing.

Usage:
    python scripts/_for_claude_build_c_elegan_upload.py
"""

import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OLIVE_REPORTS = REPO_ROOT / "results" / "stall_sites" / "enrichment" / "olive_reports"
ANALYSIS_STATS = REPO_ROOT / "results" / "stall_sites" / "enrichment" / "analysis_stats"
UPLOAD_ROOT = REPO_ROOT / "C_elegan_upload"

# First 8 reports, in the order they appear in
# results/stall_sites/enrichment/olive_reports/_MANUAL_REVIEW.md.
REPORT_STEMS = [
    "between_condition_wilcoxon_aa",
    "between_condition_wilcoxon_codon",
    "between_timepoint_wilcoxon_d10_vs_d0_aa",
    "between_timepoint_wilcoxon_d10_vs_d0_codon",
    "between_timepoint_wilcoxon_d10_vs_d5_aa",
    "between_timepoint_wilcoxon_d10_vs_d5_codon",
    "between_timepoint_wilcoxon_d5_vs_d0_aa",
    "between_timepoint_wilcoxon_d5_vs_d0_codon",
]

# Markdown image: ![alt](path). Alt text here contains parentheses but never a
# closing bracket; the path never contains a closing paren.
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def resolve_plot_pdfs(qmd_path):
    """Return ordered, de-duplicated list of plot PDF paths referenced by a .qmd.

    Each markdown image reference (a .png under plots/) is resolved relative to
    the qmd's directory and its extension rewritten to .pdf.
    """
    text = qmd_path.read_text(encoding="utf-8")
    seen = set()
    pdfs = []
    for rel in IMAGE_RE.findall(text):
        rel = rel.strip()
        img = (qmd_path.parent / rel).resolve()
        pdf = img.with_suffix(".pdf")
        if pdf not in seen:
            seen.add(pdf)
            pdfs.append(pdf)
    return pdfs


def main():
    missing = []
    UPLOAD_ROOT.mkdir(exist_ok=True)
    (UPLOAD_ROOT / "stall_sites").mkdir(exist_ok=True)
    (UPLOAD_ROOT / "global_occupancy").mkdir(exist_ok=True)

    for i, stem in enumerate(REPORT_STEMS, start=1):
        report_pdf = OLIVE_REPORTS / f"{stem}.pdf"
        qmd = OLIVE_REPORTS / f"{stem}.qmd"
        csv = ANALYSIS_STATS / f"{stem}.csv"
        dest = UPLOAD_ROOT / "stall_sites" / stem
        dest.mkdir(parents=True, exist_ok=True)

        sources = [report_pdf, csv]
        if qmd.exists():
            sources.extend(resolve_plot_pdfs(qmd))
        else:
            missing.append(qmd)

        copied = 0
        for src in sources:
            if not src.exists():
                missing.append(src)
                continue
            shutil.copy2(src, dest / src.name)
            copied += 1

        print(f"[{i}/8] {stem}: {copied} files copied -> {dest}")

    if missing:
        print(f"\nERROR: {len(missing)} expected source file(s) missing:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Package at: {UPLOAD_ROOT}")
    print("global_occupancy/ created empty (no Olive reports for that pipeline yet).")


if __name__ == "__main__":
    main()
