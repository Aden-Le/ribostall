# ribostall

*A pipeline for finding where ribosomes stall on mRNA, what amino-acid and codon contexts those stalls sit in, and how codon occupancy shifts across the transcriptome — from ribosome-profiling (Ribo-seq) data.*

Welcome. This README is the **cover and table of contents** for the whole project. It
tells the story of what ribostall does and how the pieces fit together, then points you
into a per-folder README for the in-depth "how it works under the hood." Read this page
first; open a folder's README when you want the mechanics of that layer.

---

## Table of contents

- [What ribostall does](#what-ribostall-does)
- [The pipeline, end to end](#the-pipeline-end-to-end)
- [How the statistics are organized (A1–A7)](#how-the-statistics-are-organized-a1a7)
- [Repository map — where to read next](#repository-map--where-to-read-next)
- [Inputs](#inputs)
- [Installation](#installation)
- [Quick start](#quick-start)
- [A note on outputs](#a-note-on-outputs)

---

## What ribostall does

A ribosome profiling experiment sequences the mRNA fragments that ribosomes are sitting
on at the moment of cell lysis. The density of those footprints along a transcript is a
snapshot of translation: places where ribosomes pile up are places where elongation is
slow — **stall sites**. ribostall takes the footprint data and answers three questions:

1. **Where are the stalls?** It calls positions where codon-level occupancy is unusually
   high relative to the rest of the transcript, and (optionally) requires agreement
   across replicates.
2. **What sequence context do stalls prefer?** For every stall it reads out the codons and
   amino acids sitting in the ribosome's **E, P, and A sites**, and tests which of them are
   enriched — within a condition, between conditions, and across timepoints.
3. **How does codon usage shift globally?** Separately from discrete stall sites, it
   measures transcriptome-wide **codon/amino-acid occupancy** at each ribosomal site and
   compares it across conditions and time.

The whole thing is built to compare experimental groups (e.g. a treatment vs a control,
or a developmental time course) rigorously, with careful attention to *not* fabricating
statistical power from pooled replicates (see [the statistics section](#how-the-statistics-are-organized-a1a7)).

---

## The pipeline, end to end

Everything flows from one ingestion step into two parallel analysis tracks, each of which
ends in statistics and plots:

```
        .ribo (RiboFlow)  +  reference FASTA
                     │
                     ▼
        ┌────────────────────────────┐
        │  Step 1  adj_coverage.py    │   P-site offset + CDS-aligned, length-summed
        │  → coverage pickle (.pkl.gz)│   per-transcript coverage
        └────────────────────────────┘
                     │
         ┌───────────┴─────────────────────────────┐
         ▼                                          ▼
  ┌───────────────────────┐              ┌──────────────────────────┐
  │ Step 2  Stall sites   │              │ Step 3  Global occupancy │
  │ call → annotate E/P/A  │              │ count E/P/A codon/AA      │
  │ → base CSVs            │              │ occupancy → base CSVs     │
  └───────────────────────┘              └──────────────────────────┘
         │                                          │
         ▼                                          ▼
  ┌───────────────────────┐              ┌──────────────────────────┐
  │ *_stats.py            │              │ global_codon_occ_stats.py │
  │ enrichment tests       │              │ enrichment tests          │
  │ → analysis CSVs        │              │ → merged E/P/A analysis   │
  └───────────────────────┘              └──────────────────────────┘
         │                                          │
         └──────────────┬───────────────────────────┘
                        ▼
              ┌────────────────────┐
              │ R plots (volcano,  │   driven by analyze_*.sh
              │ bar, overlay)      │
              └────────────────────┘
```

- **Step 1 — coverage.** `adj_coverage.py` reads the `.ribo` file, applies a per-read-length
  P-site offset (so each footprint is attributed to the codon in the ribosome's P-site),
  sums coverage within the CDS, and serializes a compact `{experiment: {transcript: array}}`
  pickle. Every later step reads this pickle instead of touching the big `.ribo` again.

- **Step 2 — stall sites.** The coverage is codonized, transcripts are filtered for adequate
  depth, and per-codon **z-scores** on the trimmed elongation body flag high-occupancy
  positions. Each stall is annotated with the codons/amino acids in its E, P, and A sites.
  This step comes in **two philosophies** (below), and the calling ("base CSVs") is
  deliberately separated from the statistics so the stats can run on a machine that never
  sees the `.ribo`.

- **Step 3 — global occupancy.** Independently of discrete stalls, ribostall tallies how
  much ribosome density sits on each codon/amino acid at the E, P, and A sites across the
  whole transcriptome, normalized several ways (rate, proportion, RPM).

- **Stats + plots.** Each track runs a family of enrichment tests and the R scripts turn the
  resulting CSVs into volcano plots, bar plots, and amino-acid/codon overlays.

### Two philosophies for stall calling

The Step-2 track exists in three sibling forms that differ in how they treat replicates:

- **Consensus (union)** and **consensus (intersection)** collapse replicates into one
  reproducible stall set per group (a site must recur in enough replicates). The two
  variants differ only in transcript handling: **union** lets each group keep its own
  filtered transcript set; **intersection** restricts every group to the transcripts that
  pass filtering in *all* groups.
- **Non-consensus** keeps every replicate as an independent observation and never collapses.

Why it matters is entirely about the statistics that follow.

---

## How the statistics are organized (A1–A7)

The tests carry short labels (A1–A7). The important idea is **which test is valid on which
kind of data**, and that is enforced structurally by splitting the code into separate stats
scripts rather than by a flag you could flip the wrong way.

| Label | Test | Runs on |
|-------|------|---------|
| **A1** | Within-condition binomial (observed vs background) | consensus (both variants) |
| **A2** | Between-condition Wilcoxon rank-sum (per replicate) | non-consensus |
| **A3** | Between-condition Fisher's exact | consensus **intersection** |
| **A4** | Between-condition background-aware diff | consensus **union** |
| **A5** | Between-timepoint Wilcoxon (per replicate) | non-consensus |
| **A6** | Between-timepoint Fisher within condition | consensus **intersection** |
| **A7** | Between-timepoint background-aware diff | consensus **union** |

Two design decisions explain this table:

- **No pseudoreplication.** The count-collapsing tests (binomial/Fisher/background-aware)
  pool biological replicates into a single aggregate count. Doing that on per-replicate data
  would manufacture significance. So those tests live **only** in the consensus stats scripts,
  where each group is already a single reproducibility-filtered set (pooling one set is a
  no-op). The non-consensus script contains no count-collapsing code at all — it runs only the
  Wilcoxons (A2/A5), which treat each replicate as one independent observation. The split is
  structural: there is no toggle to turn pooling back on.

- **Fisher vs background-aware.** In the **intersection** variant every condition shares one
  transcript universe, so the per-group backgrounds are identical and raw stall-site shares
  are directly comparable → **Fisher** (A3/A6) is fair. In the **union** variant the per-group
  backgrounds differ, so each condition must be normalized to its own background before
  comparison → the **background-aware diff** (A4/A7) is the valid test, and a raw Fisher would
  be confounded. (When backgrounds happen to be equal, the background-aware diff mathematically
  converges to Fisher.)

Global occupancy runs its own within-condition binomial, Wilcoxon, and Fisher tests; the
no-pseudoreplication rule currently applies to the stall-site tracks only.

The full derivations, function-by-function, live in
[ribostall/README.md](ribostall/README.md); the per-script argument tables live in
[scripts/README.md](scripts/README.md).

---

## Repository map — where to read next

Each top-level folder is a self-contained chapter. Start here, then drill in.

| Folder | What it is | Chapter |
|--------|-----------|---------|
| **scripts/** | The Python CLI entry points — one per pipeline step (coverage, stall calling, stall stats, occupancy, occupancy stats, plus a supplementary internal-stop scan). Every argument, and the step-by-step control flow of each script, is documented here. | [scripts/README.md](scripts/README.md) |
| **ribostall/** | The shared Python package the scripts import — sequence/FASTA handling, codon & amino-acid tables, stall calling, the E/P/A machinery, and the statistical kernels (`stats_core`) reused across both analysis tracks. Function reference + import graph. | [ribostall/README.md](ribostall/README.md) |
| **R_scripts/** | The unified R visualizers (volcano, bar, AA/codon overlay) that turn the stats CSVs into figures, plus their shared constants and test fixtures. | [R_scripts/README.md](R_scripts/README.md) |
| **shell_scripts/** | The orchestration layer: small, editable bash wrappers that hold your dataset's arguments and run the Python/R commands for you, grouped by organism and stage. This is what you actually run. | [shell_scripts/README.md](shell_scripts/README.md) |
| **reference/** | The reference transcriptome FASTA files (C. elegans, mouse) used for CDS sequence lookup and motif backgrounds. | [reference/README.md](reference/README.md) |
| **all_ribo_file/** | A sample RiboFlow `.ribo` input (mouse). | [all_ribo_file/README.md](all_ribo_file/README.md) |

> **This repository is the clean pipeline template.** Only the code and reference data needed
> to run it are tracked. Dataset-specific outputs and working directories — `results/`,
> `docs/`, `changelog/`, `plans/` — are git-ignored and created per analysis.

---

## Inputs

ribostall consumes two files per dataset:

- **A `.ribo` file** — a RiboFlow ([riboflow](https://github.com/ribosomeprofiling/riboflow))
  HDF5 container of RPM-normalized read coverage per experiment, read length, and transcript,
  plus CDS metadata. Read via `ribopy`. See [all_ribo_file/README.md](all_ribo_file/README.md).
- **A reference FASTA** — the transcriptome (CDS) sequences, used to translate E/P/A codons and
  to compute motif backgrounds. Gzip is auto-detected. See [reference/README.md](reference/README.md).

Example `.ribo` and reference files (human/mouse) are available in
[ribograph_sampledata](https://github.com/ribosomeprofiling/ribograph_sampledata).

**Supported platforms:** Python 3.9–3.11 on Linux/macOS (the pipeline scripts). The shell
launchers include PowerShell invocation examples for Windows via Git for Windows' bash.

---

## Installation

```bash
git clone https://github.com/reikostachibana/ribostall
cd ribostall
pip install -r requirements.txt
```

The R plotting scripts additionally need R with `ggplot2`, `optparse`, and (for logos)
the usual tidy plotting stack; see [R_scripts/README.md](R_scripts/README.md).

---

## Quick start

The intended way to run any step is through its shell wrapper: edit the `CONFIG` block at
the top for your data, then run it. From the repo root on Windows/PowerShell:

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/adj_coverage/run_adj_coverage_all.sh
```

From Git Bash / Linux / macOS:

```bash
bash shell_scripts/c_elegans/adj_coverage/run_adj_coverage_all.sh
```

The full run order (coverage → stall calling/occupancy → stats → plots) and every
launcher are in [shell_scripts/README.md](shell_scripts/README.md) and the per-organism
chapters it links to. To call the Python scripts directly instead, see the argument tables
in [scripts/README.md](scripts/README.md).

---

## A note on outputs

Every stage writes into a git-ignored `results/<organism>/<stage>/` tree with a consistent
three-part layout: `raw/` (base CSVs before statistics), `analysis/` (the stats-result
CSVs), and `plots/` (the R figures). Nothing in `results/` is version-controlled — it is
regenerated per analysis from the tracked code and your input data.

---

### Chapters

- [scripts/README.md](scripts/README.md) — Python CLI entry points
- [ribostall/README.md](ribostall/README.md) — shared analysis package
- [R_scripts/README.md](R_scripts/README.md) — R visualizers
- [shell_scripts/README.md](shell_scripts/README.md) — orchestration layer
- [reference/README.md](reference/README.md) · [all_ribo_file/README.md](all_ribo_file/README.md) — inputs
