# shell_scripts/ — Orchestration layer

*Thin, editable bash wrappers that drive the Python pipeline and the R plots one stage at a time — the "buttons" you actually press to run ribostall.*

> **[ribostall](../README.md)** › shell_scripts

Nothing in `scripts/` or `R_scripts/` is meant to be typed by hand with its full
argument list every time. Instead, each pipeline stage has a small shell script whose
job is to hold the arguments for *your* dataset in one editable place, then build and
run the underlying `python`/`Rscript` command for you. This folder is where day-to-day
runs happen: **edit the `CONFIG` block at the top of a script, then run it.**

Scripts are grouped first by **organism** (each analysis has its own data, reference,
and experimental design) and then by **pipeline stage**.

---

## Contents

| Organism | Design | README |
|----------|--------|--------|
| **c_elegans/** | Full reference pipeline — conditions `control`/`BWM` × timepoints `day_0`/`day_5`/`day_10`. Every stage and every plot launcher is present. | [c_elegans/README.md](c_elegans/README.md) |
| **mouse/** | Flat 2-vs-1 design — `control:AA_3,AA_4;treatment:Ch_WAA2`, no timepoints. Plots use `--flat-design`; the Wilcoxon launchers are intentionally omitted (see its README). | [mouse/README.md](mouse/README.md) |

Both organisms share the same six stage folders:

```
<organism>/
├── adj_coverage/                       # Step 1 — CDS-aligned coverage extraction
├── stall_sites_consensus_union/        # Step 2a (union)  — call + A1/A4/A7 stats + bg-diff/within plots
├── stall_sites_consensus_intersection/ # Step 2a (intersection) — call + A1/A3/A6 stats + fisher/within plots
├── stall_sites_non_consensus/          # Step 2b — per-replicate call + A2/A5 Wilcoxon stats (+ plots)
├── global_occupancy/                   # Step 3 — occupancy base CSVs + stats + plots
└── internal_stop_codons/               # Supplementary — internal stop-codon scan
```

---

## Conventions shared by every script

These three naming patterns and two mechanisms recur in every stage folder; learn them
once and every script reads the same way.

### Script kinds by prefix

| Prefix | Drives | Purpose |
|--------|--------|---------|
| `run_*.sh` | a `scripts/*.py` | Generates data — coverage, stall-site base CSVs, or occupancy base CSVs. |
| `run_*_stats.sh` | a `scripts/*_stats.py` | Runs the statistical tests on the base CSVs, writing the `analysis/` CSVs. |
| `analyze_*.sh` | an `R_scripts/*.R` | Renders the plots (volcano / bar / overlay) from the `analysis/` CSVs. |
| `_headline_config.sh` | *sourced, not run* | Shared direction/label config (see below). |

### The self-locating pattern

Every script begins with `#!/bin/bash`, computes the **repo root from its own path**,
and `cd`s there before doing anything. That means you can invoke a script from any
working directory and its relative paths (`scripts/…`, `results/…`, `reference/…`) still
resolve. It also means output always lands in the repo's `results/<organism>/<stage>/`
tree regardless of where you launched it.

### The editable `CONFIG` block

Immediately below the shebang each script has a clearly delimited `CONFIG` block: the
`.ribo`/pickle/reference paths, the `EXP_GROUPS` string, thresholds, and (optionally) a
conda environment to activate. **This is the only part you edit.** Below it the script
assembles the command and `eval`s it; you should rarely need to touch that half.

### The headline / direction sync (`_headline_config.sh`)

For the between-condition comparisons, the *direction* of the test (which condition is
the numerator, so a positive effect means "enriched there") and the *plot labels* must
agree. Rather than set them in two places that can drift, each pipeline has a single
`_headline_config.sh` that defines `HEADLINE_CONDITION`/`OTHER_CONDITION` and derives the
comparison label, tag, and x-axis text. It is **sourced** by both the stats runner (which
passes the numerator to `--headline-condition`) and the `analyze_*.sh` plot launchers
(which read the derived labels). Change the headline once there; stats and plots follow.

---

## Running a script (Windows / PowerShell)

The commands below assume **PowerShell**, working directory = **repo root**, and Git for
Windows' bash at `C:\Program Files\Git\bin\bash.exe`:

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/<organism>/<stage>/<name>.sh
```

> From a **Git Bash** terminal you can drop the `& "C:\..."` wrapper and run
> `bash shell_scripts/<organism>/<stage>/<name>.sh` directly. If your bash lives
> elsewhere, substitute its path.

### End-to-end order (C. elegans example)

```powershell
# Step 1 — CDS-aligned coverage
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/adj_coverage/run_adj_coverage_all.sh

# Step 2a — consensus calling (run either/both variants), then their stats
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_union/run_stall_sites_consensus_union.sh
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_union/run_stall_sites_consensus_union_stats.sh
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_intersection/run_stall_sites_consensus_intersection.sh
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_intersection/run_stall_sites_consensus_intersection_stats.sh

# Step 2b — non-consensus per-replicate calling + Wilcoxon stats
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/run_stall_sites_non_consensus.sh
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/run_stall_sites_non_consensus_stats.sh

# Step 3 — global occupancy base CSVs + stats
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/run_global_codon_occ.sh
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/run_global_codon_occ_stats.sh

# Plots — one analyze_*.sh per test family (see the stage READMEs)
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/analyze_global_occupancy_within_condition.sh
```

The per-stage READMEs list every script in that stage, its CONFIG variables, and the
exact underlying command it builds. See each organism's README for the full run recipe
and the plot launchers available.

---

## Where output goes

Each stage writes under a shared, git-ignored `results/` tree:

```
results/<organism>/<stage>/
├── raw/       # base CSVs from run_*.sh  (pre-stats)
├── analysis/  # stats CSVs from run_*_stats.sh
└── plots/     # figures from analyze_*.sh
```

The plot subfolders mirror the tests that ran; for the full global-occupancy layout
(per-timepoint / within-condition-timepoint / between-condition / between-timepoint /
within-condition, each with AA at the root and codon under a `codon/` subdir) see the
`global_occupancy` stage README.

---

## See also

- [c_elegans/README.md](c_elegans/README.md) — full timepoint pipeline (reference implementation)
- [mouse/README.md](mouse/README.md) — flat 2-vs-1 design
- [../scripts/README.md](../scripts/README.md) — the Python CLI scripts these wrappers drive
- [../R_scripts/README.md](../R_scripts/README.md) — the R plotters the `analyze_*.sh` launchers drive
- [../README.md](../README.md) — project overview and the full pipeline narrative
