# shell_scripts — Quick-Run Reference

Copy-and-paste commands for every shell script in this directory. Commands assume:
- Terminal: **PowerShell** (Windows)
- Working directory: **repo root** (`ribostall/`)
- `bash.exe`: **`C:\Program Files\Git\bin\bash.exe`** (Git for Windows default install)

> If your bash lives elsewhere, replace the path in each command. From a Git Bash terminal (not PowerShell), you can drop the `& "C:\..."` wrapper and run `bash shell_scripts/<organism>/<stage>/<name>.sh` directly.
>
> Edit the `CONFIG` block at the top of each script to point at your data before running.
>
> **Direction (headline condition):** for the stall-sites and global-occupancy stats, the between-condition direction (which condition is the numerator, so a positive effect means "enriched there") and the matching plot labels are both read from one sourced file per pipeline — `shell_scripts/<organism>/<pipeline>/_headline_config.sh`. Set `HEADLINE_CONDITION` there once; the stats runner and the plot launchers then stay in sync automatically.

---

## Folder layout

Scripts are grouped first by **organism** (each analysis has its own data and
config), then by pipeline stage:

```
shell_scripts/
├── c_elegans/                              # C. elegans analysis (all current scripts)
│   ├── adj_coverage/                       # Step 1 — CDS-aligned coverage
│   ├── stall_sites_consensus_union/        # Step 2a (union) — consensus calling + A1/A4/A7 stats + bg-diff/within plots
│   ├── stall_sites_consensus_intersection/ # Step 2a (intersection) — consensus calling + A1/A3/A6 stats + fisher/within plots
│   ├── stall_sites_non_consensus/          # Step 2b — per-replicate calling, stats, and R plots
│   └── global_occupancy/                   # Step 3 — occupancy base CSVs, stats, and R plots
└── mouse/                                  # Mouse analysis — FLAT 2-vs-1 design (control:AA_3,AA_4;treatment:Ch_WAA2), no timepoints
    ├── adj_coverage/                       # Step 1 — run_adj_coverage_all.sh (mouse_all.ribo)
    ├── stall_sites_consensus_union/        # Step 2a (union) — call + stats + within-condition & background-diff volcano launchers
    ├── stall_sites_consensus_intersection/ # Step 2a (intersection) — call + stats + within-condition & fisher volcano launchers
    ├── stall_sites_non_consensus/          # Step 2b — call + stats (no plot launcher: A2 Wilcoxon needs ≥2 reps/group; treatment has 1)
    └── global_occupancy/                   # Step 3 — base CSVs + stats + within-condition & fisher volcano launchers
```

> The mouse run is a **flat control-vs-treatment design with no timepoints**, so
> its stats emit the flat CSVs (`between_condition_fisher_*`,
> `between_condition_background_diff_*`, one `within_condition_binomial_*` per
> group) and the plot launchers pass `--flat-design` (one composite row of A/P/E
> site panels, no day axis). The two Wilcoxon bar-plot launchers from the
> C. elegans set are intentionally **omitted** for mouse: the between-condition
> Wilcoxon (A2) needs ≥2 replicates per group and treatment has only one
> (`Ch_WAA2`), and there are no timepoints for the between-timepoint Wilcoxon.
> There is no requirement to mirror every C. elegans script.

---

## Pipeline-stage runners (`run_*.sh`)

These drive the Python pipeline in `scripts/`. Run in order for a full end-to-end pipeline.

### Step 1 — Extract CDS-aligned coverage

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/adj_coverage/run_adj_coverage_all.sh
```

### Step 2a — Consensus stall-site calling (union + intersection)

Two calling variants; run whichever you need (or both). They differ only in transcript handling: union keeps each group's own filtered set, intersection restricts to the transcripts passing in all groups.

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_union/run_stall_sites_consensus_union.sh
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_intersection/run_stall_sites_consensus_intersection.sh
```

### Step 2a-stats — Enrichment tests on consensus stall-site CSVs

Union runs A1/A4/A7 (within-condition + background-aware); intersection runs A1/A3/A6 (within-condition + Fisher).

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_union/run_stall_sites_consensus_union_stats.sh
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_intersection/run_stall_sites_consensus_intersection_stats.sh
```

### Step 2b-call — Per-replicate stall calling (non-consensus)

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/run_stall_sites_non_consensus.sh
```

### Step 2b-stats — Enrichment tests on stall-site CSVs

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/run_stall_sites_non_consensus_stats.sh
```

### Step 3 — Global codon/AA occupancy (base CSVs)

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/run_global_codon_occ.sh
```

### Step 3-stats — Statistical tests on per-site occupancy CSVs (merge folded in)

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/run_global_codon_occ_stats.sh
```

---

## R-plot launchers (`analyze_*.sh`)

These drive the 3 unified R plotting scripts in `R_scripts/`. Output goes to `results/<organism>/<pipeline>/plots/` (e.g. `results/c_elegans/global_occupancy/plots/`). Each pipeline produces only the subset of folders matching the tests it runs; the full layout (global occupancy) is:

```
plots/
├── per_timepoint_fisher/
├── within_condition_timepoint_fisher/{d10_vs_d0,d10_vs_d5,d5_vs_d0}/
├── between_condition/
├── between_timepoint/{d10_vs_d0,d10_vs_d5,d5_vs_d0}/
└── within_condition/
```

Each comparison dir contains AA plots at its root and codon plots under a `codon/` subdir.

### Stall_sites (non-consensus)

The non-consensus pipeline runs only the two per-replicate Wilcoxon tests (A2 between-condition + A5 between-timepoint); the count-collapsing tests (Fisher, background-aware, within-condition binomial) live in the consensus pipelines below, so this stage has a single plot launcher.

Wilcoxon rank-sum bar plots (between-condition BWM-vs-Control + between-timepoint comparisons):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/analyze_stall_sites_non_consensus_wilcoxon.sh
```

### Stall_sites (consensus — union)

These drive the volcano scripts on the **consensus union** stats CSVs (flat control-vs-treatment, no timepoints) in `results/c_elegans/stall_sites_consensus_union/analysis/`. The union variant runs the background-aware + within-condition tests (no Fisher).

Between-condition **background-aware** volcano plots (enrichment-over-background ratio, Treatment vs Control):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_union/analyze_stall_sites_consensus_union_background_diff_volcano.sh
```

Within-condition binomial enrichment volcano plots (per group, with Beta-Jeffreys CIs):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_union/analyze_stall_sites_consensus_union_within_condition_volcano.sh
```

### Stall_sites (consensus — intersection)

These drive the volcano scripts on the **consensus intersection** stats CSVs in `results/c_elegans/stall_sites_consensus_intersection/analysis/`. The intersection variant runs the Fisher + within-condition tests (no background-aware diff).

Between-condition Fisher volcano plots (raw stall-site share, Treatment vs Control):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_intersection/analyze_stall_sites_consensus_intersection_fisher_volcano.sh
```

Within-condition binomial enrichment volcano plots (per group, with Beta-Jeffreys CIs):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_consensus_intersection/analyze_stall_sites_consensus_intersection_within_condition_volcano.sh
```

### Global occupancy

Fisher's exact test volcano plots (per-timepoint + within-condition timepoint):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/analyze_global_occupancy_fisher_volcano.sh
```

Wilcoxon rank-sum bar plots (between-condition + between-timepoint):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/analyze_global_occupancy_wilcoxon.sh
```

Within-condition binomial enrichment volcano plots (with all-groups mega-composite):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/analyze_global_occupancy_within_condition.sh
```

---

## Run all 4 R-plot shells in sequence

For a full plot regeneration after a stats rerun (PowerShell uses `;` between commands, not `&&`):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/analyze_stall_sites_non_consensus_wilcoxon.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/analyze_global_occupancy_fisher_volcano.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/analyze_global_occupancy_wilcoxon.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/global_occupancy/analyze_global_occupancy_within_condition.sh
```

The within-condition shells are the slowest (~3 minutes each); the others finish in under a minute.

---

## Mouse

The mouse analysis has call runners for the coverage and consensus stages so
far. Their `CONFIG` blocks point at `mouse_all.ribo` and
`reference/appris_mouse_v2_selected.fa.gz`; the consensus runners use a 2-vs-1
design (`control:AA_3,AA_4;treatment:Ch_WAA2`). Edit the `CONFIG` block at the
top of each script before running.

### Step 1 — Extract CDS-aligned coverage

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/adj_coverage/run_adj_coverage_all.sh
```

### Step 2a — Consensus stall-site calling (union + intersection)

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/stall_sites_consensus_union/run_stall_sites_consensus_union.sh
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/stall_sites_consensus_intersection/run_stall_sites_consensus_intersection.sh
```

### Step 2b / Step 3 — Per-replicate calling + global occupancy (base CSVs)

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/stall_sites_non_consensus/run_stall_sites_non_consensus.sh
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/global_occupancy/run_global_codon_occ.sh
```

### Stats — Enrichment tests (run before the plot launchers below)

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/stall_sites_consensus_union/run_stall_sites_consensus_union_stats.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/stall_sites_consensus_intersection/run_stall_sites_consensus_intersection_stats.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/global_occupancy/run_global_codon_occ_stats.sh
```

### R-plot launchers (`analyze_*.sh`, flat design)

All pass `--flat-design`: one composite row of A/P/E site panels per comparison
(no timepoint axis). Output goes to `results/mouse/<pipeline>/plots/`.

```powershell
# Consensus union — within-condition binomial + background-aware diff volcano
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/stall_sites_consensus_union/analyze_stall_sites_consensus_union_within_condition_volcano.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/stall_sites_consensus_union/analyze_stall_sites_consensus_union_background_diff_volcano.sh;
# Consensus intersection — within-condition binomial + Fisher volcano
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/stall_sites_consensus_intersection/analyze_stall_sites_consensus_intersection_within_condition_volcano.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/stall_sites_consensus_intersection/analyze_stall_sites_consensus_intersection_fisher_volcano.sh;
# Global occupancy — within-condition binomial + between-condition Fisher volcano
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/global_occupancy/analyze_global_occupancy_within_condition.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/mouse/global_occupancy/analyze_global_occupancy_fisher_volcano.sh
```

> No Wilcoxon bar-plot launchers for mouse (see the flat-design note under "Folder
> layout"): treatment has a single replicate, so A2 between-condition Wilcoxon is
> infeasible and A5 between-timepoint Wilcoxon has no timepoints.
