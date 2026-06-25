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
└── mouse/                                  # Mouse analysis (same stage layout, scripts added as written)
    ├── adj_coverage/                       # Step 1 — run_adj_coverage_all.sh (mouse_all.ribo)
    ├── stall_sites_consensus_union/        # Step 2a (union) — run_stall_sites_consensus_union.sh (2-vs-1 design)
    └── stall_sites_consensus_intersection/ # Step 2a (intersection) — run_stall_sites_consensus_intersection.sh (2-vs-1 design)
```

> Only the stages above have mouse scripts so far, and only the call step (no
> mouse stats/plot runners yet). The remaining stages (`stall_sites_non_consensus`,
> `global_occupancy`) are not yet present. Add a mouse script under the matching
> stage folder when one exists — there is no requirement to mirror every
> C. elegans script.

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

These drive the 3 unified R plotting scripts in `R_scripts/`. Output goes to `results/<organism>/<pipeline>/plots/` (e.g. `results/c_elegans/stall_sites_non_consensus/plots/`) — all pipelines share the same 5-folder layout:

```
plots/
├── per_timepoint_fisher/
├── within_condition_timepoint_fisher/{d10_vs_d0,d10_vs_d5,d5_vs_d0}/
├── between_condition/
├── between_timepoint/{d10_vs_d0,d10_vs_d5,d5_vs_d0}/
└── within_condition/
```

Each comparison dir contains AA plots at its root and codon plots under a `codon/` subdir.

### Stall_sites

Fisher's exact test volcano plots (per-timepoint BWM-vs-Control + within-condition timepoint comparisons):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/analyze_stall_sites_non_consensus_fisher_volcano.sh
```

Per-timepoint **background-aware** between-condition volcano plots (enrichment-over-background ratio, BWM vs Control — the background-aware counterpart of the per-timepoint Fisher plot above):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/analyze_stall_sites_non_consensus_background_diff_volcano.sh
```

Wilcoxon rank-sum bar plots (between-condition BWM-vs-Control + between-timepoint comparisons):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/analyze_stall_sites_non_consensus_wilcoxon.sh
```

Within-condition binomial enrichment volcano plots (with Beta-Jeffreys CIs):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/analyze_stall_sites_non_consensus_within_condition_enrichment.sh
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

## Run all 7 R-plot shells in sequence

For a full plot regeneration after a stats rerun (PowerShell uses `;` between commands, not `&&`):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/analyze_stall_sites_non_consensus_fisher_volcano.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/analyze_stall_sites_non_consensus_background_diff_volcano.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/analyze_stall_sites_non_consensus_wilcoxon.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/c_elegans/stall_sites_non_consensus/analyze_stall_sites_non_consensus_within_condition_enrichment.sh;
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
