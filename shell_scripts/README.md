# shell_scripts — Quick-Run Reference

Copy-and-paste commands for every shell script in this directory. Commands assume:
- Terminal: **PowerShell** (Windows)
- Working directory: **repo root** (`ribostall/`)
- `bash.exe`: **`C:\Program Files\Git\bin\bash.exe`** (Git for Windows default install)

> If your bash lives elsewhere, replace the path in each command. From a Git Bash terminal (not PowerShell), you can drop the `& "C:\..."` wrapper and run `bash shell_scripts/<subdir>/<name>.sh` directly.
>
> Edit the `CONFIG` block at the top of each script to point at your data before running.
>
> **Direction (headline condition):** for the stall-sites and global-occupancy stats, the between-condition direction (which condition is the numerator, so a positive effect means "enriched there") and the matching plot labels are both read from one sourced file per pipeline — `shell_scripts/<pipeline>/_headline_config.sh`. Set `HEADLINE_CONDITION` there once; the stats runner and the plot launchers then stay in sync automatically.

---

## Folder layout

Scripts are grouped by pipeline stage:

```
shell_scripts/
├── adj_coverage/             # Step 1 — CDS-aligned coverage
├── stall_sites_consensus/    # Step 2a — consensus stall calling
├── stall_sites_non_consensus/# Step 2b — per-replicate calling, stats, and R plots
└── global_occupancy/         # Step 3 — occupancy base CSVs, stats, and R plots
```

---

## Pipeline-stage runners (`run_*.sh`)

These drive the Python pipeline in `scripts/`. Run in order for a full end-to-end pipeline.

### Step 1 — Extract CDS-aligned coverage

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/adj_coverage/run_adj_coverage_all.sh
```

### Step 2a — Consensus stall-site calling

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_consensus/run_stall_sites_consensus.sh
```

### Step 2b-call — Per-replicate stall calling (non-consensus)

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_non_consensus/run_stall_sites_non_consensus.sh
```

### Step 2b-stats — Enrichment tests on stall-site CSVs

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_non_consensus/run_stall_sites_non_consensus_stats.sh
```

### Step 3 — Global codon/AA occupancy (base CSVs)

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/global_occupancy/run_global_codon_occ.sh
```

### Step 3-stats — Statistical tests on per-site occupancy CSVs (merge folded in)

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/global_occupancy/run_global_codon_occ_stats.sh
```

---

## R-plot launchers (`analyze_*.sh`)

These drive the 3 unified R plotting scripts in `R_scripts/`. Output goes to `results/{stall_sites,global_occupancy}/plots/` — both pipelines share the same 5-folder layout:

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
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_non_consensus/analyze_stall_sites_non_consensus_fisher_volcano.sh
```

Per-timepoint **background-aware** between-condition volcano plots (enrichment-over-background ratio, BWM vs Control — the background-aware counterpart of the per-timepoint Fisher plot above):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_non_consensus/analyze_stall_sites_non_consensus_background_diff_volcano.sh
```

Wilcoxon rank-sum bar plots (between-condition BWM-vs-Control + between-timepoint comparisons):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_non_consensus/analyze_stall_sites_non_consensus_wilcoxon.sh
```

Within-condition binomial enrichment volcano plots (with Beta-Jeffreys CIs):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_non_consensus/analyze_stall_sites_non_consensus_within_condition_enrichment.sh
```

### Stall_sites (consensus)

These drive the volcano scripts on the **consensus** stats CSVs (flat control-vs-treatment, no timepoints) in `results/stall_sites/enrichment/analysis_stats/`.

Between-condition Fisher volcano plots (raw stall-site share, Treatment vs Control):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_consensus/analyze_stall_sites_consensus_fisher_volcano.sh
```

Between-condition **background-aware** volcano plots (enrichment-over-background ratio, Treatment vs Control — the background-aware counterpart of the Fisher plot above):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_consensus/analyze_stall_sites_consensus_background_diff_volcano.sh
```

Within-condition binomial enrichment volcano plots (per group, with Beta-Jeffreys CIs):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_consensus/analyze_stall_sites_consensus_within_condition_volcano.sh
```

### Global occupancy

Fisher's exact test volcano plots (per-timepoint + within-condition timepoint):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/global_occupancy/analyze_global_occupancy_fisher_volcano.sh
```

Wilcoxon rank-sum bar plots (between-condition + between-timepoint):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/global_occupancy/analyze_global_occupancy_wilcoxon.sh
```

Within-condition binomial enrichment volcano plots (with all-groups mega-composite):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/global_occupancy/analyze_global_occupancy_within_condition.sh
```

---

## Run all 7 R-plot shells in sequence

For a full plot regeneration after a stats rerun (PowerShell uses `;` between commands, not `&&`):

```powershell
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_non_consensus/analyze_stall_sites_non_consensus_fisher_volcano.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_non_consensus/analyze_stall_sites_non_consensus_background_diff_volcano.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_non_consensus/analyze_stall_sites_non_consensus_wilcoxon.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/stall_sites_non_consensus/analyze_stall_sites_non_consensus_within_condition_enrichment.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/global_occupancy/analyze_global_occupancy_fisher_volcano.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/global_occupancy/analyze_global_occupancy_wilcoxon.sh;
& "C:\Program Files\Git\bin\bash.exe" shell_scripts/global_occupancy/analyze_global_occupancy_within_condition.sh
```

The within-condition shells are the slowest (~3 minutes each); the others finish in under a minute.
