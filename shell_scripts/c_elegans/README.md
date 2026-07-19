# c_elegans — the complete reference pipeline

*Bash orchestration for the C. elegans ribosome-profiling analysis: the one organism whose every pipeline stage — process, stats, and plots — is fully wired end to end.*

> **[ribostall](../../README.md)** › [shell_scripts](../README.md) › c_elegans

The `c_elegans/` tree is the canonical, complete walk-through of the ribostall pipeline. Every other organism folder (e.g. `mouse/`) mirrors this structure but is partially populated; this one runs from a raw `.ribo` file all the way to finished R plots. If you want to understand how a ribostall analysis is assembled from shell scripts, read here first.

Each stage folder holds a small set of `.sh` launchers. Every launcher follows the same house pattern:

- It opens with `#!/bin/bash` and a header comment describing the step.
- It carries an editable **CONFIG block** (marked `# ====== CONFIG: edit these ======`) holding every path, threshold, and design string a user is expected to touch.
- It **self-locates**: `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` resolves the script's own directory, then `cd "$SCRIPT_DIR/../../.."` walks three levels up (`shell_scripts/<organism>/<stage>/` → repo root) so the script can be launched from anywhere and still find `scripts/`, `R_scripts/`, and `reference/` by relative path.
- The `run_*.sh` and `internal_stop_codons` launchers `source ${HOME}/miniconda3/etc/profile.d/conda.sh` and `conda activate ribostall_env` before running Python; the `analyze_*.sh` launchers instead prepend R to `PATH` (`export PATH="$PATH:/c/Program Files/R/R-4.4.2/bin"`) because they drive `Rscript`.
- It assembles the target command — usually into a bash array `CMD=(python3 …)` invoked as `"${CMD[@]}"`, or (in `adj_coverage`) a string built up and run with `eval $CMD` — echoing `Running: …` first so the exact invocation is visible in the log.

## Experimental design

The C. elegans dataset is a two-factor design: **condition** crossed with **timepoint**.

- **Conditions:** `control` and `BWM`. The shared `_headline_config.sh` in every stats stage fixes `HEADLINE_CONDITION="BWM"` and `OTHER_CONDITION="control"`, so a positive effect size everywhere means *enriched in BWM relative to control*.
- **Timepoints:** `day_0`, `day_5`, `day_10`, passed as `TIMEPOINTS='day_0,day_5,day_10'` (in chronological, not sorted, order — a string sort would wrongly place `day_10` before `day_5`).
- **Replicates:** `rep2` and `rep3` of each cell (`rep1` is intentionally excluded).

That gives six `(condition, timepoint)` cells, declared in every `run_*.sh` via one long `EXP_GROUPS` string:

```
EXP_GROUPS='control_day_0:control_day0_rep2,control_day0_rep3;control_day_5:control_day5_rep2,control_day5_rep3;control_day_10:control_day10_rep2,control_day10_rep3;BWM_day_0:BWM_day0_rep2,BWM_day0_rep3;BWM_day_5:BWM_day5_rep2,BWM_day5_rep3;BWM_day_10:BWM_day10_rep2,BWM_day10_rep3'
```

Because the design carries timepoints, every stats runner passes `--timepoints day_0,day_5,day_10`, so the stats emit **per-timepoint** files (BWM-vs-control at each day) *and* **between-timepoint** files (later-vs-earlier day pairs: `d10_vs_d0`, `d10_vs_d5`, `d5_vs_d0`) in addition to the pooled between-condition results.

## Pipeline order

```
1. adj_coverage                         Step 1  — extract CDS-aligned P-site coverage → *_coverage.pkl.gz
        │
        ├─ stall_sites_consensus_union          Step 2a  — consensus stall calls, union tx sets
        │       → run → run_*_stats → analyze plots
        ├─ stall_sites_consensus_intersection   Step 2a  — consensus stall calls, shared tx universe
        │       → run → run_*_stats → analyze plots
        ├─ stall_sites_non_consensus            Step 2b  — per-replicate stall calls
        │       → run → run_*_stats → analyze plots
        └─ global_occupancy                     Step 3   — transcriptome-wide codon/AA occupancy
                → run → run_*_stats → analyze plots

   (diagnostic, off the main line)
   internal_stop_codons                         scans CDS bodies for in-frame TAA/TAG/TGA
```

Within each stage the order is always the same three beats: a `run_*.sh` builds the base CSVs (the `raw/` tree), a `run_*_stats.sh` runs the statistical tests over those CSVs (the `analysis/` tree), and one or more `analyze_*.sh` launchers turn the stats CSVs into R plots (the `plots/` tree). `adj_coverage` (Step 1) must run before any Step 2/3 stage, since they all consume the coverage pickle it writes.

## Contents

| Stage | What it runs | README |
|---|---|---|
| **adj_coverage** | Step 1 — `adj_coverage.py`: extract CDS-aligned P-site coverage from the `.ribo` into a gzipped pickle. | [./adj_coverage/README.md](./adj_coverage/README.md) |
| **stall_sites_consensus_union** | Step 2a — consensus stall calling where each group keeps its **own** filtered transcript set; stats are background-aware (A1/A4/A7). | [./stall_sites_consensus_union/README.md](./stall_sites_consensus_union/README.md) |
| **stall_sites_consensus_intersection** | Step 2a — consensus stall calling restricted to the transcripts passing in **all** groups (one shared tx universe); stats are Fisher (A1/A3/A6). | [./stall_sites_consensus_intersection/README.md](./stall_sites_consensus_intersection/README.md) |
| **stall_sites_non_consensus** | Step 2b — per-replicate stall calling; stats are per-replicate Wilcoxons only (A2/A5). | [./stall_sites_non_consensus/README.md](./stall_sites_non_consensus/README.md) |
| **global_occupancy** | Step 3 — transcriptome-wide codon/AA occupancy with a merged E/P/A analysis tree; stats A1/A2/A3/A5/A6. | [./global_occupancy/README.md](./global_occupancy/README.md) |
| **internal_stop_codons** | Diagnostic — scan every CDS body for in-frame internal stop codons and report per-replicate P-site read counts. | [./internal_stop_codons/README.md](./internal_stop_codons/README.md) |

## See also

- [shell_scripts/README.md](../README.md) — top-level guide to the shell orchestration layer and its organism/stage layout.
- [scripts/README.md](../../scripts/README.md) — the Python entry points these launchers drive.
- [R_scripts/README.md](../../R_scripts/README.md) — the R plotting scripts the `analyze_*.sh` launchers drive.
- [ribostall root README](../../README.md) — project overview.
