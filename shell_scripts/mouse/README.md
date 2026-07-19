# mouse — flat 2-vs-1 shell-script orchestration for the ribostall pipeline

*The Bash launcher layer that drives the ribostall ribosome-profiling pipeline over the mouse dataset: a flat `control`-vs-`treatment` design with no timepoints.*

> **[ribostall](../../README.md)** › [shell_scripts](../README.md) › mouse

---

## What "mouse" is

Every folder under `shell_scripts/mouse/` is a **stage** of the ribostall pipeline, and every `.sh` file inside a stage is a thin, editable wrapper around one `scripts/*.py` (processing/stats) or one `R_scripts/*.R` (plotting) call. The Bash layer exists so that a run is reproducible from a single, self-documenting file: each script self-locates the repo root from its own path, activates the conda env (for the Python runners), fills a CONFIG block with the exact inputs and thresholds for this dataset, assembles the underlying command, echoes it, and runs it.

The shared mechanics — the self-locating `#!/bin/bash` pattern, the CONFIG block, the `_headline_config.sh` direction file, the `eval`/array command-build idiom — are documented in depth in the per-stage READMEs here and, at greater length, in the C. elegans tree. This organism README and the mouse stage READMEs focus on what makes **mouse different**: the flat design.

## The mouse design: flat 2-vs-1, no timepoints

Mouse is a **flat control-vs-treatment** experiment. Across every run script the groups are declared identically:

```bash
EXP_GROUPS='control:AA_3,AA_4;treatment:Ch_WAA2'
```

- **control** has two biological replicates: `AA_3`, `AA_4`.
- **treatment** has a single replicate: `Ch_WAA2`.

There are **no timepoints**. Every stats runner leaves `TIMEPOINTS=''`, so the `--timepoints` flag is never passed, and every between-timepoint analysis (A5 Wilcoxon, A6 Fisher, A7 background-diff) is skipped automatically. The CONFIG blocks point at `all_ribo_file/mouse_all.ribo` and `reference/appris_mouse_v2_selected.fa.gz`; the coverage pickle is always derived as `mouse_all_coverage.pkl.gz` from the exact `.ribo` name (never globbed, so a co-present C. elegans pickle can't be picked up by mistake).

### Contrast with C. elegans

The C. elegans tree (`../c_elegans/README.md`) runs a **timepoint** design: its groups carry `_day_N` suffixes, `TIMEPOINTS` is populated in chronological order, and the between-timepoint analyses fire. Its plots use a by-condition × by-day grid.

Mouse strips all of that away, with three downstream consequences you'll see repeated in the stage docs:

1. **The stats emit the flat CSVs.** The consensus/occupancy stats write `between_condition_fisher_*`, `between_condition_background_diff_*`, and one `within_condition_binomial_*` per group — no per-timepoint or between-timepoint variants.
2. **The analyze launchers pass `--flat-design`.** Every mouse plot launcher adds `--flat-design` to the R script. The flat CSVs carry only a `site` column (no timepoint/condition grouping column), so the R scripts render **one composite row of A/P/E site panels** for the single comparison instead of a day axis.
3. **The Wilcoxon bar-plot launchers are intentionally omitted.** There is no `analyze_*_wilcoxon.sh` in `stall_sites_non_consensus/` or `global_occupancy/`. The between-condition Wilcoxon (A2) needs ≥2 replicates per group to compare per-replicate frequency distributions, and `treatment` has only one replicate (`Ch_WAA2`); the between-timepoint Wilcoxon (A5) needs timepoints, and mouse has none. The Wilcoxon *stats* still run (A2 in the non-consensus and occupancy stats runners), but there is nothing worth plotting at n=1 in one arm, so no launcher was written.

## Pipeline order

1. **`adj_coverage/`** — Step 1. Extract CDS-aligned P-site coverage from `mouse_all.ribo` into `mouse_all_coverage.pkl.gz`. Every later stage consumes this pickle.
2. **Stall-site calling + stats** (three parallel trees, all built on the same pickle):
   - **`stall_sites_consensus_union/`** — Step 2a (union): cross-replicate consensus where each group keeps its own filtered transcript set → A1 within-condition binomial + A4 background-aware diff.
   - **`stall_sites_consensus_intersection/`** — Step 2a (intersection): consensus restricted to the transcripts passing in *all* groups (shared universe) → A1 + A3 Fisher.
   - **`stall_sites_non_consensus/`** — Step 2b: per-replicate stall calling → A2 between-condition Wilcoxon only.
3. **`global_occupancy/`** — Step 3. Transcriptome-wide codon/AA occupancy → A1 within-condition binomial, A2 Wilcoxon, A3 Fisher.
4. **`internal_stop_codons/`** — read-only diagnostic (any time after Step 1): scan each CDS for in-frame internal stop codons and pull per-replicate P-site counts.

## Contents

| Stage | What it runs | README |
|---|---|---|
| `adj_coverage/` | Step 1: `adj_coverage.py` → `mouse_all_coverage.pkl.gz` | [./adj_coverage/README.md](./adj_coverage/README.md) |
| `stall_sites_consensus_union/` | Step 2a union: call + A1/A4 stats + 2 flat-design volcano launchers | [./stall_sites_consensus_union/README.md](./stall_sites_consensus_union/README.md) |
| `stall_sites_consensus_intersection/` | Step 2a intersection: call + A1/A3 stats + 2 flat-design volcano launchers | [./stall_sites_consensus_intersection/README.md](./stall_sites_consensus_intersection/README.md) |
| `stall_sites_non_consensus/` | Step 2b: per-replicate call + A2 Wilcoxon stats (no plot launcher) | [./stall_sites_non_consensus/README.md](./stall_sites_non_consensus/README.md) |
| `global_occupancy/` | Step 3: occupancy + A1/A2/A3 stats + Fisher & within-condition launchers | [./global_occupancy/README.md](./global_occupancy/README.md) |
| `internal_stop_codons/` | Diagnostic: internal in-frame stop scan → 2 CSVs | [./internal_stop_codons/README.md](./internal_stop_codons/README.md) |

## See also

- [C. elegans shell scripts](../c_elegans/README.md) — the timepoint counterpart; the shared launcher mechanics are documented in depth there.
- [shell_scripts top level](../README.md) — organism-by-stage overview.
- [Python entry points](../../scripts/README.md) — the `scripts/*.py` these launchers call.
- [R plotting scripts](../../R_scripts/README.md) — the `R_scripts/*.R` the analyze launchers drive.
- [Repository root](../../README.md).
