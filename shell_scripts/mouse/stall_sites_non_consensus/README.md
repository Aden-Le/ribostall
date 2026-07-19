# stall_sites_non_consensus — Step 2b: per-replicate stall calling + Wilcoxon stats (mouse)

*Wraps the per-replicate stall caller and its A2 between-condition Wilcoxon stats. No plot launcher — a single treatment replicate leaves the Wilcoxon plots unbuildable.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [mouse](../README.md) › stall_sites_non_consensus

---

## What this stage is

Step 2b is the **non-consensus** path: stalls are called **per replicate** (no cross-replicate collapsing). The resulting stats are per-replicate rank tests only — the A2 between-condition Wilcoxon (and, for a timepoint design, the A5 between-timepoint Wilcoxon). The count-collapsing tests (A1 binomial, A3/A4 Fisher/background-diff, A6/A7 between-timepoint) pool replicates, which would be pseudoreplication on per-replicate data; by design this script contains **no** count-collapsing code, so those tests live exclusively in the two consensus stats runners. The split is structural, not merely disabled.

Outputs land under `results/mouse/stall_sites_non_consensus/`:
- `raw/` — `stall_sites_{codon,aa}.csv` + `per_group_background_{codon,aa}.csv`.
- `analysis/` — `between_condition_wilcoxon_*` (A2). A5 is skipped (no timepoints).
- `plots/` — **not produced by any script in this folder** (see below).

**Flat-design note.** `TIMEPOINTS=''`, so A5 is auto-skipped and only A2 runs.

## Why there is no `analyze_*_wilcoxon.sh` here

The C. elegans non-consensus folder ships `analyze_stall_sites_non_consensus_wilcoxon.sh` to plot A2/A5 log2-FC bar charts. Mouse **omits it deliberately**:

- The **A2 between-condition Wilcoxon** compares per-replicate frequency *distributions* between groups. It needs ≥2 replicates per group to have a distribution on each side. Mouse `treatment` has only **one** replicate (`Ch_WAA2`), so the treatment side is a single point — there is nothing to plot as a distribution.
- The **A5 between-timepoint Wilcoxon** needs timepoints; mouse has **none**.

The A2 *statistic* still gets computed by the stats runner (it can be emitted at n=1 on one arm), but no bar-plot launcher was written because there is no meaningful bar plot to draw.

## Contents

| File | Role | Summary |
|---|---|---|
| `_headline_config.sh` | shared config | `HEADLINE_CONDITION=treatment`, `OTHER_CONDITION=control`; derives the Wilcoxon comparison label/tag. Sourced by the stats runner. |
| `run_stall_sites_non_consensus.sh` | `run_*` | Calls `stall_sites_non_consensus.py` → `raw/`. |
| `run_stall_sites_non_consensus_stats.sh` | `run_*_stats` | Calls `stall_sites_non_consensus_stats.py` → A2 (A5 auto-skipped). |

*(No `analyze_*` file — see the section above.)*

---

### `_headline_config.sh`

**What it does.** Sets `HEADLINE_CONDITION="treatment"` / `OTHER_CONDITION="control"` and derives `COMPARISON_LABEL`, `COMPARISON_TAG` (`treatment_vs_control`, used as the Wilcoxon `--comparison` value / plot prefix), and `X_LABEL_RATIO`. A positive `log2_FC` means enriched in `treatment`. Sourced by the stats runner so its direction matches the headline. (It would also be sourced by an analyze launcher — but none exists here.)

---

### `run_stall_sites_non_consensus.sh`

**What it does.** Self-locates to repo root, activates `ribostall_env`, derives the pickle from `RIBO_FILE`, guards it, then runs `stall_sites_non_consensus.py` directly (a plain backslash-continued call, not a `CMD` array).

**CONFIG.**

| Variable | Value | Meaning |
|---|---|---|
| `RIBO_FILE` | `./all_ribo_file/mouse_all.ribo` | Pickle derived from it. |
| `EXP_GROUPS` | `control:AA_3,AA_4;treatment:Ch_WAA2` | Flat 2-vs-1. |
| `TX_THRESHOLD` | `0.5` | Transcript-filter threshold (v2). |
| `TX_MIN_REPS_PER_GROUP` | `control:2;treatment:1` | Per-group tx-filter support. |
| `MIN_Z` / `MIN_READS` | `1.0` / `5` | Stall-calling z / read floor. |
| `TRIM_START` / `TRIM_STOP` | `20` / `10` | CDS trim. |
| `PSEUDOCOUNT` | `0.5` | Enrichment pseudocount. |
| `BASIS` / `PSITE_OFFSET` | `P` / `0` | E/P/A register + offset. |
| `DROP_STOP_CODONS` | `True` | Drop stop-codon E/P/A windows. |
| `REFERENCE_FILE` | `./reference/appris_mouse_v2_selected.fa.gz` | |
| `RAW_DIR` | `./results/mouse/stall_sites_non_consensus/raw` | Base-CSV output. |

**Command built.**

```bash
python3 scripts/stall_sites_non_consensus.py \
  --pickle all_ribo_file/mouse_all_coverage.pkl.gz --ribo …/mouse_all.ribo \
  --reference reference/appris_mouse_v2_selected.fa.gz \
  --groups 'control:AA_3,AA_4;treatment:Ch_WAA2' \
  --tx_threshold 0.5 --tx_min_reps_per_group 'control:2;treatment:1' \
  --min_z 1.0 --min_reads 5 --trim-start 20 --trim-stop 10 --pseudocount 0.5 \
  --basis P --psite-offset 0 --drop-stop-codons True \
  --out-dir results/mouse/stall_sites_non_consensus/raw
```

**Inputs / Outputs.** In: pickle + reference. Out: `raw/stall_sites_{codon,aa}.csv`, `raw/per_group_background_{codon,aa}.csv`.

---

### `run_stall_sites_non_consensus_stats.sh`

**What it does.** Ribopy-free. Sources `_headline_config.sh`, self-locates, loops `aa`/`codon` running `stall_sites_non_consensus_stats.py`. Builds `HEADLINE_FLAG` conditionally. Note: unlike the consensus stats runners, `--timepoints` is passed **unconditionally** here (with the empty `TIMEPOINTS` value); the empty string still causes A5 to be skipped.

**CONFIG.**

| Variable | Value | Meaning |
|---|---|---|
| `EXP_GROUPS` | `control:AA_3,AA_4;treatment:Ch_WAA2` | Real per-replicate names (not collapsed). |
| `TIMEPOINTS` | `''` | Empty → A5 skipped. |
| `RAW_DIR` / `OUT_DIR` | `…/raw` / `…/analysis` | |
| `RUN_BETWEEN_CONDITION_WILCOXON` | `true` | A2. |
| `RUN_BETWEEN_TIMEPOINT_WILCOXON` | `true` | A5 — auto-skipped (no timepoints). |

**Command built** (per level):

```bash
python3 scripts/stall_sites_non_consensus_stats.py \
  --stall-sites raw/stall_sites_${LEVEL}.csv \
  --groups 'control:AA_3,AA_4;treatment:Ch_WAA2' \
  --timepoints '' \
  --out-dir …/analysis \
  --headline-condition treatment \
  --between-condition-wilcoxon true \
  --between-timepoint-wilcoxon true
```

There is no `--background` argument here — the non-consensus stats consume only the stall-site CSVs.

**Inputs / Outputs.** In: `raw/stall_sites_{aa,codon}.csv`. Out: `analysis/` A2 Wilcoxon CSV(s). No plot step follows (see the "why no launcher" section).

---

## See also

- [Python entry points](../../../scripts/README.md) — `stall_sites_non_consensus{,_stats}.py` (the Python style reference).
- [R plotting scripts](../../../R_scripts/README.md) — `between_group_barplot.R` (driven by the C. elegans Wilcoxon launcher, not present for mouse).
- [mouse organism README](../README.md) — flat design + the omitted-Wilcoxon-launcher rationale.
- [C. elegans counterpart](../../c_elegans/stall_sites_non_consensus/README.md) — includes the `analyze_*_wilcoxon.sh` that mouse omits.
- [shell_scripts top level](../../README.md) · [Repository root](../../../README.md).
