# global_occupancy — Step 3: transcriptome-wide codon/AA occupancy + stats (mouse)

*Wraps the occupancy calculator and its A1/A2/A3 stats, then drives the Fisher and within-condition plot launchers. No Wilcoxon launcher — a single treatment replicate leaves the A2 bar plot unbuildable.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [mouse](../README.md) › global_occupancy

---

## What this stage is

Step 3 computes **global** codon and amino-acid occupancy: per-experiment occupancy tables over the whole transcriptome, normalized to a single shared transcriptome background (not per-group backgrounds — so there is no enrichment-ratio x-label in this stage's headline config). The compute step writes base CSVs to `raw/`; the stats step runs the tests and writes the **merged** analysis tree (the E/P/A per-site frames are computed internally, concatenated with a `site` column, and only the merged result is exported — the per-site frames are an internal intermediate).

Outputs land under `results/mouse/global_occupancy/`:
- `raw/` — base occupancy CSVs.
- `analysis/` — merged stats CSVs: `{aa,codon}_within_condition_binomial.csv` (A1), `{aa,codon}_between_condition_wilcoxon.csv` (A2), `{aa,codon}_between_condition_fisher.csv` (A3).
- `plots/` — Fisher volcanoes + within-condition bar plots.

**Flat-design note.** `TIMEPOINTS=''`: A2/A3 run flat between-condition, and the between-timepoint tests A5/A6 are skipped automatically. Both launchers pass `--flat-design`.

## Why there is no `analyze_global_occupancy_wilcoxon.sh`

The C. elegans occupancy folder ships a Wilcoxon bar-plot launcher; mouse **omits it deliberately**, for the same reason as the non-consensus stage:

- The **A2 between-condition Wilcoxon** compares per-replicate occupancy *distributions* between groups and needs ≥2 replicates per group. Mouse `treatment` has only **one** replicate (`Ch_WAA2`), so there is no treatment-side distribution to plot.
- The **A5 between-timepoint Wilcoxon** needs timepoints; mouse has **none**.

The A2 statistic is still computed by the stats runner (it emits `{aa,codon}_between_condition_wilcoxon.csv`), but no bar-plot launcher exists because there is nothing meaningful to draw at n=1 on one arm.

## Contents

| File | Role | Summary |
|---|---|---|
| `_headline_config.sh` | shared config | `HEADLINE_CONDITION=treatment`, `OTHER_CONDITION=control`; derives Wilcoxon/Fisher labels. No enrichment-ratio x-label (shared background). |
| `run_global_codon_occ.sh` | `run_*` | Calls `global_codon_occ.py` → `raw/` base CSVs. |
| `run_global_codon_occ_stats.sh` | `run_*_stats` | Loops levels; calls `global_codon_occ_stats.py` over sites E/P/A → merged `analysis/` CSVs (A1/A2/A3). |
| `analyze_global_occupancy_fisher_volcano.sh` | `analyze_*` | Drives `between_group_volcano.R` on the A3 Fisher CSVs (`--flat-design`). |
| `analyze_global_occupancy_within_condition.sh` | `analyze_*` | Drives `within_condition_volcano.R` on the A1 CSVs (`--flat-design`, unweighted-only). |

---

### `_headline_config.sh`

**What it does.** Sets `HEADLINE_CONDITION="treatment"` / `OTHER_CONDITION="control"` and derives `COMPARISON_LABEL="treatment vs control"` and `COMPARISON_TAG`. Unlike the stall-site configs, it defines **no** `X_LABEL_RATIO`: global occupancy normalizes to one shared transcriptome background, so only the Wilcoxon (A2) and Fisher (A3) directions are headline-driven. Sourced by the stats runner and the Fisher launcher.

---

### `run_global_codon_occ.sh`

**What it does.** Self-locates to repo root, activates `ribostall_env`, derives the pickle from `RIBO_FILE`, guards it, then runs `global_codon_occ.py` via a `CMD=(...)` array.

**CONFIG.**

| Variable | Value | Meaning |
|---|---|---|
| `RIBO_FILE` | `./all_ribo_file/mouse_all.ribo` | Pickle derived from it. |
| `EXP_GROUPS` | `control:AA_3,AA_4;treatment:Ch_WAA2` | Filters coverage dict to declared reps. |
| `TRIM_START` / `TRIM_STOP` | `20` / `10` | CDS trim (codons). |
| `DROP_STOP_CODONS` | `True` | Exclude TAA/TAG/TGA before computing occupancy. |
| `REFERENCE_FILE` | `./reference/appris_mouse_v2_selected.fa.gz` | |
| `OUT_DIR` | `./results/mouse/global_occupancy` | Base CSVs go to `OUT_DIR/raw/`. |

**Command built.**

```bash
python3 scripts/global_codon_occ.py \
  --pickle all_ribo_file/mouse_all_coverage.pkl.gz --ribo …/mouse_all.ribo \
  --reference reference/appris_mouse_v2_selected.fa.gz \
  --groups 'control:AA_3,AA_4;treatment:Ch_WAA2' \
  --trim-start 20 --trim-stop 10 --drop-stop-codons True \
  --out-dir results/mouse/global_occupancy
```

**Inputs / Outputs.** In: pickle + reference. Out: base occupancy CSVs in `results/mouse/global_occupancy/raw/`.

**Related.** `OUT_DIR` must match the stats runner's.

---

### `run_global_codon_occ_stats.sh`

**What it does.** Self-locates to repo root, sources `_headline_config.sh`, and loops `for level in codon aa` (defined by `LEVELS=(codon aa)`), passing all three sites at once (`SITES=(E P A)`) to `global_codon_occ_stats.py`. The Python side computes the per-site frames internally and writes only the merged CSVs. Builds `HEADLINE_FLAG` and `TIMEPOINTS_FLAG` conditionally (the latter empty → omitted).

**CONFIG.**

| Variable | Value | Meaning |
|---|---|---|
| `EXP_GROUPS` | `control:AA_3,AA_4;treatment:Ch_WAA2` | Per-replicate (occupancy is not consensus-collapsed). |
| `TIMEPOINTS` | `''` | Flat → `--timepoints` omitted; A5/A6 skipped. |
| `OUT_DIR` | `./results/mouse/global_occupancy` | Reads `OUT_DIR/raw`, writes `OUT_DIR/analysis`. |
| `SITES` | `(E P A)` | All three ribosome sites per invocation. |
| `LEVELS` | `(codon aa)` | Looped. |
| `RUN_WITHIN_CONDITION` | `true` | A1. |
| `RUN_BETWEEN_CONDITION_WILCOXON` | `true` | A2. |
| `RUN_BETWEEN_CONDITION_FISHER` | `true` | A3. |
| `RUN_BETWEEN_TIMEPOINT_WILCOXON` | `true` | A5 — auto-skipped. |
| `RUN_BETWEEN_TIMEPOINT_FISHER` | `true` | A6 — auto-skipped. |

**Command built** (per level):

```bash
python3 scripts/global_codon_occ_stats.py \
  --raw-dir results/mouse/global_occupancy/raw \
  --analysis-dir results/mouse/global_occupancy/analysis \
  --level ${level} --sites E P A \
  --groups 'control:AA_3,AA_4;treatment:Ch_WAA2' \
  --headline-condition treatment \
  --within-condition true \
  --between-condition-wilcoxon true \
  --between-condition-fisher true \
  --between-timepoint-wilcoxon true \
  --between-timepoint-fisher true
```

**Inputs / Outputs.** In: `raw/` base CSVs. Out: merged `analysis/{aa,codon}_within_condition_binomial.csv` (A1), `{aa,codon}_between_condition_wilcoxon.csv` (A2), `{aa,codon}_between_condition_fisher.csv` (A3).

---

### `analyze_global_occupancy_fisher_volcano.sh`

**What it does.** Adds R to PATH, sources `_headline_config.sh`, self-locates. Defines a `run_volcano()` helper that skips-with-a-note on a missing input, then calls it for AA (→ `plots/between_condition_fisher/`) and codon (→ its `codon/` subdir).

**CONFIG.** `INPUT_DIR=…/analysis`, `PLOTS_DIR=…/plots`, `FORMAT=both`, `DPI=300`.

**Command built** (per level):

```bash
Rscript R_scripts/between_group_volcano.R \
  --input analysis/${LEVEL}_between_condition_fisher.csv \
  --outdir plots/between_condition_fisher[/codon] \
  --level ${LEVEL} \
  --flat-design \
  --comparison-label "treatment vs control" \
  --composite-tag fisher \
  --format both --dpi 300
```

Note the occupancy CSV naming is `{level}_between_condition_fisher.csv` (level-prefixed), whereas the stall-site trees use `between_condition_fisher_{level}.csv`. `--flat-design` → one composite row of A/P/E panels.

**Inputs / Outputs.** In: A3 Fisher CSVs. Out: PDF+PNG volcanoes under `plots/between_condition_fisher/`.

---

### `analyze_global_occupancy_within_condition.sh`

**What it does.** Adds R to PATH, self-locates, runs `within_condition_volcano.R` for aa then codon. Builds `OPTIONAL_FLAGS=(--flat-design --mega-composite)` and appends `--y-cap` only if `Y_CAP` is non-empty (it is empty here, so no cap is passed).

**CONFIG.**

| Variable | Value | Meaning |
|---|---|---|
| `INPUT_DIR` | `…/analysis` | |
| `OUTPUT_DIR` | `…/plots/within_condition` | AA at root, codon in `codon/`. |
| `ENRICHMENT_TYPE` | `unweighted` | Shared transcriptome background makes the frequency-weighted enrichment add little; weighted volcanoes were dropped from the global reports. |
| `FORMAT` / `DPI` | `both` / `300` | |
| `Y_CAP` | `""` (empty) | No y-axis cap. |

**Command built** (aa; codon → `…/codon` with `--level codon`):

```bash
Rscript R_scripts/within_condition_volcano.R \
  --input analysis/aa_within_condition_binomial.csv \
  --outdir plots/within_condition \
  --level aa --enrichment-type unweighted --format both --dpi 300 \
  --flat-design --mega-composite
```

Unlike the stall-site within-condition launchers, this one uses `--enrichment-type unweighted`, passes **no** `--show-ci`, and (with `Y_CAP=""`) no `--y-cap`.

**Inputs / Outputs.** In: A1 CSVs. Out: PDF+PNG bar plots under `plots/within_condition/`.

---

## See also

- [Python entry points](../../../scripts/README.md) — `global_codon_occ{,_stats}.py`.
- [R plotting scripts](../../../R_scripts/README.md) — `between_group_volcano.R`, `within_condition_volcano.R`.
- [mouse organism README](../README.md) — flat design + the omitted-Wilcoxon-launcher rationale.
- [C. elegans counterpart](../../c_elegans/global_occupancy/README.md) — includes the Wilcoxon launcher mouse omits.
- [shell_scripts top level](../../README.md) · [Repository root](../../../README.md).
