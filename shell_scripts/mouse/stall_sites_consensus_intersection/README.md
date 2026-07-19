# stall_sites_consensus_intersection — Step 2a (intersection): consensus stall calling + Fisher stats (mouse)

*Wraps the intersection consensus caller and its A1/A3 stats, then drives the two flat-design volcano launchers — for the mouse `control`-vs-`treatment` design where every group shares one transcript universe.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [mouse](../README.md) › stall_sites_consensus_intersection

---

## What this stage is

This is the **intersection** variant of Step 2a. Where the union caller lets each group keep its own filtered transcript set, the intersection caller restricts *every* group to the transcripts that pass filtering in **all** groups — so all conditions share one transcript universe. Raw stall-site shares are then apples-to-apples, which makes **Fisher's exact test** (A3) the valid between-condition comparison. (The per-group backgrounds are identical here, which would make the background-aware diff degenerate — that test lives in the union tree.) The within-condition binomial (A1) still uses the per-group background.

Outputs land under `results/mouse/stall_sites_consensus_intersection/`:
- `raw/` — `stall_sites_{codon,aa}.csv` + `per_group_background_{codon,aa}.csv`.
- `analysis/` — `within_condition_binomial_{aa,codon}.csv` (A1), `between_condition_fisher_{aa,codon}.csv` (A3).
- `plots/` — R volcano plots.

**Flat-design note.** `TIMEPOINTS=''`: A3 emits a single flat between-condition Fisher CSV and A6 (within-condition-timepoint Fisher) is skipped automatically. The launchers pass `--flat-design` (one composite row of A/P/E panels).

The two consensus trees are **siblings**: the run scripts differ only in transcript handling (the Python entry point they call), and their CONFIG blocks are otherwise identical.

## Contents

| File | Role | Summary |
|---|---|---|
| `_headline_config.sh` | shared config | `HEADLINE_CONDITION=treatment`, `OTHER_CONDITION=control`; derives labels. Sourced by the stats runner and both launchers. |
| `run_stall_sites_consensus_intersection.sh` | `run_*` | Calls `stall_sites_consensus_intersection.py` → `raw/`. |
| `run_stall_sites_consensus_intersection_stats.sh` | `run_*_stats` | Calls `stall_sites_consensus_intersection_stats.py` → A1 + A3. |
| `analyze_stall_sites_consensus_intersection_fisher_volcano.sh` | `analyze_*` | Drives `between_group_volcano.R` on the A3 Fisher CSVs (`--flat-design`). |
| `analyze_stall_sites_consensus_intersection_within_condition_volcano.sh` | `analyze_*` | Drives `within_condition_volcano.R` on the A1 CSVs (`--flat-design`). |

---

### `_headline_config.sh`

**What it does.** Identical structure to the union tree's config. Sets `HEADLINE_CONDITION="treatment"` / `OTHER_CONDITION="control"` and derives `COMPARISON_LABEL="treatment vs control"`, `COMPARISON_TAG`, and `X_LABEL_RATIO`. Here a positive **log2 odds ratio** means "enriched in `treatment`". Sourced by the stats runner (→ `--headline-condition`) and both plot launchers, so numerator and labels stay locked together.

---

### `run_stall_sites_consensus_intersection.sh`

**What it does.** Same launcher pattern as its union sibling: self-locates to repo root, activates `ribostall_env`, derives the pickle from `RIBO_FILE`, guards it, and runs `stall_sites_consensus_intersection.py` via a `CMD=(...)` array.

**CONFIG.** Identical to the union run script except the output tree:

| Variable | Value |
|---|---|
| `RIBO_FILE` | `./all_ribo_file/mouse_all.ribo` |
| `EXP_GROUPS` | `control:AA_3,AA_4;treatment:Ch_WAA2` |
| `TX_THRESHOLD` | `0.5` |
| `TX_MIN_REPS_PER_GROUP` | `control:2;treatment:1` |
| `MIN_Z` / `MIN_READS` | `1.0` / `5` |
| `TRIM_START` / `TRIM_STOP` | `20` / `10` |
| `PSEUDOCOUNT` | `0.5` |
| `STALL_MIN_REPS_PER_GROUP` | `control:2;treatment:1` |
| `TOL` / `MIN_SEP` | `0` / `0` |
| `REFERENCE_FILE` | `./reference/appris_mouse_v2_selected.fa.gz` |
| `BASIS` / `PSITE_OFFSET` | `P` / `0` |
| `DROP_STOP_CODONS` | `True` |
| `OUT_DIR` | `results/mouse/stall_sites_consensus_intersection/raw` |

**Command built.**

```bash
python3 scripts/stall_sites_consensus_intersection.py \
  --pickle all_ribo_file/mouse_all_coverage.pkl.gz --ribo …/mouse_all.ribo \
  --reference reference/appris_mouse_v2_selected.fa.gz \
  --groups 'control:AA_3,AA_4;treatment:Ch_WAA2' \
  --tx_threshold 0.5 --tx_min_reps_per_group 'control:2;treatment:1' \
  --min_z 1.0 --min_reads 5 --trim-start 20 --trim-stop 10 --pseudocount 0.5 \
  --stall_min_reps_per_group 'control:2;treatment:1' --tol 0 --min_sep 0 \
  --basis P --psite-offset 0 --drop-stop-codons True \
  --out-dir results/mouse/stall_sites_consensus_intersection/raw
```

**Inputs / Outputs.** In: pickle + reference. Out: `raw/stall_sites_{codon,aa}.csv`, `raw/per_group_background_{codon,aa}.csv`.

---

### `run_stall_sites_consensus_intersection_stats.sh`

**What it does.** Ribopy-free. Sources `_headline_config.sh`, self-locates, loops `aa`/`codon` running `stall_sites_consensus_intersection_stats.py`. Builds `HEADLINE_FLAG` and `TIMEPOINTS_FLAG` conditionally (the latter empty here).

**CONFIG.**

| Variable | Value | Meaning |
|---|---|---|
| `EXP_GROUPS` | `control:control;treatment:treatment` | Consensus rep == group. |
| `TIMEPOINTS` | `''` | Flat → `--timepoints` omitted, A6 skipped. |
| `RAW_DIR` / `OUT_DIR` | `…/raw` / `…/analysis` | |
| `RUN_WITHIN_CONDITION` | `true` | A1. |
| `RUN_BETWEEN_CONDITION_FISHER` | `true` | A3. |
| `RUN_BETWEEN_TIMEPOINT_FISHER` | `true` | A6 — auto-skipped. |

**Command built** (per level):

```bash
python3 scripts/stall_sites_consensus_intersection_stats.py \
  --stall-sites raw/stall_sites_${LEVEL}.csv \
  --background  raw/per_group_background_${LEVEL}.csv \
  --groups 'control:control;treatment:treatment' \
  --out-dir …/analysis \
  --headline-condition treatment \
  --within-condition true \
  --between-condition-fisher true \
  --between-timepoint-fisher true
```

**Inputs / Outputs.** In: `raw/` CSVs. Out: `analysis/within_condition_binomial_{aa,codon}.csv` (A1), `analysis/between_condition_fisher_{aa,codon}.csv` (A3).

---

### `analyze_stall_sites_consensus_intersection_fisher_volcano.sh`

**What it does.** Adds R to PATH, sources `_headline_config.sh`, self-locates. Defines a `run_volcano()` helper that skips-with-a-note if the input CSV is absent, then calls it for AA (→ `plots/between_condition_fisher/`) and codon (→ its `codon/` subdir).

**CONFIG.** `INPUT_DIR=…/analysis`, `PLOTS_DIR=…/plots`, `FORMAT=both`, `DPI=300`; label from `_headline_config.sh`.

**Command built** (per level):

```bash
Rscript R_scripts/between_group_volcano.R \
  --input analysis/between_condition_fisher_${LEVEL}.csv \
  --outdir plots/between_condition_fisher[/codon] \
  --level ${LEVEL} \
  --flat-design \
  --comparison-label "treatment vs control" \
  --composite-tag fisher \
  --format both --dpi 300
```

No `--effect-col`/`--x-label` overrides here (unlike the union background-diff launcher): the Fisher CSV's default odds-ratio column and axis are used directly. `--flat-design` → one composite row of A/P/E panels.

**Inputs / Outputs.** In: A3 CSVs. Out: PDF+PNG volcanoes under `plots/between_condition_fisher/`.

---

### `analyze_stall_sites_consensus_intersection_within_condition_volcano.sh`

**What it does.** Byte-for-byte the same launcher as the union tree's within-condition script, pointed at the intersection `analysis/`/`plots/` paths. Adds R to PATH, self-locates, runs `within_condition_volcano.R` for aa then codon with `OPTIONAL_FLAGS=(--flat-design --mega-composite)` plus conditional `--show-ci` / `--y-cap`.

**CONFIG.** `INPUT_DIR=…/analysis`, `OUTPUT_DIR=…/plots/within_condition`, `ENRICHMENT_TYPE=both`, `SHOW_CI=--show-ci`, `FORMAT=both`, `DPI=300`, `Y_CAP=25`.

**Command built** (aa; codon → `…/codon` with `--level codon`):

```bash
Rscript R_scripts/within_condition_volcano.R \
  --input analysis/within_condition_binomial_aa.csv \
  --outdir plots/within_condition \
  --level aa --enrichment-type both --format both --dpi 300 \
  --flat-design --mega-composite --show-ci --y-cap 25
```

**Inputs / Outputs.** In: A1 CSVs. Out: PDF+PNG volcanoes under `plots/within_condition/`.

---

## See also

- [Python entry points](../../../scripts/README.md) — `stall_sites_consensus_intersection{,_stats}.py`.
- [R plotting scripts](../../../R_scripts/README.md) — `between_group_volcano.R`, `within_condition_volcano.R`.
- [mouse organism README](../README.md) — flat design + omitted Wilcoxon launchers.
- [union sibling](../stall_sites_consensus_union/README.md) — differs only in transcript handling (union) vs Fisher (intersection).
- [C. elegans counterpart](../../c_elegans/stall_sites_consensus_intersection/README.md).
- [shell_scripts top level](../../README.md) · [Repository root](../../../README.md).
