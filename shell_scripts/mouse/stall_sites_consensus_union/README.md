# stall_sites_consensus_union — Step 2a (union): consensus stall calling + background-aware stats (mouse)

*Wraps the union consensus caller and its A1/A4 stats, then drives the two flat-design volcano launchers — for the mouse `control`-vs-`treatment` design where each group keeps its own filtered transcript set.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [mouse](../README.md) › stall_sites_consensus_union

---

## What this stage is

This is one of the two **consensus** variants of Step 2a. The **union** caller keeps, for each group, its *own* reproducibility-filtered transcript set. Because the two groups therefore have *different* backgrounds, the valid between-condition test is the **background-aware diff** (each condition normalized to its own background), not Fisher — Fisher lives in the sibling intersection tree.

Outputs land under `results/mouse/stall_sites_consensus_union/` in the standard three-subdir layout:
- `raw/` — `stall_sites_{codon,aa}.csv` + `per_group_background_{codon,aa}.csv` (the caller's base CSVs).
- `analysis/` — stats CSVs: `within_condition_binomial_{aa,codon}.csv` (A1) and `between_condition_background_diff_{aa,codon}.csv` (A4).
- `plots/` — R volcano plots.

**Flat-design note.** Mouse has no timepoints, so `TIMEPOINTS=''` in the stats runner: A4 emits a single flat between-condition CSV and A7 (between-timepoint diff) is skipped automatically. The plot launchers pass `--flat-design`, so each volcano is one composite row of A/P/E site panels rather than a day grid.

## Contents

| File | Role | Summary |
|---|---|---|
| `_headline_config.sh` | shared config | Sets `HEADLINE_CONDITION=treatment`, `OTHER_CONDITION=control`; derives comparison label / x-axis label. Sourced by the stats runner *and* both launchers so direction can't drift. |
| `run_stall_sites_consensus_union.sh` | `run_*` | Calls `stall_sites_consensus_union.py` → `raw/` base CSVs. |
| `run_stall_sites_consensus_union_stats.sh` | `run_*_stats` | Calls `stall_sites_consensus_union_stats.py` → A1 + A4 in `analysis/`. |
| `analyze_stall_sites_consensus_union_background_diff_volcano.sh` | `analyze_*` | Drives `between_group_volcano.R` on the A4 CSVs (`--flat-design`). |
| `analyze_stall_sites_consensus_union_within_condition_volcano.sh` | `analyze_*` | Drives `within_condition_volcano.R` on the A1 CSVs (`--flat-design`). |

---

### `_headline_config.sh`

**What it does.** A tiny sourced config that fixes the comparison direction in one place. It sets only two editable values and derives the rest:

```bash
HEADLINE_CONDITION="treatment"   # numerator / positive-effect reference
OTHER_CONDITION="control"        # denominator
COMPARISON_LABEL="treatment vs control"
COMPARISON_TAG="treatment_vs_control"
X_LABEL_RATIO="Log2 Enrichment Ratio (treatment / control)"
```

A positive `delta_log2_enrichment` means "enriched vs background in `treatment`". Because the stats runner passes `HEADLINE_CONDITION` to `--headline-condition` **and** the launchers read `COMPARISON_LABEL` / `X_LABEL_RATIO` from the same file, the stats numerator and the plot labels cannot disagree.

**Related.** Sourced by all three runnable scripts in this folder.

---

### `run_stall_sites_consensus_union.sh`

**What it does.** Self-locates to repo root, activates `ribostall_env`, derives the coverage pickle from `RIBO_FILE`, guards its existence, then runs `stall_sites_consensus_union.py` via a Bash array (`CMD=(...)`; `"${CMD[@]}"`).

**CONFIG.**

| Variable | Value | Meaning |
|---|---|---|
| `RIBO_DIR` / `RIBO_FILE` | `./all_ribo_file` / `…/mouse_all.ribo` | Ribo file (pickle derived from it). |
| `EXP_GROUPS` | `control:AA_3,AA_4;treatment:Ch_WAA2` | Flat 2-vs-1 design. |
| `TX_THRESHOLD` | `0.5` | Transcript-filter coverage threshold (v2 set). |
| `TX_MIN_REPS_PER_GROUP` | `control:2;treatment:1` | Per-group tx-filter support; names every group, no global fallback. |
| `MIN_Z` / `MIN_READS` | `1.0` / `5` | Stall-calling z and read floor (v2 set). |
| `TRIM_START` / `TRIM_STOP` | `20` / `10` | CDS trim (codons). |
| `PSEUDOCOUNT` | `0.5` | Enrichment pseudocount. |
| `STALL_MIN_REPS_PER_GROUP` | `control:2;treatment:1` | Per-group consensus support: control needs both reps, treatment its one. |
| `TOL` / `MIN_SEP` | `0` / `0` | Consensus position tolerance / minimum separation. |
| `REFERENCE_FILE` | `./reference/appris_mouse_v2_selected.fa.gz` | For E/P/A annotation. |
| `BASIS` / `PSITE_OFFSET` | `P` / `0` | E/P/A register + codon offset. |
| `DROP_STOP_CODONS` | `True` | Drop windows whose E/P/A hits TAA/TAG/TGA. |
| `OUT_DIR` | `results/mouse/stall_sites_consensus_union/raw` | Base-CSV output. |

**Command built.**

```bash
python3 scripts/stall_sites_consensus_union.py \
  --pickle all_ribo_file/mouse_all_coverage.pkl.gz --ribo …/mouse_all.ribo \
  --reference reference/appris_mouse_v2_selected.fa.gz \
  --groups 'control:AA_3,AA_4;treatment:Ch_WAA2' \
  --tx_threshold 0.5 --tx_min_reps_per_group 'control:2;treatment:1' \
  --min_z 1.0 --min_reads 5 --trim-start 20 --trim-stop 10 --pseudocount 0.5 \
  --stall_min_reps_per_group 'control:2;treatment:1' --tol 0 --min_sep 0 \
  --basis P --psite-offset 0 --drop-stop-codons True \
  --out-dir results/mouse/stall_sites_consensus_union/raw
```

**Inputs / Outputs.** In: coverage pickle + reference. Out: `raw/stall_sites_{codon,aa}.csv`, `raw/per_group_background_{codon,aa}.csv`.

**Related.** Feeds `run_stall_sites_consensus_union_stats.sh`.

---

### `run_stall_sites_consensus_union_stats.sh`

**What it does.** Ribopy-free. Sources `_headline_config.sh`, self-locates to repo root, and loops `for LEVEL in aa codon` running `stall_sites_consensus_union_stats.py` on the `raw/` CSVs. It builds two optional flag arrays: `HEADLINE_FLAG` (added only when `HEADLINE_CONDITION` is non-empty) and `TIMEPOINTS_FLAG` (added only when `TIMEPOINTS` is non-empty — here it is empty, so it is omitted).

**CONFIG.**

| Variable | Value | Meaning |
|---|---|---|
| `EXP_GROUPS` | `control:control;treatment:treatment` | Consensus set replicate == group, so each "rep" is the group name. |
| `TIMEPOINTS` | `''` (empty) | Flat design → `--timepoints` omitted, A7 skipped. |
| `RAW_DIR` / `OUT_DIR` | `…/raw` / `…/analysis` | Input / output. |
| `RUN_WITHIN_CONDITION` | `true` | A1 within-condition binomial. |
| `RUN_BETWEEN_CONDITION_BACKGROUND_DIFF` | `true` | A4 background-aware diff. |
| `RUN_BETWEEN_TIMEPOINT_BACKGROUND_DIFF` | `true` | A7 — auto-skipped (no timepoints). |
| `HEADLINE_CONDITION` | `treatment` (from config) | Passed to `--headline-condition`. |

**Command built** (per level):

```bash
python3 scripts/stall_sites_consensus_union_stats.py \
  --stall-sites raw/stall_sites_${LEVEL}.csv \
  --background  raw/per_group_background_${LEVEL}.csv \
  --groups 'control:control;treatment:treatment' \
  --out-dir …/analysis \
  --headline-condition treatment \
  --within-condition true \
  --between-condition-background-diff true \
  --between-timepoint-background-diff true
```

(`--timepoints` is omitted because `TIMEPOINTS` is empty, which is what makes A7 a no-op.)

**Inputs / Outputs.** In: `raw/` CSVs. Out: `analysis/within_condition_binomial_{aa,codon}.csv` (A1), `analysis/between_condition_background_diff_{aa,codon}.csv` (A4).

**Related.** Its A4 output feeds the background-diff launcher; its A1 output feeds the within-condition launcher.

---

### `analyze_stall_sites_consensus_union_background_diff_volcano.sh`

**What it does.** Adds R to PATH, sources `_headline_config.sh`, self-locates to repo root, and loops `for LEVEL in aa codon` calling `between_group_volcano.R` on the A4 CSVs. AA plots go to `plots/between_condition_background_diff/`, codon to its `codon/` subdir.

**CONFIG.** `INPUT_DIR=…/analysis`, `PLOTS_DIR=…/plots`, `FORMAT=both`, `DPI=300`; direction from `_headline_config.sh`.

**Command built** (per level, when the CSV exists):

```bash
Rscript R_scripts/between_group_volcano.R \
  --input analysis/between_condition_background_diff_${LEVEL}.csv \
  --outdir plots/between_condition_background_diff[/codon] \
  --level ${LEVEL} \
  --flat-design \
  --comparison-label "treatment vs control" \
  --effect-col delta_log2_enrichment --effect-is-log2 \
  --x-label "Log2 Enrichment Ratio (treatment / control)" \
  --title-test-label "Background-Aware Enrichment" \
  --composite-tag binomial \
  --format both --dpi 300
```

The effect column is `delta_log2_enrichment` (an enrichment ratio, already log2 — hence `--effect-is-log2` and the honest `--x-label`, not "Odds Ratio"). `--flat-design` tells the R script the CSV carries only `site`, so it renders one composite row of A/P/E panels.

**Inputs / Outputs.** In: A4 CSVs. Out: PDF+PNG volcanoes under `plots/between_condition_background_diff/`.

**Related.** A7 (between-timepoint) has no launcher here — mouse has no timepoints.

---

### `analyze_stall_sites_consensus_union_within_condition_volcano.sh`

**What it does.** Adds R to PATH, self-locates to repo root, and runs `within_condition_volcano.R` twice (aa then codon) on the A1 CSVs. It assembles `OPTIONAL_FLAGS=(--flat-design --mega-composite)` and conditionally appends `--show-ci` and `--y-cap`.

**CONFIG.** `INPUT_DIR=…/analysis`, `OUTPUT_DIR=…/plots/within_condition`, `ENRICHMENT_TYPE=both`, `SHOW_CI=--show-ci`, `FORMAT=both`, `DPI=300`, `Y_CAP=25`.

**Command built** (aa shown; codon identical to `…/codon` with `--level codon`):

```bash
Rscript R_scripts/within_condition_volcano.R \
  --input analysis/within_condition_binomial_aa.csv \
  --outdir plots/within_condition \
  --level aa --enrichment-type both --format both --dpi 300 \
  --flat-design --mega-composite --show-ci --y-cap 25
```

`--flat-design` builds the flat per-group layout (rows = group, cols = sites) instead of by-condition/by-day grids; `--mega-composite` also emits the all-groups grid.

**Inputs / Outputs.** In: A1 CSVs. Out: PDF+PNG volcanoes under `plots/within_condition/`.

**Related.** Reads the A1 output of the stats runner.

---

## See also

- [Python entry points](../../../scripts/README.md) — `stall_sites_consensus_union{,_stats}.py`.
- [R plotting scripts](../../../R_scripts/README.md) — `between_group_volcano.R`, `within_condition_volcano.R`.
- [mouse organism README](../README.md) — flat design + why the Wilcoxon launchers are omitted repo-wide.
- [C. elegans counterpart](../../c_elegans/stall_sites_consensus_union/README.md) — the timepoint version.
- [shell_scripts top level](../../README.md) · [Repository root](../../../README.md).
