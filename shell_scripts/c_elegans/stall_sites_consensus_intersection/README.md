# stall_sites_consensus_intersection — Step 2a, intersection variant

*Consensus stall-site calling restricted to the transcripts that pass filtering in every group — one shared transcript universe — feeding Fisher enrichment stats (A1 / A3 / A6) and their volcano plots.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [c_elegans](../README.md) › stall_sites_consensus_intersection

This is the second **consensus** flavour of Step 2a. As with the union variant, replicate stall calls are collapsed into one agreed set per group (replicate == group downstream). The difference is the transcript universe: **every group is restricted to the transcripts that pass reproducibility filtering in *all* groups**. All conditions therefore share one transcript universe.

With a single shared universe, raw stall-site shares are apples-to-apples across conditions, so **Fisher's exact test** is the valid between-group comparison. Conversely, the per-group backgrounds become identical, which makes the background-aware diff degenerate — those tests (A4/A7) live in the sibling *union* stage. So this tree's stats are A1 (within-condition binomial, still using the per-group background), **A3** (between-condition / per-timepoint Fisher), and **A6** (between-timepoint Fisher within each condition).

Output lands under `results/c_elegans/stall_sites_consensus_intersection/`: `raw/`, `analysis/`, `plots/`.

## Contents

| File | Kind | Summary |
|---|---|---|
| `_headline_config.sh` | shared config (sourced) | Fixes `HEADLINE_CONDITION`/`OTHER_CONDITION` + derived labels; sourced by the stats runner **and** both plot launchers so direction and labels can't drift. |
| `run_stall_sites_consensus_intersection.sh` | `run_*` (drives Python) | Runs `stall_sites_consensus_intersection.py`: consensus stall calling on the shared tx universe; writes base `stall_sites_*` / `per_group_background_*` CSVs. |
| `run_stall_sites_consensus_intersection_stats.sh` | `run_*_stats` (drives stats Python) | Runs `stall_sites_consensus_intersection_stats.py`: A1 within-condition binomial, A3 Fisher, A6 between-timepoint Fisher. |
| `analyze_stall_sites_consensus_intersection_fisher_volcano.sh` | `analyze_*` (drives R) | Volcano plots of the A3 / A6 Fisher CSVs via `between_group_volcano.R`. |
| `analyze_stall_sites_consensus_intersection_within_condition_volcano.sh` | `analyze_*` (drives R) | Volcano plots of the A1 within-condition binomial CSVs via `within_condition_volcano.R`. |

The two `run_*` scripts activate `ribostall_env`; the two `analyze_*` scripts add R to `PATH`. All self-locate with `cd "$SCRIPT_DIR/../../.."`.

---

### `_headline_config.sh`

**What it does.** Sourced-only fragment fixing the comparison direction and its labels so the Fisher numerator and the plot labels come from one place.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `HEADLINE_CONDITION="BWM"` | Numerator; a positive log2 odds ratio means enriched in BWM. |
| `OTHER_CONDITION="control"` | Denominator. |
| `COMPARISON_LABEL` *(derived)* | `"BWM vs control"` — Fisher `--comparison-label`. |
| `COMPARISON_TAG` *(derived)* | `"BWM_vs_control"` — file prefix / grouping-column value. |
| `X_LABEL_RATIO` *(derived)* | `"Log2 Enrichment Ratio (BWM / control)"` (present for parity with the union config). |

**Related.** Consumed by the three scripts below.

---

### `run_stall_sites_consensus_intersection.sh`

**What it does.** Same shape as the union runner: activate env, self-locate, find the coverage pickle via `ls "$RIBO_DIR"/*_coverage.pkl.gz | head -1`, echo a banner, build a `CMD=(python3 …)` array, run it.

**CONFIG.** Identical variable set to the union runner — `RIBO_DIR`, `RIBO_FILE`, `EXP_GROUPS` (the six `(condition, timepoint)` cells, `rep2`/`rep3`), `TX_THRESHOLD=0.5`, `TX_MIN_REPS_PER_GROUP`, `MIN_Z=1.0`, `MIN_READS=5`, `TRIM_START=20`, `TRIM_STOP=10`, `PSEUDOCOUNT=0.5`, `STALL_MIN_REPS_PER_GROUP`, `TOL=0`, `MIN_SEP=0`, `REFERENCE_FILE`, `BASIS="P"`, `PSITE_OFFSET=0`, `DROP_STOP_CODONS="True"` — with `OUT_DIR="results/c_elegans/stall_sites_consensus_intersection/raw"`. (See the [union README](../stall_sites_consensus_union/README.md) for what each means; the only difference between the two runners is which Python script they call and thus the tx-universe policy.)

**Command built.** `scripts/stall_sites_consensus_intersection.py`, passing the same flag set as the union runner: `--pickle --ribo --reference --groups --tx_threshold --tx_min_reps_per_group --min_z --min_reads --trim-start --trim-stop --pseudocount --stall_min_reps_per_group --tol --min_sep --basis --psite-offset --drop-stop-codons --out-dir`.

**Inputs / Outputs.** Coverage pickle + reference → `stall_sites_{codon,aa}.csv` and `per_group_background_{codon,aa}.csv` in `raw/`.

**Related.** `scripts/stall_sites_consensus_intersection.py` — see [scripts/README.md](../../../scripts/README.md).

---

### `run_stall_sites_consensus_intersection_stats.sh`

**What it does.** Sources `_headline_config.sh`, self-locates, and loops `for LEVEL in aa codon` running `stall_sites_consensus_intersection_stats.py`. `HEADLINE_FLAG` and `TIMEPOINTS_FLAG` are populated only when their config vars are non-empty.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `EXP_GROUPS` | Consensus form `control_day_0:control_day_0;…` (replicate == group). |
| `TIMEPOINTS='day_0,day_5,day_10'` | When set, A3 slices per-timepoint and A6 runs across day-pairs; empty → flat design (A6 skipped). |
| `RAW_DIR`, `OUT_DIR` | Input `raw/`, output `analysis/`. |
| `RUN_WITHIN_CONDITION=true` | A1 — within-condition binomial. |
| `RUN_BETWEEN_CONDITION_FISHER=true` | A3 — between-condition (or per-timepoint) Fisher. |
| `RUN_BETWEEN_TIMEPOINT_FISHER=true` | A6 — between-timepoint Fisher within condition (timepoint mode only). |
| (sourced) `HEADLINE_CONDITION` | `--headline-condition` when set; BWM numerator. |

**Command built (per level).**

```bash
python3 scripts/stall_sites_consensus_intersection_stats.py \
  --stall-sites "$RAW_DIR/stall_sites_${LEVEL}.csv" \
  --background  "$RAW_DIR/per_group_background_${LEVEL}.csv" \
  --groups "$EXP_GROUPS" --out-dir "$OUT_DIR" \
  "${HEADLINE_FLAG[@]}" "${TIMEPOINTS_FLAG[@]}" \
  --within-condition "${RUN_WITHIN_CONDITION:-true}" \
  --between-condition-fisher "${RUN_BETWEEN_CONDITION_FISHER:-true}" \
  --between-timepoint-fisher "${RUN_BETWEEN_TIMEPOINT_FISHER:-true}"
```

**Inputs / Outputs.**

- **Input:** the four `raw/` CSVs.
- **Output (in `analysis/`):** `within_condition_binomial_{aa,codon}.csv` (A1), `per_timepoint_fisher_{aa,codon}.csv` (A3, timepoint mode), `timepoint_fisher_within_condition_{d10_vs_d0,d10_vs_d5,d5_vs_d0}_{aa,codon}.csv` (A6).

**Related.** `scripts/stall_sites_consensus_intersection_stats.py` — see [scripts/README.md](../../../scripts/README.md).

---

### `analyze_stall_sites_consensus_intersection_fisher_volcano.sh`

**What it does.** Adds R to `PATH`, sources `_headline_config.sh`, self-locates, and defines a `run_volcano()` helper that runs `between_group_volcano.R` on one CSV, **skipping with a note** if the input is absent (so a flat run, which omits the per-timepoint / within-condition-timepoint CSVs, degrades gracefully). It then calls the helper for A3 (aa + codon) and, in a loop over `d10_vs_d0 d10_vs_d5 d5_vs_d0`, for A6.

**CONFIG.** `INPUT_DIR` (`analysis/`), `PLOTS_DIR` (`plots/`), `FORMAT="both"`, `DPI=300`; labels from the sourced `_headline_config.sh`.

**Command built.** `R_scripts/between_group_volcano.R` with `--composite-tag fisher`:

- **A3** (`per_timepoint_fisher_{level}.csv`): `--group-col timepoint --comparison-label "$COMPARISON_LABEL"` → `plots/per_timepoint_fisher/`.
- **A6** (`timepoint_fisher_within_condition_${comparison}_{level}.csv`): `--group-col condition --comparison-label "$pretty"` (where `$pretty` is `d10_vs_d0` → `Day 10 vs Day 0` via `sed`) → `plots/within_condition_timepoint_fisher/${comparison}/`.

The default effect column is the Fisher `odds_ratio` (no `--effect-col` / `--effect-is-log2` overrides here, unlike the union background-diff launcher).

**Inputs / Outputs.** A3/A6 CSVs → PNG/PDF volcanoes under `plots/per_timepoint_fisher/` and `plots/within_condition_timepoint_fisher/` (codon in `codon/` subdirs).

**Related.** `R_scripts/between_group_volcano.R` — see [R_scripts/README.md](../../../R_scripts/README.md).

---

### `analyze_stall_sites_consensus_intersection_within_condition_volcano.sh`

**What it does.** Identical in shape to the union stage's within-condition launcher. Adds R to `PATH`, self-locates, builds `OPTIONAL_FLAGS`, and runs `within_condition_volcano.R` for aa and codon.

**CONFIG.** `INPUT_DIR` (`analysis/`), `OUTPUT_DIR` (`plots/within_condition`), `ENRICHMENT_TYPE="both"`, `SHOW_CI="--show-ci"`, `FORMAT="both"`, `DPI=300`, `Y_CAP=25`.

**Command built.** `R_scripts/within_condition_volcano.R --input within_condition_binomial_{level}.csv --level {aa,codon} --enrichment-type "$ENRICHMENT_TYPE" --format --dpi` plus `OPTIONAL_FLAGS=(--mega-composite [--show-ci] [--y-cap 25])`. No `--flat-design`, because the A1 CSV spans condition × timepoint under the timepoint stats run.

**Inputs / Outputs.** `within_condition_binomial_{aa,codon}.csv` → volcanoes under `plots/within_condition/`.

**Related.** `R_scripts/within_condition_volcano.R` — see [R_scripts/README.md](../../../R_scripts/README.md).

## See also

- [stall_sites_consensus_union](../stall_sites_consensus_union/README.md) — the sibling variant with per-group tx sets and background-aware stats.
- [c_elegans stage index](../README.md) — pipeline order and experimental design.
- [scripts/README.md](../../../scripts/README.md) · [R_scripts/README.md](../../../R_scripts/README.md).
- [shell_scripts/README.md](../../README.md) · [ribostall root README](../../../README.md).
