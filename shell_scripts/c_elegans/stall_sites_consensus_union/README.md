# stall_sites_consensus_union — Step 2a, union variant

*Consensus stall-site calling where each experimental group keeps its own filtered transcript set, feeding background-aware enrichment stats (A1 / A4 / A7) and their volcano plots.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [c_elegans](../README.md) › stall_sites_consensus_union

This stage is one of the two **consensus** flavours of Step 2a. "Consensus" means replicate stall calls are collapsed into a single agreed set per group (so downstream, replicate == group). "Union" refers to the transcript universe: **each group keeps its own reproducibility-filtered transcript set**, so different conditions can have different backgrounds.

Because the backgrounds differ between conditions, the valid between-group comparison is **background-aware**: each condition is normalized to *its own* background before being compared. That is why this tree's stats are A1 (within-condition binomial), **A4** (between-condition / per-timepoint background-aware diff), and **A7** (between-timepoint background-aware diff) — the Fisher tests belong to the sibling *intersection* stage, whose shared transcript universe makes Fisher fair and the background-diff degenerate.

Everything this stage produces lands under `results/c_elegans/stall_sites_consensus_union/`: `raw/` (base CSVs), `analysis/` (stats CSVs), `plots/` (R plots).

## Contents

| File | Kind | Summary |
|---|---|---|
| `_headline_config.sh` | shared config (sourced) | Fixes `HEADLINE_CONDITION`/`OTHER_CONDITION` and the derived comparison labels; sourced by the stats runner **and** both plot launchers so direction and labels can never drift. |
| `run_stall_sites_consensus_union.sh` | `run_*` (drives Python) | Runs `stall_sites_consensus_union.py`: calls consensus stall sites (union tx) and writes the base `stall_sites_*` / `per_group_background_*` CSVs. |
| `run_stall_sites_consensus_union_stats.sh` | `run_*_stats` (drives stats Python) | Runs `stall_sites_consensus_union_stats.py`: A1 within-condition binomial, A4 background-aware diff, A7 between-timepoint diff. |
| `analyze_stall_sites_consensus_union_background_diff_volcano.sh` | `analyze_*` (drives R) | Volcano plots of the A4 / A7 background-aware CSVs via `between_group_volcano.R`. |
| `analyze_stall_sites_consensus_union_within_condition_volcano.sh` | `analyze_*` (drives R) | Volcano plots of the A1 within-condition binomial CSVs via `within_condition_volcano.R`. |

Every one of these self-locates to the repo root with the `SCRIPT_DIR=…; cd "$SCRIPT_DIR/../../.."` pattern. The two `run_*` scripts activate `ribostall_env`; the two `analyze_*` scripts instead add R to `PATH`.

---

### `_headline_config.sh`

**What it does.** A tiny sourced-only fragment (no shebang execution intended). It sets the two condition names and derives the labels the rest of the stage uses, so the statistical *direction* and every plot *label* come from one file and cannot drift apart.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `HEADLINE_CONDITION="BWM"` | Numerator / direction reference. A positive `delta_log2_enrichment` means enriched vs background in BWM. Must be the part before the first underscore of an `EXP_GROUPS` label. |
| `OTHER_CONDITION="control"` | The denominator condition. |
| `COMPARISON_LABEL` *(derived)* | `"BWM vs control"` — passed as `--comparison-label`. |
| `COMPARISON_TAG` *(derived)* | `"BWM_vs_control"` — file-prefix / grouping-column value. |
| `X_LABEL_RATIO` *(derived)* | `"Log2 Enrichment Ratio (BWM / control)"` — the background-diff x-axis label. |

**Related.** Consumed by the three scripts below; no Python/R of its own.

---

### `run_stall_sites_consensus_union.sh`

**What it does.** Activates the env, self-locates to repo root, finds the coverage pickle via `ls "$RIBO_DIR"/*_coverage.pkl.gz | head -1`, echoes a parameter banner, then builds a `CMD=(python3 …)` array and runs it as `"${CMD[@]}"`.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `RIBO_DIR`, `RIBO_FILE` | Coverage-pickle directory and the exact C. elegans `.ribo`. |
| `EXP_GROUPS` | The six `(condition, timepoint)` cells, `rep2`/`rep3` each (see the [organism README](../README.md)). |
| `TX_THRESHOLD=0.5` | Per-transcript coverage floor for the reproducibility filter (v2: lowered from 1.0 to retain low-coverage BWM groups). |
| `TX_MIN_REPS_PER_GROUP` | Per-group replicate support for the transcript filter; must name **every** declared group (no global fallback). |
| `MIN_Z=1.0` | Z-score threshold for calling a position a stall. |
| `MIN_READS=5` | Minimum reads at a position to be eligible (v2: raised from 2 to clear the noise floor). |
| `TRIM_START=20`, `TRIM_STOP=10` | Codons trimmed from the CDS start / stop before analysis. |
| `PSEUDOCOUNT=0.5` | Added before ratio/enrichment math. |
| `STALL_MIN_REPS_PER_GROUP` | Per-group consensus support — how many replicates must agree for a site to enter the consensus set. Must name every group. |
| `TOL=0` | Codon tolerance for matching a stall across replicates. |
| `MIN_SEP=0` | Min codon separation to collapse nearby sites; inert under the default `keep_both` conflict resolution, so 0. |
| `REFERENCE_FILE` | The APPRIS C. elegans FASTA, required for E/P/A annotation. |
| `BASIS="P"`, `PSITE_OFFSET=0` | E/P/A register and codon offset applied before deriving E/P/A. |
| `DROP_STOP_CODONS="True"` | Drop stall windows whose E/P/A site is a stop codon. |
| `OUT_DIR="results/c_elegans/stall_sites_consensus_union/raw"` | Where the base CSVs are written. |

**Command built.** `scripts/stall_sites_consensus_union.py`, passing `--pickle --ribo --reference --groups --tx_threshold --tx_min_reps_per_group --min_z --min_reads --trim-start --trim-stop --pseudocount --stall_min_reps_per_group --tol --min_sep --basis --psite-offset --drop-stop-codons --out-dir`.

**Inputs / Outputs.**

- **Input:** the coverage pickle + reference FASTA.
- **Output (in `raw/`):** `stall_sites_codon.csv`, `stall_sites_aa.csv`, `per_group_background_codon.csv`, `per_group_background_aa.csv`.

**Related.** `scripts/stall_sites_consensus_union.py` — see [scripts/README.md](../../../scripts/README.md).

---

### `run_stall_sites_consensus_union_stats.sh`

**What it does.** Sources `_headline_config.sh`, self-locates, and loops `for LEVEL in aa codon`, running `stall_sites_consensus_union_stats.py` once per level. It builds `HEADLINE_FLAG` and `TIMEPOINTS_FLAG` as arrays that are only populated when the corresponding config var is non-empty, so an empty value cleanly omits the flag.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `EXP_GROUPS` | Consensus form: `replicate == group`, so each cell is `control_day_0:control_day_0;…`. |
| `TIMEPOINTS='day_0,day_5,day_10'` | Chronological timepoints; when set, A4 slices per-timepoint and A7 runs across day-pairs. Empty → flat design, A7 skipped. |
| `RAW_DIR`, `OUT_DIR` | Input `raw/` and output `analysis/` directories. |
| `RUN_WITHIN_CONDITION=true` | A1 — within-condition binomial. |
| `RUN_BETWEEN_CONDITION_BACKGROUND_DIFF=true` | A4 — between-condition (or per-timepoint) background-aware diff. |
| `RUN_BETWEEN_TIMEPOINT_BACKGROUND_DIFF=true` | A7 — between-timepoint background-aware diff (timepoint mode only). |
| (sourced) `HEADLINE_CONDITION` | Passed as `--headline-condition` when set; fixes the BWM-vs-control direction. |

**Command built (per level).**

```bash
python3 scripts/stall_sites_consensus_union_stats.py \
  --stall-sites "$RAW_DIR/stall_sites_${LEVEL}.csv" \
  --background  "$RAW_DIR/per_group_background_${LEVEL}.csv" \
  --groups "$EXP_GROUPS" --out-dir "$OUT_DIR" \
  "${HEADLINE_FLAG[@]}" "${TIMEPOINTS_FLAG[@]}" \
  --within-condition "${RUN_WITHIN_CONDITION:-true}" \
  --between-condition-background-diff "${RUN_BETWEEN_CONDITION_BACKGROUND_DIFF:-true}" \
  --between-timepoint-background-diff "${RUN_BETWEEN_TIMEPOINT_BACKGROUND_DIFF:-true}"
```

Each `RUN_*` value is passed straight through as a `true`/`false` string; the Python script skips any analysis given `false`.

**Inputs / Outputs.**

- **Input:** the four `raw/` CSVs above.
- **Output (in `analysis/`):** `within_condition_binomial_{aa,codon}.csv` (A1), `per_timepoint_background_diff_{aa,codon}.csv` (A4, timepoint mode), `between_timepoint_background_diff_{aa,codon}.csv` (A7).

**Related.** `scripts/stall_sites_consensus_union_stats.py` — see [scripts/README.md](../../../scripts/README.md).

---

### `analyze_stall_sites_consensus_union_background_diff_volcano.sh`

**What it does.** The timepoint-mode volcano launcher for the background-aware CSVs. Adds R to `PATH`, sources `_headline_config.sh`, self-locates, and runs `between_group_volcano.R` twice per block (aa + codon) over the A4 and A7 CSVs. Each input is existence-guarded (missing file → error/exit).

**CONFIG.** `INPUT_DIR` (`analysis/`), `PLOTS_DIR` (`plots/`), `FORMAT="both"`, `DPI=300`; direction labels come from the sourced `_headline_config.sh`.

**Command built.** `R_scripts/between_group_volcano.R`, driven via its generalized options because the effect column is an already-log2 enrichment *ratio*, not an odds ratio:

- **A4** (`per_timepoint_background_diff_{level}.csv`): `--group-col timepoint --comparison-label "$COMPARISON_LABEL" --effect-col delta_log2_enrichment --effect-is-log2 --x-label "$X_LABEL_RATIO" --title-test-label "Background-Aware Enrichment" --composite-tag binomial`.
- **A7** (`between_timepoint_background_diff_{level}.csv`): same options but `--group-col comparison`, a fixed `--comparison-label "Later vs Earlier Timepoint"` and `--x-label "Log2 Enrichment Ratio (later / earlier)"` (direction is later-vs-earlier, independent of the headline).

**Inputs / Outputs.** A4/A7 CSVs from `analysis/` → PNG/PDF volcanoes under `plots/per_timepoint_background_diff/` and `plots/between_timepoint_background_diff/` (codon plots in a `codon/` subdir).

**Related.** `R_scripts/between_group_volcano.R` — see [R_scripts/README.md](../../../R_scripts/README.md).

---

### `analyze_stall_sites_consensus_union_within_condition_volcano.sh`

**What it does.** Volcano launcher for the A1 within-condition binomial CSVs. Adds R to `PATH`, self-locates, builds an `OPTIONAL_FLAGS` array, and runs `within_condition_volcano.R` once for aa and once for codon.

**CONFIG.** `INPUT_DIR` (`analysis/`), `OUTPUT_DIR` (`plots/within_condition`), `ENRICHMENT_TYPE="both"` (unweighted + weighted), `SHOW_CI="--show-ci"`, `FORMAT="both"`, `DPI=300`, `Y_CAP=25` (caps `-log10(p_adj)`).

**Command built.** `R_scripts/within_condition_volcano.R` with `--input within_condition_binomial_{level}.csv --level {aa,codon} --enrichment-type "$ENRICHMENT_TYPE" --format --dpi` plus `OPTIONAL_FLAGS=(--mega-composite [--show-ci] [--y-cap 25])`. Crucially it does **not** pass `--flat-design`: the stats ran with `--timepoints`, so the CSV spans condition × timepoint and the R script builds the by-condition and by-day composites.

**Inputs / Outputs.** `within_condition_binomial_{aa,codon}.csv` → volcanoes under `plots/within_condition/` (codon in `codon/`).

**Related.** `R_scripts/within_condition_volcano.R` — see [R_scripts/README.md](../../../R_scripts/README.md).

## See also

- [stall_sites_consensus_intersection](../stall_sites_consensus_intersection/README.md) — the sibling variant with a shared transcript universe and Fisher stats.
- [c_elegans stage index](../README.md) — pipeline order and experimental design.
- [scripts/README.md](../../../scripts/README.md) · [R_scripts/README.md](../../../R_scripts/README.md) — underlying Python and R.
- [shell_scripts/README.md](../../README.md) · [ribostall root README](../../../README.md).
