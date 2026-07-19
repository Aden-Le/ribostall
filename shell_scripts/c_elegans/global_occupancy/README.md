# global_occupancy — Step 3: transcriptome-wide codon/AA occupancy

*Compute per-experiment codon and amino-acid ribosome occupancy across the whole transcriptome, run the full stats suite over all three ribosomal sites, and plot the merged E/P/A results.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [c_elegans](../README.md) › global_occupancy

This is **Step 3**. Rather than calling discrete stall sites, it measures *global* occupancy — how much ribosome density each codon or amino acid carries across the transcriptome, at each of the three ribosomal sites (**E**, **P**, **A**). Every condition is normalized to a single shared transcriptome background (unlike the per-group backgrounds of the stall-site stages), so there is no per-condition enrichment-ratio axis here; direction is headline-driven only for the Wilcoxon (A2) and per-timepoint Fisher (A3) tests.

The stats runner processes all three sites for one level per invocation and writes a **merged** analysis tree: the E/P/A per-site frames are concatenated with a `site` column into a single CSV per analysis (the old `merge_global_occupancy_analysis.py` step is folded in). The per-site frames are an internal intermediate and are **not** exported — only the merged CSVs under `analysis/` are.

Output lands under `results/c_elegans/global_occupancy/`: `raw/` (base occupancy CSVs), `analysis/` (merged E/P/A stats CSVs), `plots/`.

## Contents

| File | Kind | Summary |
|---|---|---|
| `_headline_config.sh` | shared config (sourced) | Fixes `HEADLINE_CONDITION`/`OTHER_CONDITION` + derived labels; sourced by the stats runner **and** the Fisher/Wilcoxon plot launchers so direction and labels can't drift. |
| `run_global_codon_occ.sh` | `run_*` (drives Python) | Runs `global_codon_occ.py`: computes codon/AA occupancy tables; writes base CSVs to `raw/`. |
| `run_global_codon_occ_stats.sh` | `run_*_stats` (drives stats Python) | Runs `global_codon_occ_stats.py` per level across E/P/A; writes merged stats CSVs to `analysis/`. |
| `analyze_global_occupancy_fisher_volcano.sh` | `analyze_*` (drives R) | Volcano plots of the A3 / A6 Fisher CSVs via `between_group_volcano.R`. |
| `analyze_global_occupancy_wilcoxon.sh` | `analyze_*` (drives R) | Bar plots of the A2 / A5 Wilcoxon CSVs via `between_group_barplot.R`. |
| `analyze_global_occupancy_within_condition.sh` | `analyze_*` (drives R) | Volcano plots of the A1 within-condition binomial CSVs via `within_condition_volcano.R`. |

The two `run_*` scripts activate `ribostall_env`; the three `analyze_*` scripts add R to `PATH`. All self-locate with `cd "$SCRIPT_DIR/../../.."`.

---

### `_headline_config.sh`

**What it does.** Sourced-only fragment fixing direction for the between-condition tests. Note it defines **no** `X_LABEL_RATIO`, because global occupancy uses one shared transcriptome background and has no enrichment-ratio x-axis.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `HEADLINE_CONDITION="BWM"` | Numerator; positive `log2_FC` / log2 odds ratio means enriched in BWM. |
| `OTHER_CONDITION="control"` | Denominator. |
| `COMPARISON_LABEL` *(derived)* | `"BWM vs control"` — per-timepoint Fisher `--comparison-label`. |
| `COMPARISON_TAG` *(derived)* | `"BWM_vs_control"` — between-condition Wilcoxon `--comparison` value. |

**Related.** Consumed by the stats runner and the Fisher / Wilcoxon plot launchers.

---

### `run_global_codon_occ.sh`

**What it does.** Activates env, self-locates, finds the coverage pickle via `ls "$RIBO_DIR"/*_coverage.pkl.gz | head -1`, echoes a banner, builds a `CMD=(python3 …)` array and runs it. Prints a reminder to run the stats script afterwards.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `RIBO_DIR`, `RIBO_FILE` | Coverage-pickle directory and the exact C. elegans `.ribo`. |
| `EXP_GROUPS` | The six `(condition, timepoint)` cells (`rep2`/`rep3`), used to filter the coverage dict to declared replicates. |
| `TRIM_START=20`, `TRIM_STOP=10` | Codons trimmed from CDS start / stop before computing occupancy. |
| `DROP_STOP_CODONS="True"` | Exclude stop codons from background, totals, rates, proportions, and rpm. |
| `REFERENCE_FILE` | APPRIS C. elegans FASTA. |
| `OUT_DIR="./results/c_elegans/global_occupancy"` | Output root; base CSVs go to `$OUT_DIR/raw/`. **Must match** the stats runner's `OUT_DIR`. |

**Command built.** `scripts/global_codon_occ.py`, passing `--pickle --ribo --reference --groups --trim-start --trim-stop --drop-stop-codons --out-dir`.

**Inputs / Outputs.** Coverage pickle + reference → base codon/AA occupancy CSVs in `$OUT_DIR/raw/`.

**Related.** `scripts/global_codon_occ.py` — see [scripts/README.md](../../../scripts/README.md).

---

### `run_global_codon_occ_stats.sh`

**What it does.** Sources `_headline_config.sh`, self-locates, and loops `for level in "${LEVELS[@]}"` (codon, aa), running `global_codon_occ_stats.py` once per level with **all three sites passed together** (`--sites E P A`). The Python script computes each per-site frame internally and writes the concatenated (merged) result to `analysis/`. `HEADLINE_FLAG` and `TIMEPOINTS_FLAG` are populated only when their config vars are non-empty.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `EXP_GROUPS` | The six per-replicate cells. |
| `TIMEPOINTS='day_0,day_5,day_10'` | Chronological; drives per-timepoint Fisher (A3) and between-timepoint pairs (A5/A6). |
| `OUT_DIR="./results/c_elegans/global_occupancy"` | Reads `$OUT_DIR/raw/`, writes `$OUT_DIR/analysis/`. Must match the compute runner. |
| `SITES=(E P A)` | Ribosomal sites processed together and merged into the `site` column. |
| `LEVELS=(codon aa)` | Occupancy levels looped over. |
| `RUN_WITHIN_CONDITION=true` | A1 — within-condition binomial. |
| `RUN_BETWEEN_CONDITION_WILCOXON=false` | A2 — between-condition Wilcoxon. **Shipped `false`.** |
| `RUN_BETWEEN_CONDITION_FISHER=true` | A3 — between-condition (or per-timepoint) Fisher. |
| `RUN_BETWEEN_TIMEPOINT_WILCOXON=false` | A5 — between-timepoint Wilcoxon (pooled). **Shipped `false`.** |
| `RUN_BETWEEN_TIMEPOINT_FISHER=true` | A6 — between-timepoint Fisher within condition. |
| (sourced) `HEADLINE_CONDITION` | `--headline-condition` when set; BWM numerator. |

**Command built (per level).**

```bash
python3 scripts/global_codon_occ_stats.py \
  --raw-dir "$OUT_DIR/raw" --analysis-dir "$OUT_DIR/analysis" \
  --level "$level" --sites E P A \
  --groups "$EXP_GROUPS" \
  "${TIMEPOINTS_FLAG[@]}" "${HEADLINE_FLAG[@]}" \
  --within-condition "${RUN_WITHIN_CONDITION:-true}" \
  --between-condition-wilcoxon "${RUN_BETWEEN_CONDITION_WILCOXON:-true}" \
  --between-condition-fisher "${RUN_BETWEEN_CONDITION_FISHER:-true}" \
  --between-timepoint-wilcoxon "${RUN_BETWEEN_TIMEPOINT_WILCOXON:-true}" \
  --between-timepoint-fisher "${RUN_BETWEEN_TIMEPOINT_FISHER:-true}"
```

**Inputs / Outputs.**

- **Input:** base occupancy CSVs in `raw/`.
- **Output (merged, in `analysis/`, per enabled analysis):** `{level}_within_condition_binomial.csv` (A1), `{level}_wilcoxon_condition.csv` (A2), `{level}_per_timepoint_fisher.csv` (A3), `{level}_wilcoxon_timepoint_{cmp}.csv` (A5), `{level}_timepoint_fisher_within_condition_{cmp}.csv` (A6) — each carrying the E/P/A `site` column.

**Related.** `scripts/global_codon_occ_stats.py` — see [scripts/README.md](../../../scripts/README.md).

---

### `analyze_global_occupancy_fisher_volcano.sh`

**What it does.** Adds R to `PATH`, sources `_headline_config.sh`, self-locates, and runs `between_group_volcano.R` for the per-timepoint Fisher (A3, aa + codon) and, in a loop over `d10_vs_d0 d10_vs_d5 d5_vs_d0`, the within-condition-timepoint Fisher (A6). `$pretty` (`Day 10 vs Day 0`) is built via `sed`.

**CONFIG.** `INPUT_DIR` (`analysis/`), `PLOTS_DIR` (`plots/`), `FORMAT="both"`, `DPI=300`; labels from the sourced config.

**Command built.** `R_scripts/between_group_volcano.R --composite-tag fisher`:

- **A3** (`{level}_per_timepoint_fisher.csv`): `--group-col timepoint --comparison-label "$COMPARISON_LABEL"` → `plots/per_timepoint_fisher/`.
- **A6** (`{level}_timepoint_fisher_within_condition_${comparison}.csv`): `--group-col condition --comparison-label "$pretty"` → `plots/within_condition_timepoint_fisher/${comparison}/`.

**Inputs / Outputs.** A3/A6 merged CSVs → PNG/PDF volcanoes (codon in `codon/` subdirs).

**Related.** `R_scripts/between_group_volcano.R` — see [R_scripts/README.md](../../../R_scripts/README.md).

---

### `analyze_global_occupancy_wilcoxon.sh`

**What it does.** Adds R to `PATH`, sources `_headline_config.sh`, self-locates, and runs `between_group_barplot.R` for the between-condition Wilcoxon (A2, aa + codon) and, in a loop over the three day-pairs, the between-timepoint Wilcoxon (A5). Here `$pretty` uses `Day_10` (underscore) so the value stays path-safe while the barplot R script's `gsub("_"," ")` renders the title as `Day 10 vs Day 0`.

**CONFIG.** `INPUT_DIR` (`analysis/`), `PLOTS_DIR` (`plots/`), `FORMAT="both"`, `DPI=300`; between-condition `--comparison` tag from the sourced `COMPARISON_TAG`.

**Command built.** `R_scripts/between_group_barplot.R`:

- **A2** (`{level}_wilcoxon_condition.csv`): `--comparison "$COMPARISON_TAG"` → `plots/between_condition/`.
- **A5** (`{level}_wilcoxon_timepoint_${comparison}.csv`): `--comparison "$pretty"` → `plots/between_timepoint/${comparison}/`.

**Inputs / Outputs.** A2/A5 merged CSVs → PNG/PDF bar plots (codon in `codon/` subdirs). (These analyses ship `false` in the stats runner, so their CSVs must be produced by enabling A2/A5 first.)

**Related.** `R_scripts/between_group_barplot.R` — see [R_scripts/README.md](../../../R_scripts/README.md).

---

### `analyze_global_occupancy_within_condition.sh`

**What it does.** Adds R to `PATH`, self-locates, builds `OPTIONAL_FLAGS`, and runs `within_condition_volcano.R` for the A1 within-condition binomial CSVs at both levels.

**CONFIG.** `INPUT_DIR` (`analysis/`), `OUTPUT_DIR` (`plots/within_condition`), `ENRICHMENT_TYPE="unweighted"` (weighted volcanoes were dropped from the global reports, since one shared background makes the frequency-weighting add little), `FORMAT="both"`, `DPI=300`, `Y_CAP=""` (empty → disabled).

**Command built.** `R_scripts/within_condition_volcano.R --input {level}_within_condition_binomial.csv --level {aa,codon} --enrichment-type "$ENRICHMENT_TYPE" --format --dpi` plus `OPTIONAL_FLAGS=(--mega-composite [--y-cap …])`. `--mega-composite` preserves global's prior all-groups-grid behaviour (gated behind a flag so the stall-site runs don't produce it).

**Inputs / Outputs.** `{level}_within_condition_binomial.csv` → bar plots under `plots/within_condition/` (codon in `codon/`).

**Related.** `R_scripts/within_condition_volcano.R` — see [R_scripts/README.md](../../../R_scripts/README.md).

## See also

- [c_elegans stage index](../README.md) — pipeline order and experimental design.
- [scripts/README.md](../../../scripts/README.md) · [R_scripts/README.md](../../../R_scripts/README.md).
- [shell_scripts/README.md](../../README.md) · [ribostall root README](../../../README.md).
