# stall_sites_non_consensus — Step 2b, per-replicate calling

*Per-replicate stall-site calling — no consensus collapsing — feeding per-replicate Wilcoxon tests (A2 / A5) and their bar plots.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [c_elegans](../README.md) › stall_sites_non_consensus

This is **Step 2b**, the non-consensus branch. Unlike the two consensus stages, stall sites are called **per replicate** and kept per replicate — nothing is collapsed into a group-level agreed set. That distinction is deliberate and structural: because each replicate is an independent observation, the only statistically honest tests here are the **per-replicate Wilcoxon** rank-sum tests, which treat replicates as the sampling unit. Those are **A2** (between-condition Wilcoxon) and **A5** (between-timepoint Wilcoxon).

The count-collapsing tests (A1 binomial, A3/A4 Fisher + background-diff, A6/A7 between-timepoint) pool replicates, which would be pseudoreplication on per-replicate data — so they are **absent from this stats runner entirely**. They live only in the consensus stats runners, which run them at n=1 per cell on reproducibility-filtered sets. The split is by construction: this stage contains no count-collapsing code, so pseudoreplication is impossible, not merely disabled.

Output lands under `results/c_elegans/stall_sites_non_consensus/`: `raw/`, `analysis/`, `plots/`.

## Contents

| File | Kind | Summary |
|---|---|---|
| `_headline_config.sh` | shared config (sourced) | Fixes `HEADLINE_CONDITION`/`OTHER_CONDITION` + derived labels; sourced by the stats runner **and** the plot launcher so the Wilcoxon direction and plot labels can't drift. |
| `run_stall_sites_non_consensus.sh` | `run_*` (drives Python) | Runs `stall_sites_non_consensus.py`: per-replicate stall calling; writes base `stall_sites_*` / `per_group_background_*` CSVs. |
| `run_stall_sites_non_consensus_stats.sh` | `run_*_stats` (drives stats Python) | Runs `stall_sites_non_consensus_stats.py`: A2 between-condition Wilcoxon, A5 between-timepoint Wilcoxon (no count-collapsing tests). |
| `analyze_stall_sites_non_consensus_wilcoxon.sh` | `analyze_*` (drives R) | Bar plots of the A2 / A5 Wilcoxon CSVs via `between_group_barplot.R`. |

The `run_*` scripts activate `ribostall_env`; the `analyze_*` script adds R to `PATH`. All self-locate with `cd "$SCRIPT_DIR/../../.."`.

---

### `_headline_config.sh`

**What it does.** Sourced-only fragment fixing the comparison direction so the Wilcoxon numerator and the plot labels share one source.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `HEADLINE_CONDITION="BWM"` | Numerator; a positive `log2_FC` means enriched in BWM. |
| `OTHER_CONDITION="control"` | Denominator. |
| `COMPARISON_LABEL` *(derived)* | `"BWM vs control"`. |
| `COMPARISON_TAG` *(derived)* | `"BWM_vs_control"` — the Wilcoxon `--comparison` value (a plot prefix/label). |
| `X_LABEL_RATIO` *(derived)* | `"Log2 Enrichment Ratio (BWM / control)"` (present for parity). |

**Related.** Consumed by the stats runner and the plot launcher.

---

### `run_stall_sites_non_consensus.sh`

**What it does.** This file is the project's Python-style reference for the pipeline. It activates the env, self-locates, finds the coverage pickle via `ls "$RIBO_DIR"/*_coverage.pkl.gz | head -1`, echoes a parameter banner, then runs `stall_sites_non_consensus.py` directly (a plain multi-line `python3 … \` invocation rather than a `CMD` array).

**CONFIG.**

| Variable | Meaning |
|---|---|
| `RIBO_DIR`, `RIBO_FILE` | Coverage-pickle directory and the exact C. elegans `.ribo`. |
| `EXP_GROUPS` | The six `(condition, timepoint)` cells, `rep2`/`rep3` each — the reference design the two consensus runners point back to. |
| `TX_THRESHOLD=0.5` | Per-transcript coverage floor (v2: lowered from 1.0). |
| `TX_MIN_REPS_PER_GROUP` | Per-group replicate support for the tx filter; must name every group. |
| `MIN_Z=1.0` | Stall Z-score threshold. |
| `MIN_READS=5` | Min reads at a position (v2: raised from 2). |
| `TRIM_START=20`, `TRIM_STOP=10` | CDS start/stop codon trims (v2 raised TRIM_START 10→20 to match the global script and exclude the initiation ramp). |
| `PSEUDOCOUNT=0.5` | Ratio/enrichment pseudocount. |
| `BASIS="P"`, `PSITE_OFFSET=0` | E/P/A register and codon offset. |
| `DROP_STOP_CODONS="True"` | Drop stall windows whose E/P/A site is a stop codon. |
| `REFERENCE_FILE` | APPRIS C. elegans FASTA. |
| `RAW_DIR="./results/c_elegans/stall_sites_non_consensus/raw"` | Output base-CSV directory. |

Note there are **no** consensus parameters here (no `STALL_MIN_REPS_PER_GROUP`, `TOL`, `MIN_SEP`) — this stage does not build a consensus.

**Command built.** `scripts/stall_sites_non_consensus.py`, passing `--pickle --ribo --reference --groups --tx_threshold --tx_min_reps_per_group --min_z --min_reads --trim-start --trim-stop --pseudocount --basis --psite-offset --drop-stop-codons --out-dir`.

**Inputs / Outputs.** Coverage pickle + reference → `stall_sites_{codon,aa}.csv` and `per_group_background_{codon,aa}.csv` in `raw/`.

**Related.** `scripts/stall_sites_non_consensus.py` — see [scripts/README.md](../../../scripts/README.md).

---

### `run_stall_sites_non_consensus_stats.sh`

**What it does.** Sources `_headline_config.sh`, self-locates, and loops `for LEVEL in aa codon` running `stall_sites_non_consensus_stats.py`. `HEADLINE_FLAG` is populated only when `HEADLINE_CONDITION` is set. There is **no** `--background` flag here (Wilcoxons work on per-replicate frequencies, not a collapsed background).

**CONFIG.**

| Variable | Meaning |
|---|---|
| `EXP_GROUPS` | The six per-replicate cells (`control_day_0:control_day0_rep2,control_day0_rep3;…`). |
| `TIMEPOINTS='day_0,day_5,day_10'` | Chronological; drives the later-vs-earlier day-pairs for A5. Empty → A5 skipped. |
| `RAW_DIR`, `OUT_DIR` | Input `raw/`, output `analysis/`. |
| `RUN_BETWEEN_CONDITION_WILCOXON=false` | A2 — between-condition Wilcoxon. **Note: shipped `false`** (toggle to `true` to emit A2). |
| `RUN_BETWEEN_TIMEPOINT_WILCOXON=false` | A5 — between-timepoint Wilcoxon. **Note: shipped `false`.** |
| (sourced) `HEADLINE_CONDITION` | `--headline-condition` when set; BWM numerator for A2. |

**Command built (per level).**

```bash
python3 scripts/stall_sites_non_consensus_stats.py \
  --stall-sites "$RAW_DIR/stall_sites_${LEVEL}.csv" \
  --groups "$EXP_GROUPS" --timepoints "$TIMEPOINTS" --out-dir "$OUT_DIR" \
  "${HEADLINE_FLAG[@]}" \
  --between-condition-wilcoxon "${RUN_BETWEEN_CONDITION_WILCOXON:-true}" \
  --between-timepoint-wilcoxon "${RUN_BETWEEN_TIMEPOINT_WILCOXON:-true}"
```

Each `RUN_*` value is passed straight through; `false` skips that analysis. Both defaults are `false` as shipped, so re-running this script as-is writes nothing until a toggle is flipped.

**Inputs / Outputs.**

- **Input:** `stall_sites_{aa,codon}.csv` from `raw/` (the background CSV is not consumed).
- **Output (in `analysis/`, when enabled):** `between_condition_wilcoxon_{aa,codon}.csv` (A2), `between_timepoint_wilcoxon_{d10_vs_d0,d10_vs_d5,d5_vs_d0}_{aa,codon}.csv` (A5).

**Related.** `scripts/stall_sites_non_consensus_stats.py` — see [scripts/README.md](../../../scripts/README.md).

---

### `analyze_stall_sites_non_consensus_wilcoxon.sh`

**What it does.** Adds R to `PATH`, sources `_headline_config.sh`, self-locates, and runs `between_group_barplot.R` for both the between-condition and between-timepoint Wilcoxon CSVs, at both levels. Between-timepoint runs in a loop over `d10_vs_d0 d10_vs_d5 d5_vs_d0`, with `$pretty` derived via `sed` (`d10_vs_d0` → `Day10_vs_Day0`).

**CONFIG.** `INPUT_DIR` (`analysis/`), `OUTPUT_DIR` (`plots/between_condition`), `FORMAT="both"`, `DPI=300`; the between-condition `--comparison` tag comes from the sourced `COMPARISON_TAG`. The between-timepoint block writes to a separate `BT_OUTPUT_DIR` (`plots/between_timepoint`).

**Command built.** `R_scripts/between_group_barplot.R`:

- **A2** (`between_condition_wilcoxon_{level}.csv`): `--level {aa,codon} --comparison "$COMPARISON_TAG"` → `plots/between_condition/`.
- **A5** (`between_timepoint_wilcoxon_${comparison}_{level}.csv`): `--comparison "$pretty"` → `plots/between_timepoint/${comparison}/`.

**Inputs / Outputs.** A2/A5 CSVs → PNG/PDF bar plots under `plots/between_condition/` and `plots/between_timepoint/` (codon in `codon/` subdirs). (This is the stage's only plot launcher — non-consensus runs Wilcoxons only, so there are no volcano launchers.)

**Related.** `R_scripts/between_group_barplot.R` — see [R_scripts/README.md](../../../R_scripts/README.md).

## See also

- [stall_sites_consensus_union](../stall_sites_consensus_union/README.md) · [stall_sites_consensus_intersection](../stall_sites_consensus_intersection/README.md) — the consensus branches that own the count-collapsing tests.
- [c_elegans stage index](../README.md) — pipeline order and experimental design.
- [scripts/README.md](../../../scripts/README.md) · [R_scripts/README.md](../../../R_scripts/README.md).
- [shell_scripts/README.md](../../README.md) · [ribostall root README](../../../README.md).
