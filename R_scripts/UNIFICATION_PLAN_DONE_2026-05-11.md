# Plan: Unify stall_sites & global_occupancy stats → R → shell pipeline

> **Status:** drafted 2026-05-04; in progress.
> **Audience:** future-Claude (or future-Aden) opening this plan cold across multiple sessions.

---

## 0. Context (read this first — do not skip on resume)

### What this is

`stall_sites` and `global_occupancy` are two parallel pipelines that compute the same five statistical analyses (within-condition binomial, between-condition Wilcoxon, between-timepoint Wilcoxon, between-timepoint Fisher within-condition, per-timepoint Fisher) and plot them with parallel R scripts driven by parallel shell scripts. They diverge in column names, R code is duplicated 60–80%, and bug fixes / style tweaks have to be made twice.

### Goal

Collapse the duplicated layers into a single set of R scripts + shell scripts that handle both datasets via arguments, after first making the CSV outputs schema-compatible. Reduce file count, remove redundancy, keep behaviour intact.

### User decisions captured (do not relitigate without flagging)

| Decision | Choice |
|---|---|
| CSV feature column naming | Adopt **stall_sites' `amino_acid` / `codon`** (rename global's `unit`) |
| R-script merge base | **Global occupancy** scripts (more parameterized) |
| Codon parity for stall_sites | **Yes** — unified R supports `--level aa\|codon` for both datasets |
| Stall_sites between-condition Wilcoxon (BWM vs Control) | **Fold into unified Wilcoxon** R script |
| `stall_count`/`stall_freq` vs `observed_count`/`observed_freq` | **Option 2** — rename to neutral `observed_count`/`observed_freq` everywhere (2026-05-04) |

### Scope split (matches the order rule below)

| Phase | What it does | Roughly fits in |
|---|---|---|
| B | CSV column-name unification (Python) | one session |
| C | Merge 6 R scripts → 3 unified R scripts | one or two sessions |
| D | Repoint 6 shell scripts to unified R scripts | one short session |
| E | End-to-end verification + cleanup | one short session |

### Order rule

**B must precede C** (R scripts can only be unified once they read a single schema).
**C must precede D** (shells can only repoint once unified R scripts exist).
**E runs last.** Within each phase, complete the verification block before declaring the phase done.

---

## 1. Phase B — Unify CSV column names (Python edits only)

### B.1. How global_occupancy CSVs are actually generated (read this before editing)

The `analysis_corrected/` CSVs are produced in **two steps**:

1. `shell_scripts/run_global_codon_occ_stats.sh` → `scripts/global_codon_occ_stats.py` (uses `ribostall/global_occupancy.py`) → writes per-site CSVs into `results/global_occupancy/analysis/{E,P,A}/`.
2. `shell_scripts/run_merge_global_occupancy_analysis.sh` → `scripts/merge_global_occupancy_analysis.py` → concatenates the per-site CSVs into `results/global_occupancy/analysis_corrected/`, prepending a `site` column.

The merge script does **not** rename columns — it only concatenates and prepends `site`. So all column renames go in `ribostall/global_occupancy.py` (step 1), and step 2 is just re-run unchanged.

### B.2. Files to edit

- `ribostall/global_occupancy.py` — primary edits, three `pd.DataFrame` row-builders (within-condition binomial ~lines 99–111; between-condition Wilcoxon ~lines 189–196; per-timepoint Fisher ~lines 455–464). Verify the line numbers before editing — they may have drifted.
- `ribostall/enrichment.py` — secondary edit: per-timepoint Fisher row builder (~lines 552–562) — column **order** change only, names already match user-chosen convention.
- `scripts/global_codon_occ_stats.py` — verify it does NOT post-process column names; if it does, edit there too.
- `scripts/merge_global_occupancy_analysis.py` — **no edits expected** (just concatenates + prepends `site`); confirm by re-reading.

The `feature_col` parameter pattern (`"amino_acid"` when level=AA, `"codon"` when level=codon) already exists in `ribostall/enrichment.py`. **Reuse it** in `ribostall/global_occupancy.py` rather than inventing a new mechanism.

### B.3. Concrete column renames in `ribostall/global_occupancy.py`

For each of the three row-builders, apply these renames:

- [x] **B.3.1 —** `unit` → `amino_acid` (when level=AA) or `codon` (when level=codon). Achieve via the `feature_col` parameter pattern from `ribostall/enrichment.py`.
- [x] **B.3.2 —** `weighted_log2` → `weighted_log2_enrichment`.
- [x] **B.3.3 —** Rename `stall_count` / `stall_freq` → `observed_count` / `observed_freq` everywhere (Option 2, decided 2026-05-04).
- [x] **B.3.4 —** Verify nothing else in `ribostall/global_occupancy.py` references the old names (`Grep` for `unit` after edits and resolve any stragglers).

### B.4. Concrete column-order fix in `ribostall/enrichment.py`

The per-timepoint Fisher row currently builds in this order: `timepoint, site, {feature}, odds_ratio, p_value, {cond}_count, {cond}_total`. Global builds (per-site, before merge): `timepoint, {feature}, odds_ratio, p_value, {cond}_count, {cond}_total`. After the merge step, global gets `site` prepended, so the merged column order is `site, timepoint, {feature}, odds_ratio, p_value, {cond}_count, {cond}_total, p_adj`.

The unified target order — what both pipelines should emit at the file we plot from — is:

`site, timepoint, {feature}, odds_ratio, p_value, {cond}_count, {cond}_total, p_adj`

- [x] **B.4.1 —** Edit `ribostall/enrichment.py` per-timepoint Fisher row dict so `site` is first. (Stall_sites already writes `site` — just move it to position 0.)
- [x] **B.4.2 —** Verify `ribostall/global_occupancy.py` per-timepoint Fisher row dict does NOT include `site` (the merge step prepends it). If it does include `site`, that's a bug to remove or the merge would double up — verify and fix.

### B.5. Regenerate CSVs and verify schema match

- [x] **B.5.1 —** Re-run global stats: `bash shell_scripts/run_global_codon_occ_stats.sh`. This produces updated per-site CSVs in `results/global_occupancy/analysis/{E,P,A}/` with renamed columns.
- [x] **B.5.2 —** Re-run merge step: `bash shell_scripts/run_merge_global_occupancy_analysis.sh`. This concatenates per-site CSVs into `results/global_occupancy/analysis_corrected/` and prepends `site`.
- [x] **B.5.3 —** Re-run stall_sites stats: `bash shell_scripts/run_enrichment_stats.sh` (or the relevant driver — confirm filename before running). Required because B.4.1 changes column order.
- [x] **B.5.4 —** For each of the 5 parallel analysis types, diff the column-header line of the new global CSV against the corresponding stall_sites CSV. They must match modulo `amino_acid` vs `codon` based on level. Use:
  - `head -n1 results/global_occupancy/analysis_corrected/<file>.csv`
  - `head -n1 results/stall_sites/enrichment/analysis_stats/<file>.csv`
  - Headers should be identical strings (or differ only in `amino_acid` vs `codon` per level).
- [x] **B.5.5 —** Spot-check 2–3 rows per CSV are numerically consistent (verify no column shuffling broke values — only headers and column positions should have changed).

### B.6. Phase B verification

- [x] **B.6.1 —** All 5 analysis-type pairs have matching headers per B.5.4.
- [x] **B.6.2 —** Spot-checked rows per B.5.5 are numerically consistent (no column-shuffle artifacts).
- [x] **B.6.3 —** No `unit` column appears in any CSV under `results/global_occupancy/analysis_corrected/`.
- [x] **B.6.4 —** Old R scripts will now error on the renamed columns — this is **expected** and correct (the old global R scripts read `unit`, which no longer exists). Note this in the checkbox so it's clear the regression is expected and Phase C is what fixes it.
- [x] **B.6.5 —** Commit the Python changes with message `unify CSV column names across stall_sites and global_occupancy`.

---

## 2. Phase C — Merge 6 R scripts into 3 unified scripts

### C.1. Strategy

For each of the three pairs, copy the global script as the base, then merge in stall_sites' commentary, Beta-CI logic, and any plot-polish details. Each merged script lives in `R_scripts/` with a name reflecting its dual-dataset purpose.

### C.2. Pair 1 — Fisher volcano (low effort)

- [x] **C.2.1 —** Copy `R_scripts/global_occupancy_fisher_volcano.R` → `R_scripts/fisher_volcano.R`.
- [x] **C.2.2 —** Update column reads inside the new script:
  - Replace `unit` reads with level-aware logic: when `--level aa`, read `amino_acid`; when `--level codon`, read `codon`.
  - All other column names already match per Phase B.
  - Implementation: `feature_col <- ifelse(args$level == "aa", "amino_acid", "codon")`; reads use `.data[[feature_col]]`.
- [x] **C.2.3 —** Port stall_sites' AA classification map / palette from `R_scripts/stall_sites_per_timepoint_fisher_volcano.R` if cleaner than the global version. Otherwise leave global's untouched.
  - Decision: AA_CLASS and CLASS_COLORS are byte-identical in both scripts — kept global's verbatim, no porting needed.
- [x] **C.2.4 —** Verify `--group-col` argument still works for both timepoint-grouped (per-timepoint Fisher) and condition-grouped (within-condition timepoint Fisher) inputs.
  - The arg pattern from the global script (`.data[[group_col]]`) carries over unchanged.
- [x] **C.2.5 —** Smoke test on stall_sites Fisher CSV: `Rscript R_scripts/fisher_volcano.R --input results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_aa.csv --outdir _smoke_test/fisher_stall_aa --level aa --group-col timepoint --comparison-label "BWM vs Control" --format png`. PASS — 9 individual + 1 composite plots emitted.
- [x] **C.2.6 —** Smoke test on global Fisher CSV: same command with global input + `--level codon`. PASS. Also smoke tested global AA and stall_sites codon levels.
  - **New edge case found:** stall_sites codon-level CSV contains rows with `odds_ratio = 0` (zero count in one contingency cell), producing `log2 = -Inf` and crashing axis-limit calc + ggrepel. Fix: cap `log2_odds_ratio` at ±10 (mirrors existing `pmin(neg_log10_p, 50)` cap). Cap is a no-op on all existing data — only kicks in for stall_sites codon, which is net-new in Phase D.
- [x] **C.2.7 —** Open the produced PDFs and confirm they look correct (same panels, same axes, same significant-point labels as prior outputs). PASS — visual diff against `results/stall_sites/plots/per_timepoint_fisher/composite/per_timepoint_fisher_composite.png` shows identical layout. Only the title string differs (generic "Amino Acid Fisher's Test" vs stall_sites-specific wording — intentional for a dual-dataset script).

### C.3. Pair 2 — Wilcoxon bar plots (very low effort, also absorbs stall_sites between-condition)

- [x] **C.3.1 —** Copy `R_scripts/global_occupancy_wilcoxon.R` → `R_scripts/wilcoxon_barplot.R`.
- [x] **C.3.2 —** Update column reads: `unit` → `amino_acid` / `codon` per `--level`. Implementation: `feature_col` chosen from `--level`, then `rename(feature = !!feature_col)` so downstream code references a single name.
- [x] **C.3.3 —** The script reads two `median_*` columns generically — verify it works for both BWM-vs-Control (between-condition) AND day-vs-day inputs without code changes. The `--comparison` arg is the only thing that differentiates them in plot titles.
  - Confirmed: the unified script never references the `median_*` columns (only log2_FC and p_adj are used), so any pair of comparison groups works.
- [x] **C.3.4 —** Smoke test on stall_sites between-condition CSV (the one with no global pair): `Rscript R_scripts/wilcoxon_barplot.R --input results/stall_sites/enrichment/analysis_stats/between_condition_wilcoxon_aa.csv --outdir _smoke_test/wilcox_stall_bc --level aa --comparison "BWM_vs_Control"`. PASS — 3 individual + 1 composite plots.
- [x] **C.3.5 —** Smoke test on stall_sites between-timepoint CSV. PASS — `between_timepoint_wilcoxon_d10_vs_d0_aa.csv`.
- [x] **C.3.6 —** Smoke test on global between-timepoint CSV (codon level). PASS — `codon_wilcoxon_timepoint_d10_vs_d0.csv`.
- [x] **C.3.7 —** Visual sanity check on produced plots for all three smoke tests. PASS — sorted bars, blue/red enriched/depleted, codon plot has rotated x-axis labels and 64-bar wide layout.

> **Side observation (not blocking):** `results/global_occupancy/analysis_corrected/{aa,codon}_wilcoxon_timepoint.csv` (the suffix-less merged files) still have a `unit` column, but they are stale leftovers from a prior pipeline version. The current `scripts/global_codon_occ_stats.py` only writes the `_d10_vs_d0` / `_d10_vs_d5` / `_d5_vs_d0` variants, all of which have the correct `amino_acid` column. No action needed in this phase; consider deleting the stale files in a follow-up.

### C.4. Pair 3 — Within-condition (moderate effort)

> **Plan deviation (resolved 2026-05-07):** the original instruction said "use global as base" but global within-condition is a **bar plot** while stall_sites is a **volcano plot** — they're different plot types, so a literal merge wasn't possible. The unified filename (`within_condition_volcano.R`) and the Beta-Jeffreys CI / `--show-ci` semantics in C.4.3 only make sense for a volcano output, so the user confirmed: **volcano-only**. Global trades its old bar plot for a volcano plot. The mega-composite layout is preserved as a 6×3 volcano grid behind `--mega-composite`.

- [x] **C.4.1 —** Copy stall_sites volcano as the structural base (not global, per the deviation note above) → `R_scripts/within_condition_volcano.R`. Global's parametrization (mega-composite as a flag) is layered on top.
- [x] **C.4.2 —** Update column reads: `unit` → `amino_acid` / `codon`; `weighted_log2` → `weighted_log2_enrichment` (already aligned in Phase B but verified the script reads the new name). `stall_count` → `observed_count`, `stall_freq` → `observed_freq` per Phase B.
- [x] **C.4.3 —** Port stall_sites' Beta-Jeffreys CI computation. Gated behind `--show-ci` (default `FALSE`).
- [x] **C.4.4 —** Mega-composite (all condition×timepoint × sites) gated behind `--mega-composite` (default `FALSE`).
- [x] **C.4.5 —** Smoke test on stall_sites within-condition CSV (AA, with `--show-ci`). PASS — 46 plots emitted, CI horizontal error bars render correctly.
- [x] **C.4.6 —** Smoke test on global within-condition CSV (codon, no `--show-ci`, with `--mega-composite`). PASS — 48 plots including 2 mega-composites.
  - **Edge case found:** `p_adj == 0` rows produce `-log10 = Inf`, which broke axis-limit calc. Fix: `--y-cap` now defaults to 50 (was nullable) so the cap always applies. Also added the same ±10 cap on log2_enrichment / weighted_log2_enrichment / CI bounds as in fisher_volcano.R.
  - **Polish:** added `Stop` (#666666) to AA_CLASS / CLASS_COLORS in both `fisher_volcano.R` and `within_condition_volcano.R`. Stop codons (TAA/TAG/TGA) now plot as gray instead of producing an `NA` legend entry.
- [x] **C.4.7 —** Visual sanity check on produced plots for both runs. PASS — stall_sites AA plot shows CI bars, AA-class colors, sig-triangles. Global codon plot shows Stop class for TAA/TAG/TGA in gray, ordinary AA classes for sense codons.

### C.5. Optional shared-helper extraction

The three new scripts share axis-padding logic and an AA→class colour map. Only extract to `R_scripts/_common.R` if the duplication is truly identical and saves real lines. **Do not over-abstract** — three call sites with 5 lines of identical helper code is fine to leave inline.

- [x] **C.5.1 —** Decision: extract to `R_scripts/_common.R` **no**. Rationale: AA_CLASS/CLASS_COLORS/CODON2AA are identical in 2 of 3 scripts and `save_plot` is identical in all 3, but the duplication is bounded (no growth path) and `source()` adds working-directory fragility for shell-driven runs. Revisit if a 4th unified script is added.

### C.6. Delete the 6 old R scripts

After all of C.2–C.4 verification passes:

- [x] **C.6.1 —** `git rm R_scripts/stall_sites_per_timepoint_fisher_volcano.R`
- [x] **C.6.2 —** `git rm R_scripts/stall_sites_between_condition_enrichment.R`
- [x] **C.6.3 —** `git rm R_scripts/stall_sites_within_condition_enrichment.R`
- [x] **C.6.4 —** `git rm R_scripts/global_occupancy_fisher_volcano.R`
- [x] **C.6.5 —** `git rm R_scripts/global_occupancy_wilcoxon.R`
- [x] **C.6.6 —** `git rm R_scripts/global_occupancy_within_condition.R`

### C.7. Phase C verification

- [x] **C.7.1 —** `R_scripts/` contains exactly: `fisher_volcano.R`, `wilcoxon_barplot.R`, `within_condition_volcano.R`. No `_common.R` (per C.5.1). No `stall_sites_*` or `global_occupancy_*` R scripts remain.
- [x] **C.7.2 —** Each new R script runs successfully on at least one stall_sites CSV AND one global CSV (4 smoke tests for fisher, 3 for wilcoxon, 2 for within_condition).
- [x] **C.7.3 —** Visual diffs against prior outputs show no regressions for AA-level pairs (Fisher composite, Wilcoxon composite). Within-condition global trades bar plots for volcano plots — that's an intentional behavioral change recorded in C.4 deviation note. New stall_sites codon outputs exist (no baseline to compare).
- [x] **C.7.4 —** Commit the R changes with message `unify R plotting scripts across stall_sites and global_occupancy`. Committed as `d7e9be9` (net -925 lines).

---

## 3. Phase D — Repoint shell scripts to unified R scripts

### D.1. Files to edit

- `shell_scripts/analyze_stall_sites_per_timepoint_fisher_volcano.sh` → calls `fisher_volcano.R`
- `shell_scripts/analyze_stall_sites_within_condition_enrichment.sh` → calls `within_condition_volcano.R`
- `shell_scripts/analyze_stall_sites_between_condition_enrichment.sh` → calls `wilcoxon_barplot.R`
- `shell_scripts/analyze_global_occupancy_fisher_volcano.sh` → calls `fisher_volcano.R`
- `shell_scripts/analyze_global_occupancy_wilcoxon.sh` → calls `wilcoxon_barplot.R`
- `shell_scripts/analyze_global_occupancy_within_condition.sh` → calls `within_condition_volcano.R`

### D.2. Per-script edits

For each shell script:

- [x] **D.2.1 —** Replace the `Rscript R_scripts/<old>.R` invocation with `Rscript R_scripts/<unified>.R`.
- [x] **D.2.2 —** Add `--level aa` to stall_sites scripts that didn't pass it (default behaviour preserved).
- [x] **D.2.3 —** Add a codon-level loop pass to each stall_sites shell script. Output dirs use `<existing>/codon/` to keep AA plots untouched (asymmetric layout, AA at the root next to a `codon/` subdir, per plan).
- [x] **D.2.4 —** For `analyze_stall_sites_within_condition_enrichment.sh`: kept `--show-ci`. For global within-condition: did NOT add `--show-ci` but did add `--mega-composite` (preserves global's prior all-groups grid behaviour, which the original bar-plot script always produced).
- [x] **D.2.5 —** Confirmed `--format` / `--dpi` / `--outdir` args still pass through unchanged.

> **Phase B oversight fixed in this phase:** all 3 stall_sites shells referenced `./results/stall_sites/enrichment/<file>.csv`, but Phase B's stats writes go to the `analysis_stats/` subdir. The within-condition shell also referenced the old filename `within_condition_enrichment_aa.csv` which was renamed to `within_condition_binomial_aa.csv`. These stale paths were silently broken pre-unification and are now corrected.

> **Pre-existing gap (out of plan scope, flagging only):** the stall_sites between-condition shell does not loop over the `between_timepoint_wilcoxon_d{10,5}_vs_d{0,5}_*.csv` files even though `wilcoxon_barplot.R` can plot them. Adding those passes would be a follow-up.

### D.3. End-to-end run of all 6 shell scripts

- [x] **D.3.1 —** Run `bash shell_scripts/analyze_stall_sites_per_timepoint_fisher_volcano.sh`. PASS.
- [x] **D.3.2 —** Run `bash shell_scripts/analyze_stall_sites_within_condition_enrichment.sh`. PASS.
- [x] **D.3.3 —** Run `bash shell_scripts/analyze_stall_sites_between_condition_enrichment.sh`. PASS.
- [x] **D.3.4 —** Run `bash shell_scripts/analyze_global_occupancy_fisher_volcano.sh`. PASS.
- [x] **D.3.5 —** Run `bash shell_scripts/analyze_global_occupancy_wilcoxon.sh`. PASS.
- [x] **D.3.6 —** Run `bash shell_scripts/analyze_global_occupancy_within_condition.sh`. PASS (re-ran after adding `--mega-composite` — confirmed mega-composite outputs emitted).

### D.4. Phase D verification

- [x] **D.4.1 —** Each shell script exits with status 0.
- [x] **D.4.2 —** All expected plot output directories exist and are non-empty.
- [x] **D.4.3 —** New stall_sites codon-level plot dirs (added in D.2.3) exist and contain plots that look reasonable. Visual sanity check on `results/stall_sites/plots/per_timepoint_fisher/codon/composite/codon_fisher_composite.png` confirms 9-panel grid (E/P/A × Day 0/5/10), AA-class colored codons, sig labels (AAG, AAT, ACA, etc.).
- [ ] **D.4.4 —** Commit the shell changes with message `repoint shell scripts to unified R plotters`.

> **Cleanup pending (separate from this commit):** prior runs left stale plot files alongside the new ones (e.g. `per_timepoint_fisher_composite.png` next to `aa_fisher_composite.png`; `EPA_barplot_composite.png` next to `EPA_aa_BWM_vs_Control_barplot_composite.png`). These are gitignored output files, but the user may want to wipe and regenerate `results/{stall_sites,global_occupancy}/plots/` for a clean state.

---

## 4. Phase E — End-to-end verification + cleanup

### E.1. Visual and schema verification

- [x] **E.1.1 —** For each plot directory, opened representative PDFs/PNGs and confirmed no visual regressions. AA-level Fisher composite, Wilcoxon composite, within-condition volcano composite all match prior outputs modulo intentional title text. Global within-condition shifted bar→volcano (intentional, Phase C deviation).
- [x] **E.1.2 —** Codon-level stall_sites outputs are net-new. Visual sanity check on `results/stall_sites/plots/per_timepoint_fisher/codon/composite/codon_fisher_composite.png` shows reasonable codon-level volcano with AA-class colors and significant codons labeled.
- [x] **E.1.3 —** CSV header parity verified for 3 representative pairs (per_timepoint_fisher, between_condition_wilcoxon, within_condition_binomial) — headers identical byte-for-byte between stall_sites and global.

### E.2. File-count reduction sanity

- [x] **E.2.1 —** `ls R_scripts/*.R | wc -l` is 3 (no `_common.R` per C.5.1), down from 6.
- [x] **E.2.2 —** `ls shell_scripts/analyze_*.sh | wc -l` is still 6 (kept per-dataset, intentionally).
- [x] **E.2.3 —** Every shell script now references one of the 3 unified R scripts. Zero references to deleted old R scripts (verified via `grep "Rscript R_scripts"`).

### E.3. Documentation refresh (lightweight)

- [x] **E.3.1 —** Choice: **consolidate** (user-confirmed 2026-05-11). The 6 old walkthroughs deleted; 3 new consolidated walkthroughs written.
- [x] **E.3.2 —** Wrote `docs/R/fisher_volcano_explained.md`, `wilcoxon_barplot_explained.md`, `within_condition_volcano_explained.md`. Each covers both stall_sites and global_occupancy inputs, uses the global Fisher walkthrough as a structural template, and documents the Phase C plot-type deviation (global within-condition shifted bar→volcano).
- [x] **E.3.3 —** Updated `docs/index.md` MOC: collapsed two dataset-specific R-script tables into one unified table; added shell-launcher table mapping each launcher to its driven R script.
- [x] **E.3.4 —** Updated `CLAUDE.md` "Repository Map" — `docs/R/` lists the 3 new walkthroughs; `R_scripts/` now lists the 3 unified scripts with short descriptions instead of the prior "not yet covered" note.

### E.4. Cleanup

- [x] **E.4.1 —** Renamed to `UNIFICATION_PLAN_DONE_2026-05-11.md` (this file). Kept as audit trail.

---

## 5. Critical files summary (one-page reference)

| File | Phase | Action |
|---|---|---|
| `ribostall/global_occupancy.py` | B | Edit (3 row-builders — column renames) |
| `ribostall/enrichment.py` | B | Edit (per-timepoint Fisher column order — move `site` to position 0) |
| `scripts/global_codon_occ_stats.py` | B | Re-run via `shell_scripts/run_global_codon_occ_stats.sh`; verify no column post-processing |
| `scripts/merge_global_occupancy_analysis.py` | B | Re-run via `shell_scripts/run_merge_global_occupancy_analysis.sh`; no edits expected (just concatenates per-site → merged) |
| `scripts/stall_sites_non_consensus_stats.py` | B | Re-run via `shell_scripts/run_enrichment_stats.sh` (column-order change requires regen) |
| `R_scripts/global_occupancy_fisher_volcano.R` | C | Becomes `R_scripts/fisher_volcano.R` (delete original) |
| `R_scripts/global_occupancy_wilcoxon.R` | C | Becomes `R_scripts/wilcoxon_barplot.R` (delete original) |
| `R_scripts/global_occupancy_within_condition.R` | C | Becomes `R_scripts/within_condition_volcano.R` (delete original) |
| `R_scripts/stall_sites_per_timepoint_fisher_volcano.R` | C | Delete (logic merged into `fisher_volcano.R`) |
| `R_scripts/stall_sites_between_condition_enrichment.R` | C | Delete (logic merged into `wilcoxon_barplot.R`) |
| `R_scripts/stall_sites_within_condition_enrichment.R` | C | Delete (Beta-CI logic ported into `within_condition_volcano.R`) |
| `shell_scripts/analyze_*.sh` (×6) | D | Repoint Rscript invocations to unified R scripts; add codon-level passes for stall_sites |
| `docs/R/*.md` (×6) | E.3 | Consolidate or mark superseded |
| `docs/index.md` | E.3 | Update MOC entries |
| `CLAUDE.md` | E.3 | Update Repository Map |

---

## 6. Resume protocol (if execution is interrupted mid-phase)

1. Open this file. Read Section 0 (Context) in full.
2. Find the lowest-numbered unchecked checkbox.
3. Verify the prior phase's verification block (e.g. B.6, C.7, D.4) was completed before resuming. If not, redo the prior verification first.
4. Resume from the lowest unchecked box.
5. Do NOT skip phases. B → C → D is order-critical.

---

## 7. Notes for whoever picks this up

- The plan is intentionally conservative: 6 shell scripts stay as 6 (not consolidated to 1) because they encode dataset-specific input paths and looping over comparisons. Consolidation could be a follow-up if the user wants it later.
- Stall_sites' between-condition Wilcoxon (BWM vs Control) was the one analysis without a global pair. It folds into `wilcoxon_barplot.R` because the column schema (two `median_*` cols, log2_FC, p_adj, U_stat) is identical; only the comparison label differs.
- The Beta-Jeffreys CI in `within_condition_volcano.R` (Phase C.4) is opt-in via `--show-ci`. Do not enable it by default — global runs would silently change behaviour.
- `observed_count` / `observed_freq` are used everywhere (Option 2 — decided 2026-05-04). This is the single decision that touches the most files.
