---
input_csv: results/global_occupancy/analysis_corrected/aa_wilcoxon_condition.csv
family: between_condition_wilcoxon
test_type: Mann-Whitney U (two-sample Wilcoxon rank-sum), BWM vs control
test_type_source: user-confirmed
n_tests: 60
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.9372
p_floor: 0.00216
pseudoreplicated: false
synced_from_olive_qmd: 2026-06-07
caveats:
  - {label: "timepoint-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "mw-floor-tight", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "n=6-modest-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "bh-per-site", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
caveats_considered:
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: denied, why: "Concern that scipy.stats.mannwhitneyu could fall back to the asymptotic Z approximation under tied per-replicate occupancy values, shifting raw p away from the exact discrete bound (~0.00216 floor).", user_note: "Denied — ruled out empirically. This file directly: the 60 (site, AA) tests collapse to 10 distinct p-values, every U_stat is integer-valued (range 9-27), and all 10 distinct p-values fall on the exact n=6 vs 6 two-sided MW grid (exact branch, no ties). The identical n=6 vs 6 design audited on the per-replicate sister data (_for_claude_mw_branch_audit.py): 0/60 rank ties, method='auto' picked exact for all 60, recomputed raw p matches the pipeline to ~1e-16, and forcing method='asymptotic' shifts raw p by at most 0.0155 (median 0.0102), flips 0 raw-p<0.05 decisions, and leaves the per-site BH conclusion unchanged (0 hits at FDR<0.05 either way; asymptotic marginally more conservative). Audit-sourced, not CSV-verifiable."}
headline: "0 of 60 amino-acid tests significant at FDR<0.05 (and none at raw p<0.05); min p_adj = 0.9372 (E-site, all 20 AAs tied), min raw p = 0.1797 — a clean coordinated null at AA resolution; largest non-significant cells are A-site Y (log2_FC -0.2237) and A-site Q (+0.1046)."
user_directives:
  - "(triage) 'Confirm test type? Filename wilcoxon; columns median_BWM/median_control/log2_FC/U_stat/p_value/p_adj (n=6 vs 6).' -> 'Confirm Mann-Whitney U'"
  - "(triage) 'How firmly should this read? 0 hits at FDR and raw-p.' -> 'Mixed' (firm-null headline, note 1-2 largest-magnitude cells as non-significant leads)"
  - "(triage) 'Any CSV-specific caveat beyond the 4 family caveats?' -> 'None — family caveats suffice'"
  - "(triage) 'Spotlight any site/feature/group?' -> 'No spotlight'"
  - "(readback 2026-06-07) 'Reconciled shared content from the corrected .qmd' -> 'Top hits adopted Olive 6 per-direction sub-tables + raw p_value column (bare AA codes kept); headline largest-enriched corrected E-site C (+0.0849) -> A-site Q (+0.1046); min-p_adj attribution (E-site, A) -> (E-site; all 20 AAs tied at 0.9372); adopted the asymptotic-with-ties Considered-but-not-applicable entry (user-approved). Olive-only sections not imported.'"
---

# Interpretation — aa_wilcoxon_condition

> Source: `results/global_occupancy/analysis_corrected/aa_wilcoxon_condition.csv`
> Family: `between_condition_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U (two-sample Wilcoxon rank-sum), BWM vs control (source: user-confirmed)

## User directives
- (triage) "Confirm test type? Filename suggests wilcoxon; columns are median_BWM, median_control, log2_FC, U_stat, p_value, p_adj (BWM vs control, n=6 vs 6 pooled across timepoints)." → "Confirm Mann-Whitney U"
- (triage) "How firmly should this read? 0 hits at FDR<0.05 and at raw p<0.05 (min raw p = 0.1797)." → "Mixed" (firm-null headline, note 1-2 largest-magnitude cells as non-significant leads)
- (triage) "Any caveat unique to this file beyond the 4 locked family caveats?" → "None — family caveats suffice"
- (triage) "Spotlight any site/feature/group?" → "No spotlight"
- (readback 2026-06-07) Top hits reconciled to Olive's `.qmd` table structure (per-direction sub-tables A/P/E x Enriched/Depleted + raw `p_value` column); the headline's largest-enriched cell corrected from E-site C (+0.0849) to A-site Q (+0.1046) and the min-p_adj attribution from "(E-site, A)" to "(E-site; all 20 AAs tied at 0.9372)" to match the corrected `.qmd`; adopted the asymptotic-with-ties `Considered but not applicable` entry (user-approved). Dylan conventions kept (bare AA codes, terse headline, Methods provenance, Confirmed/Considered caveats); Olive-only sections (Composite, Overview, Biological interpretation, Plots) not imported. Provenance in front-matter `synced_from_olive_qmd`.

## Headline
0 of 60 amino-acid tests are significant at FDR<0.05, and none clear raw p<0.05 either (min raw p = 0.1797). The minimum adjusted p is 0.9372 at the E-site, where all 20 amino acids tie at that value. This is a coordinated null at AA resolution. As non-significant leads only (Mixed framing): the largest-magnitude depleted cell is A-site Y (`log2_FC` -0.2237, p_adj 1.0000) and the largest-magnitude enriched cell is A-site Q (`log2_FC` +0.1046, p_adj 1.0000).

## Top hits

Per (site, direction): top 5 rows by raw `p_value` ascending, with `|log2_FC|` descending as the tiebreaker. `p_value` is the raw two-sided Mann-Whitney p; `p_adj` is BH-corrected per A/P/E site (each site = own family of 20 AAs). Positive `log2_FC` = BWM-enriched; negative = BWM-depleted. No row reaches FDR<0.05 or raw p<0.05 (min raw p 0.1797), so every cell is a non-significant lead; `flag` is blank because no per-row floor / nominal-only condition is met.

### A site - Enriched (BWM > control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| Q | +0.1046 | 0.2403 | 1.0000 |  |
| D | +0.0424 | 0.5887 | 1.0000 |  |
| R | +0.0384 | 0.5887 | 1.0000 |  |
| S | +0.0212 | 0.5887 | 1.0000 |  |
| M | +0.0366 | 0.6991 | 1.0000 |  |

### A site - Depleted (BWM < control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| Y | -0.2237 | 0.3095 | 1.0000 |  |
| E | -0.0350 | 0.3095 | 1.0000 |  |
| K | -0.0637 | 0.5887 | 1.0000 |  |
| H | -0.0626 | 0.5887 | 1.0000 |  |
| F | -0.0183 | 0.8182 | 1.0000 |  |

### P site - Enriched (BWM > control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| W | +0.0647 | 0.2403 | 1.0000 |  |
| Q | +0.0503 | 0.2403 | 1.0000 |  |
| H | +0.0220 | 0.4848 | 1.0000 |  |
| L | +0.0215 | 0.5887 | 1.0000 |  |
| M | +0.0580 | 0.6991 | 1.0000 |  |

### P site - Depleted (BWM < control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| P | -0.0637 | 0.3095 | 1.0000 |  |
| N | -0.0136 | 0.4848 | 1.0000 |  |
| E | -0.0603 | 0.5887 | 1.0000 |  |
| C | -0.0233 | 0.8182 | 1.0000 |  |
| Y | -0.1054 | 0.9372 | 1.0000 |  |

### E site - Enriched (BWM > control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| C | +0.0849 | 0.1797 | 0.9372 |  |
| Y | +0.0888 | 0.3939 | 0.9372 |  |
| F | +0.0620 | 0.4848 | 0.9372 |  |
| W | +0.0571 | 0.4848 | 0.9372 |  |
| S | +0.0569 | 0.4848 | 0.9372 |  |

### E site - Depleted (BWM < control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| K | -0.0808 | 0.1797 | 0.9372 |  |
| N | -0.0544 | 0.1797 | 0.9372 |  |
| P | -0.0868 | 0.6991 | 0.9372 |  |
| R | -0.0398 | 0.6991 | 0.9372 |  |
| Q | -0.0348 | 0.6991 | 0.9372 |  |

## Numbers at a glance
- `n_tests`: 60 (20 per site, A/P/E)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.9372 (E-site; all 20 amino acids tied at this value)
- `min raw p`: 0.1797
- `p_floor`: 0.00216 (theoretical two-sided Mann-Whitney floor for n=6 vs n=6). Observed min raw p (0.1797) sits far above this floor, so the null here is signal/power-limited, not floor-limited.
- Per site: site A min p_adj 1.0000; site P min p_adj 1.0000; site E min p_adj 0.9372 (after BH, all 20 cells within each site share that site's adjusted p).

## Methods
Columns present: `site`, `amino_acid`, `median_BWM`, `median_control`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing BWM vs control occupancy per (site, amino acid); user confirmed. The `U_stat` column and the paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's 20-feature family, per `merge_global_occupancy_analysis.py`). The test compares the two condition distributions per cell; it does not address timepoint structure (the 6 reps per condition span day_0/day_5/day_10) and does not estimate effect direction beyond the rank shift summarised by `log2_FC`.

## Caveats
### Confirmed
- **timepoint-pooled-confound** (family-wide) — n=6 per condition is built by treating the 6 reps across day_0/day_5/day_10 as replicates of one condition; if the BWM effect varies across timepoints, pooling timepoints as within-condition noise can mask signal. Applies to both family members.
- **mw-floor-tight** (family-wide) — the theoretical two-sided MW floor for n=6 vs n=6 is ~0.00216; per-site BH families of ~20 make FDR<0.05 feasible but tight. Here the data does not approach the floor (min raw p 0.1797), so the constraint is not binding for this file.
- **n=6-modest-power** (family-wide) — Mann-Whitney with n=6 vs n=6 has modest power; this null is weakly informative rather than strong evidence of no difference.
- **bh-per-site** (family-wide) — `p_adj` is BH-corrected within each site's ~20-feature family, not across the merged E/P/A file.

### Considered but not applicable
- **asymptotic-with-ties** (per-CSV) — flagged that scipy could fall back to the asymptotic Z approximation under tied per-replicate occupancy values, shifting raw p off the exact discrete bound (~0.00216 floor); ruled out empirically. This file directly: the 60 (site, AA) tests collapse to 10 distinct p-values, every `U_stat` is integer-valued (range 9-27), and all 10 distinct p-values fall on the exact n=6 vs 6 two-sided MW grid — the exact-branch, no-ties signature. The identical n=6 vs 6 design audited on the per-replicate sister data (`_for_claude_mw_branch_audit.py`): 0/60 rank ties, `method='auto'` picked exact for all 60, recomputed raw p matches the pipeline to ~1e-16, and forcing `method='asymptotic'` shifts raw p by at most 0.0155 (median 0.0102), flips 0 raw-p<0.05 decisions, and leaves the per-site BH conclusion unchanged (0 hits at FDR<0.05 either way; asymptotic marginally more conservative). Audit-sourced (`_for_claude_mw_branch_audit.py`), not CSV-verifiable.

## For Chumeng (joint-reading hooks)
- Family: `between_condition_wilcoxon` — sister CSV to reconcile: `codon_wilcoxon_condition.csv` (codon resolution of the same BWM-vs-control contrast).
- Does the A-site Y depletion (largest aa cell here, -0.2237 but p_adj 1.0) reappear with consistent direction in `per_timepoint_fisher` (aa) at any timepoint, or in `timepoint_fisher_within_condition`? If yes across designs → the timepoint-pooled MW may be power-blocked; if no → it is a discreteness/noise cell, not signal.
- Where this between-condition MW says null, do the corresponding cells show signal in the per-timepoint Fisher that rides on replicate variance MW would have caught? Falsifier: a cell null here but FDR-significant in per-timepoint Fisher tests whether the pooled framing is lossy vs whether the Fisher p is large-N amplification.
