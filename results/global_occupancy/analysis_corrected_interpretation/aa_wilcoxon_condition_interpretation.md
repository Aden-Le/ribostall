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
caveats:
  - {label: "timepoint-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "mw-floor-tight", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "n=6-modest-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "bh-per-site", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
caveats_considered: []
headline: "0 of 60 amino-acid tests significant at FDR<0.05 (and none at raw p<0.05); min p_adj = 0.9372, min raw p = 0.1797 — a clean coordinated null at AA resolution; largest non-significant cells are A-site Y (log2_FC -0.2237) and E-site C (+0.0849)."
user_directives:
  - "(triage) 'Confirm test type? Filename wilcoxon; columns median_BWM/median_control/log2_FC/U_stat/p_value/p_adj (n=6 vs 6).' -> 'Confirm Mann-Whitney U'"
  - "(triage) 'How firmly should this read? 0 hits at FDR and raw-p.' -> 'Mixed' (firm-null headline, note 1-2 largest-magnitude cells as non-significant leads)"
  - "(triage) 'Any CSV-specific caveat beyond the 4 family caveats?' -> 'None — family caveats suffice'"
  - "(triage) 'Spotlight any site/feature/group?' -> 'No spotlight'"
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

## Headline
0 of 60 amino-acid tests are significant at FDR<0.05, and none clear raw p<0.05 either (min raw p = 0.1797). The minimum adjusted p is 0.9372 (E-site, amino acid A). This is a coordinated null at AA resolution. As non-significant leads only (Mixed framing): the largest-magnitude depleted cell is A-site Y (`log2_FC` -0.2237, p_adj 1.0000) and the largest-magnitude enriched cell is A-site Q (`log2_FC` +0.1046, p_adj 1.0000); the min-adjusted-p cell sits at E-site (p_adj 0.9372).

## Top hits

Tables are split per site (decoding-site order A / P / E) with the largest-magnitude enriched and depleted cells in each. No row reaches FDR<0.05 or raw p<0.05, so every cell below is a non-significant lead; `flag` is left blank because no per-row floor / nominal-only condition is met.

### Site A

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | Q | +0.1046 | 1.0000 |  |
| enriched | D | +0.0424 | 1.0000 |  |
| enriched | R | +0.0384 | 1.0000 |  |
| enriched | M | +0.0366 | 1.0000 |  |
| enriched | S | +0.0212 | 1.0000 |  |
| depleted | Y | -0.2237 | 1.0000 |  |
| depleted | K | -0.0637 | 1.0000 |  |
| depleted | H | -0.0626 | 1.0000 |  |
| depleted | E | -0.0350 | 1.0000 |  |
| depleted | F | -0.0183 | 1.0000 |  |

<details>
<summary>Site P and Site E</summary>

### Site P

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | W | +0.0647 | 1.0000 |  |
| enriched | M | +0.0580 | 1.0000 |  |
| enriched | Q | +0.0503 | 1.0000 |  |
| enriched | H | +0.0220 | 1.0000 |  |
| enriched | L | +0.0215 | 1.0000 |  |
| depleted | Y | -0.1054 | 1.0000 |  |
| depleted | P | -0.0637 | 1.0000 |  |
| depleted | E | -0.0603 | 1.0000 |  |
| depleted | C | -0.0233 | 1.0000 |  |
| depleted | N | -0.0136 | 1.0000 |  |

### Site E

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | Y | +0.0888 | 0.9372 |  |
| enriched | C | +0.0849 | 0.9372 |  |
| enriched | F | +0.0620 | 0.9372 |  |
| enriched | W | +0.0571 | 0.9372 |  |
| enriched | S | +0.0569 | 0.9372 |  |
| depleted | P | -0.0868 | 0.9372 |  |
| depleted | K | -0.0808 | 0.9372 |  |
| depleted | N | -0.0544 | 0.9372 |  |
| depleted | R | -0.0398 | 0.9372 |  |
| depleted | Q | -0.0348 | 0.9372 |  |

</details>

## Numbers at a glance
- `n_tests`: 60 (20 per site, E/P/A)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.9372 (E-site, A)
- `min raw p`: 0.1797
- `p_floor`: 0.00216 (theoretical two-sided Mann-Whitney floor for n=6 vs n=6). Observed min raw p (0.1797) sits far above this floor, so the null here is signal/power-limited, not floor-limited.
- Per site: site A min p_adj 1.0000; site P min p_adj 1.0000; site E min p_adj 0.9372.

## Methods
Columns present: `site`, `amino_acid`, `median_BWM`, `median_control`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing BWM vs control occupancy per (site, amino acid); user confirmed. The `U_stat` column and the paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's 20-feature family, per `merge_global_occupancy_analysis.py`). The test compares the two condition distributions per cell; it does not address timepoint structure (the 6 reps per condition span day_0/day_5/day_10) and does not estimate effect direction beyond the rank shift summarised by `log2_FC`.

## Caveats
### Confirmed
- **timepoint-pooled-confound** (family-wide) — n=6 per condition is built by treating the 6 reps across day_0/day_5/day_10 as replicates of one condition; if the BWM effect varies across timepoints, pooling timepoints as within-condition noise can mask signal. Applies to both family members.
- **mw-floor-tight** (family-wide) — the theoretical two-sided MW floor for n=6 vs n=6 is ~0.00216; per-site BH families of ~20 make FDR<0.05 feasible but tight. Here the data does not approach the floor (min raw p 0.1797), so the constraint is not binding for this file.
- **n=6-modest-power** (family-wide) — Mann-Whitney with n=6 vs n=6 has modest power; this null is weakly informative rather than strong evidence of no difference.
- **bh-per-site** (family-wide) — `p_adj` is BH-corrected within each site's ~20-feature family, not across the merged E/P/A file.

## For Chumeng (joint-reading hooks)
- Family: `between_condition_wilcoxon` — sister CSV to reconcile: `codon_wilcoxon_condition.csv` (codon resolution of the same BWM-vs-control contrast).
- Does the A-site Y depletion (largest aa cell here, -0.2237 but p_adj 1.0) reappear with consistent direction in `per_timepoint_fisher` (aa) at any timepoint, or in `timepoint_fisher_within_condition`? If yes across designs → the timepoint-pooled MW may be power-blocked; if no → it is a discreteness/noise cell, not signal.
- Where this between-condition MW says null, do the corresponding cells show signal in the per-timepoint Fisher that rides on replicate variance MW would have caught? Falsifier: a cell null here but FDR-significant in per-timepoint Fisher tests whether the pooled framing is lossy vs whether the Fisher p is large-N amplification.
