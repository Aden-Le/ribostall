---
input_csv: results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d10_vs_d0.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_10 vs day_0
test_type_source: user-confirmed
n_tests: 60
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.2857
p_floor: 0.02857
pseudoreplicated: false
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
caveats_considered: []
headline: "0 of 60 day_10-vs-day_0 amino-acid MW tests significant at FDR<0.05 (min p_adj 0.2857); 4 cells reach raw p<0.05, all sitting exactly at the n=4-vs-n=4 two-sided MW floor (0.02857), so the absence of FDR hits is structural floor-blocking, not a biological negative; coordinated floor-limited null. Largest-magnitude nominal-only leads: P-site C (+0.1660) enriched, P-site I (-0.0955) depleted; largest cell overall P-site K (-0.2094) sits above the floor (raw p 0.1143)."
user_directives:
  - "(triage, family-batched) 'Confirm test type? Columns median_day_10/median_day_0/log2_FC/U_stat/p_value/p_adj at n=4 vs 4; my read two-sample Wilcoxon rank-sum (Mann-Whitney U), day_10 vs day_0.' -> 'Confirm Mann-Whitney U (all 6)'"
  - "(triage, family-batched) 'How firmly should these read? 0 FDR hits; raw-p hits only at the n=4 floor.' -> 'Mixed' (firm coordinated-floor-blocked-null headline; floor-level cells reported as nominal-only exploratory leads, symmetric on enriched/depleted)"
  - "(triage, family-batched) 'Per-CSV flags beyond the 4 family caveats?' -> 'Inline stop annotation + formal stop-codon-instability caveat on codon files' (neither applies to this AA file: the AA path drops stop windows, so no stop codon is present here)"
  - "(triage, family-batched) 'Spotlight any site/feature/group?' -> 'No spotlight'"
---

# Interpretation — aa_wilcoxon_timepoint_d10_vs_d0

> Source: `results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d10_vs_d0.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_10 vs day_0 (source: user-confirmed)

## User directives
- (triage, family-batched) "Confirm test type? Columns are median_day_10, median_day_0, log2_FC, U_stat, p_value, p_adj (day_10 vs day_0, n=4 vs 4 pooling 2 BWM + 2 control reps per timepoint)." → "Confirm Mann-Whitney U (all 6)"
- (triage, family-batched) "How firmly should these read? 0 FDR hits across the family; raw-p hits only at the n=4 MW floor (0.0286)." → "Mixed" (firm coordinated-floor-blocked-null headline; note floor-level cells as nominal-only exploratory leads; symmetric enriched/depleted per A.2.2)
- (triage, family-batched) "Per-CSV flags beyond the 4 locked family caveats?" → "Inline stop annotation + formal stop-codon-instability caveat on the codon files." Neither applies to this AA file — the amino-acid path drops stop windows, so this file has no stop codon (60 tests = 20 features × 3 sites).
- (triage, family-batched) "Spotlight any site/feature/group?" → "No spotlight"

## Headline
0 of 60 day_10-vs-day_0 amino-acid tests are significant at FDR<0.05; minimum adjusted p is 0.2857. Four cells reach raw p<0.05, and all four sit exactly at the n=4-vs-n=4 two-sided Mann-Whitney floor (0.02857) — the smallest p the test can return at this N. The absence of FDR hits is therefore a structural floor-blocking outcome, not a biological negative. This is a coordinated floor-limited null. As nominal-only exploratory leads (Mixed framing): the largest-magnitude floor cells are P-site C (`log2_FC` +0.1660, enriched in day_10) and P-site I (-0.0955, depleted); the largest-magnitude cell overall, P-site K (-0.2094), sits above the floor (raw p 0.1143).

## Top hits

Tables are split per site (decoding-site order A / P / E), each showing the top-5 enriched and top-5 depleted cells by raw p (the family clears no FDR threshold, so selection is by raw p), displayed by `|log2_FC|` descending. Every cell flagged `nominal-only` sits exactly at the n=4 two-sided MW floor (raw p 0.02857): its raw significance is the maximum the test can produce at this N, not evidence of a strong effect.

### Site A

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | M | +0.1381 | 0.2857 |  |
| enriched | F | +0.1290 | 0.3810 |  |
| enriched | E | +0.0782 | 0.2857 | nominal-only |
| enriched | S | +0.0749 | 0.2857 | nominal-only |
| enriched | R | +0.0694 | 0.3810 |  |
| depleted | W | -0.1350 | 0.7473 |  |
| depleted | P | -0.1266 | 0.7473 |  |
| depleted | Q | -0.0978 | 0.7473 |  |
| depleted | K | -0.0929 | 0.2857 |  |
| depleted | G | -0.0472 | 0.5714 |  |

<details>
<summary>Site P and Site E</summary>

### Site P

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | E | +0.1775 | 0.4571 |  |
| enriched | C | +0.1660 | 0.2857 | nominal-only |
| enriched | D | +0.1547 | 0.9714 |  |
| enriched | S | +0.0409 | 0.3810 |  |
| enriched | N | +0.0356 | 0.9714 |  |
| depleted | K | -0.2094 | 0.4571 |  |
| depleted | I | -0.0955 | 0.2857 | nominal-only |
| depleted | G | -0.0505 | 0.9714 |  |
| depleted | H | -0.0392 | 0.6667 |  |
| depleted | Q | -0.0303 | 0.9714 |  |

### Site E

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | S | +0.2177 | 0.5714 |  |
| enriched | L | +0.1148 | 0.6667 |  |
| enriched | M | +0.0968 | 0.6667 |  |
| enriched | C | +0.0803 | 0.6667 |  |
| enriched | T | +0.0334 | 0.6667 |  |
| depleted | P | -0.1301 | 0.8571 |  |
| depleted | K | -0.0864 | 0.5714 |  |
| depleted | G | -0.0846 | 0.8831 |  |
| depleted | I | -0.0548 | 0.8831 |  |
| depleted | V | -0.0476 | 0.9143 |  |

</details>

## Numbers at a glance
- `n_tests`: 60 (20 per site, E/P/A)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.2857 (sites A and P)
- `min raw p`: 0.0286 (= the n=4 floor; 4 cells tie here)
- `p_floor`: 0.02857 (theoretical two-sided Mann-Whitney floor for n=4 vs n=4 = 2/C(8,4)). The observed min raw p sits at this floor, so the test maxed out its rank separation at 4 cells yet still cannot clear FDR.
- Per site: site A min p_adj 0.2857; site P min p_adj 0.2857; site E min p_adj 0.5714.

## Methods
Columns present: `site`, `amino_acid`, `median_day_10`, `median_day_0`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing day_10 vs day_0 occupancy per (site, amino acid) at n=4 vs n=4; user confirmed. The `U_stat` column and the paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's 20-feature family, per `merge_global_occupancy_analysis.py`). The test compares the two timepoint distributions per cell; it does not separate the BWM and control reps that are pooled into each n=4 arm, and at n=4 vs n=4 a single FDR-significant cell would require ~12+ features tied at the floor simultaneously.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — the n=4-vs-n=4 two-sided MW floor is exactly 0.02857; with a per-site BH family of 20, clearing FDR<0.05 needs p_adj = 0.02857 × 20/k < 0.05, i.e. k > 11.4 cells tied at the floor at once. Four cells reach the floor here, so FDR<0.05 is mathematically out of reach.
- **condition-pooled-confound** (family-wide) — each n=4 arm pools 2 BWM + 2 control reps; if BWM and control diverge in their day_10-vs-day_0 response, pooling masks divergent signals.
- **n=4-low-power** (family-wide) — n=4 vs n=4 MW has very low power; this null is weakly informative, not strong evidence of no change.
- **p-floor-aware-headline** (family-wide) — the headline states min raw p ≈ 0.0286 and why "no FDR hits" here is uninformative rather than a biological negative.

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs to reconcile: the codon resolution of this same contrast (`codon_wilcoxon_timepoint_d10_vs_d0.csv`) and the other two contrasts (`*_d10_vs_d5`, `*_d5_vs_d0`); see the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- Do the floor cells here (P-site C +, P-site I -, A-site E/S +) recur with consistent direction in `timepoint_fisher_within_condition` (which tests the same day_10-vs-day_0 axis within each condition at whole-transcriptome N)? Falsifier: a floor cell here that is FDR-significant there tests whether the n=4 pooled MW is power-blocked vs whether the Fisher p is large-N amplification of a baseline difference.
- Is site E flatter than A/P across the family (here min p_adj 0.5714 at E vs 0.2857 at A/P)? Falsifier: does the same site-E flatness appear in the within-condition Fisher d10_vs_d0 contrasts, or is it specific to the timepoint-pooled MW?
