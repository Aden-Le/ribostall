---
input_csv: results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d10_vs_d5.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_10 vs day_5
test_type_source: user-confirmed
n_tests: 60
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.5714
p_floor: 0.02857
pseudoreplicated: false
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
caveats_considered: []
headline: "0 of 60 day_10-vs-day_5 amino-acid MW tests significant at FDR<0.05 (min p_adj 0.5714); a single cell reaches raw p<0.05 — P-site I (-0.1469), sitting exactly at the n=4 MW floor (0.02857); coordinated floor-limited null, the absence of FDR hits is floor-blocking not a biological negative. Largest enriched cell P-site G (+0.1524) is just above the floor (raw p 0.0571); site E is the flattest site (min p_adj 1.0)."
user_directives:
  - "(triage, family-batched) 'Confirm test type? Columns median_day_10/median_day_5/log2_FC/U_stat/p_value/p_adj at n=4 vs 4; my read two-sample Wilcoxon rank-sum (Mann-Whitney U), day_10 vs day_5.' -> 'Confirm Mann-Whitney U (all 6)'"
  - "(triage, family-batched) 'How firmly should these read? 0 FDR hits; raw-p hits only at the n=4 floor.' -> 'Mixed' (firm coordinated-floor-blocked-null headline; floor-level cells reported as nominal-only exploratory leads, symmetric on enriched/depleted)"
  - "(triage, family-batched) 'Per-CSV flags beyond the 4 family caveats?' -> 'Inline stop annotation + formal stop-codon-instability caveat on codon files' (neither applies to this AA file: the AA path drops stop windows, so no stop codon is present here)"
  - "(triage, family-batched) 'Spotlight any site/feature/group?' -> 'No spotlight'"
---

# Interpretation — aa_wilcoxon_timepoint_d10_vs_d5

> Source: `results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d10_vs_d5.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_10 vs day_5 (source: user-confirmed)

## User directives
- (triage, family-batched) "Confirm test type? Columns are median_day_10, median_day_5, log2_FC, U_stat, p_value, p_adj (day_10 vs day_5, n=4 vs 4 pooling 2 BWM + 2 control reps per timepoint)." → "Confirm Mann-Whitney U (all 6)"
- (triage, family-batched) "How firmly should these read? 0 FDR hits across the family; raw-p hits only at the n=4 MW floor (0.0286)." → "Mixed" (firm coordinated-floor-blocked-null headline; note floor-level cells as nominal-only exploratory leads; symmetric enriched/depleted per A.2.2)
- (triage, family-batched) "Per-CSV flags beyond the 4 locked family caveats?" → "Inline stop annotation + formal stop-codon-instability caveat on the codon files." Neither applies to this AA file — the amino-acid path drops stop windows, so this file has no stop codon (60 tests = 20 features × 3 sites).
- (triage, family-batched) "Spotlight any site/feature/group?" → "No spotlight"

## Headline
0 of 60 day_10-vs-day_5 amino-acid tests are significant at FDR<0.05; minimum adjusted p is 0.5714. A single cell reaches raw p<0.05 — P-site I (`log2_FC` -0.1469) — and it sits exactly at the n=4-vs-n=4 two-sided Mann-Whitney floor (0.02857). The absence of FDR hits is floor-blocking, not a biological negative. This is a coordinated floor-limited null. As a nominal-only exploratory lead (Mixed framing): P-site I (-0.1469, depleted in day_10) is the only nominally-significant cell; the largest-magnitude enriched cell, P-site G (+0.1524), sits just above the floor (raw p 0.0571); site E is the flattest site (every E cell has p_adj 1.0).

## Top hits

Tables are split per site (decoding-site order A / P / E), each showing the top-5 enriched and top-5 depleted cells by raw p (no FDR threshold is cleared, so selection is by raw p), displayed by `|log2_FC|` descending. The single `nominal-only` cell sits exactly at the n=4 two-sided MW floor (raw p 0.02857): its raw significance is the maximum the test can produce at this N, not evidence of a strong effect.

### Site A

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | G | +0.1299 | 0.8571 |  |
| enriched | W | +0.1063 | 0.9796 |  |
| enriched | R | +0.0463 | 0.8571 |  |
| enriched | C | +0.0361 | 0.8571 |  |
| enriched | S | +0.0152 | 0.9796 |  |
| depleted | K | -0.1202 | 0.8571 |  |
| depleted | Q | -0.1162 | 0.8571 |  |
| depleted | N | -0.0671 | 0.8571 |  |
| depleted | T | -0.0373 | 0.8571 |  |
| depleted | M | -0.0186 | 0.8571 |  |

<details>
<summary>Site P and Site E</summary>

### Site P

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | G | +0.1524 | 0.5714 |  |
| enriched | D | +0.1213 | 0.9143 |  |
| enriched | E | +0.0840 | 0.9143 |  |
| enriched | P | +0.0418 | 0.9143 |  |
| enriched | C | +0.0349 | 0.9143 |  |
| depleted | I | -0.1469 | 0.5714 | nominal-only |
| depleted | L | -0.0740 | 0.9143 |  |
| depleted | M | -0.0649 | 0.9143 |  |
| depleted | S | -0.0374 | 0.9143 |  |
| depleted | V | -0.0176 | 0.9143 |  |

### Site E

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | P | +0.0894 | 1.0000 |  |
| enriched | S | +0.0671 | 1.0000 |  |
| enriched | T | +0.0325 | 1.0000 |  |
| enriched | H | +0.0236 | 1.0000 |  |
| enriched | A | +0.0185 | 1.0000 |  |
| depleted | R | -0.0590 | 1.0000 |  |
| depleted | E | -0.0388 | 1.0000 |  |
| depleted | C | -0.0320 | 1.0000 |  |
| depleted | V | -0.0267 | 1.0000 |  |
| depleted | N | -0.0034 | 1.0000 |  |

</details>

## Numbers at a glance
- `n_tests`: 60 (20 per site, E/P/A)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.5714 (site P)
- `min raw p`: 0.0286 (= the n=4 floor; 1 cell, P-site I)
- `p_floor`: 0.02857 (theoretical two-sided Mann-Whitney floor for n=4 vs n=4 = 2/C(8,4)). One cell reaches it; site E does not reach it at all (min raw p there is higher, all p_adj 1.0).
- Per site: site A min p_adj 0.8571; site P min p_adj 0.5714; site E min p_adj 1.0000.

## Methods
Columns present: `site`, `amino_acid`, `median_day_10`, `median_day_5`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing day_10 vs day_5 occupancy per (site, amino acid) at n=4 vs n=4; user confirmed. The `U_stat` column and the paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's 20-feature family, per `merge_global_occupancy_analysis.py`). The test compares the two timepoint distributions per cell; it does not separate the BWM and control reps pooled into each n=4 arm, and at this N a single FDR-significant cell needs ~12+ features tied at the floor at once.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — the n=4-vs-n=4 two-sided MW floor is exactly 0.02857; with a per-site BH family of 20, FDR<0.05 needs k > 11.4 cells tied at the floor. Only one cell reaches the floor here, so FDR<0.05 is out of reach.
- **condition-pooled-confound** (family-wide) — each n=4 arm pools 2 BWM + 2 control reps; divergent BWM-vs-control day_10-vs-day_5 responses would be masked by pooling.
- **n=4-low-power** (family-wide) — n=4 vs n=4 MW has very low power; this null is weakly informative.
- **p-floor-aware-headline** (family-wide) — the headline states min raw p ≈ 0.0286 and why "no FDR hits" is uninformative here, not a biological negative.

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs to reconcile: the codon resolution of this contrast (`codon_wilcoxon_timepoint_d10_vs_d5.csv`) and the other two contrasts; see the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- This is the least-separated AA contrast in the family (only 1 floor cell; site E reaches p_adj 1.0 everywhere). Falsifier: does the within-condition Fisher d10_vs_d5 contrast also show the flattest signal of the three day-pairs, or does whole-transcriptome N reveal structure the n=4 MW could not?
- The single floor cell P-site I (-0.1469) — does it recur as a depletion in `*_d5_vs_d0` (which also shows a P-site Ile-adjacent pattern) or in the within-condition Fisher? Falsifier framing: a one-off floor cell that does not recur is most parsimoniously a discreteness artefact, not a day_10-vs-day_5 signal.
