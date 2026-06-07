---
input_csv: results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d5_vs_d0.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_5 vs day_0
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
headline: "0 of 60 day_5-vs-day_0 amino-acid MW tests significant at FDR<0.05 (min p_adj 0.2857); 5 cells reach raw p<0.05, all sitting exactly at the n=4 MW floor (0.02857); coordinated floor-limited null, the absence of FDR hits is floor-blocking not a biological negative. Largest-magnitude nominal-only leads: P-site G (-0.2029) depleted, A-site M (+0.1567) enriched; largest cell overall A-site W (-0.2413) sits above the floor (raw p 0.1143)."
user_directives:
  - "(triage, family-batched) 'Confirm test type? Columns median_day_5/median_day_0/log2_FC/U_stat/p_value/p_adj at n=4 vs 4; my read two-sample Wilcoxon rank-sum (Mann-Whitney U), day_5 vs day_0.' -> 'Confirm Mann-Whitney U (all 6)'"
  - "(triage, family-batched) 'How firmly should these read? 0 FDR hits; raw-p hits only at the n=4 floor.' -> 'Mixed' (firm coordinated-floor-blocked-null headline; floor-level cells reported as nominal-only exploratory leads, symmetric on enriched/depleted)"
  - "(triage, family-batched) 'Per-CSV flags beyond the 4 family caveats?' -> 'Inline stop annotation + formal stop-codon-instability caveat on codon files' (neither applies to this AA file: the AA path drops stop windows, so no stop codon is present here)"
  - "(triage, family-batched) 'Spotlight any site/feature/group?' -> 'No spotlight'"
---

# Interpretation — aa_wilcoxon_timepoint_d5_vs_d0

> Source: `results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d5_vs_d0.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_5 vs day_0 (source: user-confirmed)

## User directives
- (triage, family-batched) "Confirm test type? Columns are median_day_5, median_day_0, log2_FC, U_stat, p_value, p_adj (day_5 vs day_0, n=4 vs 4 pooling 2 BWM + 2 control reps per timepoint)." → "Confirm Mann-Whitney U (all 6)"
- (triage, family-batched) "How firmly should these read? 0 FDR hits across the family; raw-p hits only at the n=4 MW floor (0.0286)." → "Mixed" (firm coordinated-floor-blocked-null headline; note floor-level cells as nominal-only exploratory leads; symmetric enriched/depleted per A.2.2)
- (triage, family-batched) "Per-CSV flags beyond the 4 locked family caveats?" → "Inline stop annotation + formal stop-codon-instability caveat on the codon files." Neither applies to this AA file — the amino-acid path drops stop windows, so this file has no stop codon (60 tests = 20 features × 3 sites).
- (triage, family-batched) "Spotlight any site/feature/group?" → "No spotlight"

## Headline
0 of 60 day_5-vs-day_0 amino-acid tests are significant at FDR<0.05; minimum adjusted p is 0.2857. Five cells reach raw p<0.05, and all five sit exactly at the n=4-vs-n=4 two-sided Mann-Whitney floor (0.02857). The absence of FDR hits is a structural floor-blocking outcome, not a biological negative. This is a coordinated floor-limited null. As nominal-only exploratory leads (Mixed framing): the largest-magnitude floor cells are P-site G (`log2_FC` -0.2029, depleted in day_5) and A-site M (+0.1567, enriched); the largest-magnitude cell overall, A-site W (-0.2413), sits above the floor (raw p 0.1143).

## Top hits

Tables are split per site (decoding-site order A / P / E), each showing the top-5 enriched and top-5 depleted cells by raw p (no FDR threshold is cleared, so selection is by raw p), displayed by `|log2_FC|` descending. Every cell flagged `nominal-only` sits exactly at the n=4 two-sided MW floor (raw p 0.02857): its raw significance is the maximum the test can produce at this N, not evidence of a strong effect.

### Site A

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | N | +0.2178 | 0.6234 |  |
| enriched | M | +0.1567 | 0.2857 | nominal-only |
| enriched | D | +0.1073 | 0.6234 |  |
| enriched | F | +0.1019 | 0.6234 |  |
| enriched | S | +0.0597 | 0.4571 |  |
| depleted | W | -0.2413 | 0.4571 |  |
| depleted | G | -0.1770 | 0.4571 |  |
| depleted | A | -0.0903 | 0.6234 |  |
| depleted | P | -0.0866 | 0.2857 | nominal-only |
| depleted | L | -0.0425 | 0.6234 |  |

<details>
<summary>Site P and Site E</summary>

### Site P

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | C | +0.1311 | 0.5714 |  |
| enriched | E | +0.0936 | 0.8831 |  |
| enriched | N | +0.0905 | 0.8831 |  |
| enriched | S | +0.0783 | 0.2857 | nominal-only |
| enriched | L | +0.0604 | 0.8831 |  |
| depleted | G | -0.2029 | 0.2857 | nominal-only |
| depleted | K | -0.1539 | 0.8831 |  |
| depleted | P | -0.0888 | 0.5714 |  |
| depleted | R | -0.0654 | 0.8831 |  |
| depleted | A | -0.0437 | 0.9841 |  |

### Site E

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | S | +0.1506 | 0.6667 |  |
| enriched | C | +0.1122 | 0.6667 |  |
| enriched | M | +0.0727 | 0.5714 | nominal-only |
| enriched | L | +0.0725 | 0.8571 |  |
| enriched | H | +0.0450 | 0.6667 |  |
| depleted | P | -0.2195 | 0.6667 |  |
| depleted | K | -0.1014 | 0.8571 |  |
| depleted | G | -0.0976 | 0.6667 |  |
| depleted | I | -0.0516 | 1.0000 |  |
| depleted | Y | -0.0422 | 1.0000 |  |

</details>

## Numbers at a glance
- `n_tests`: 60 (20 per site, E/P/A)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.2857 (sites A and P)
- `min raw p`: 0.0286 (= the n=4 floor; 5 cells tie here)
- `p_floor`: 0.02857 (theoretical two-sided Mann-Whitney floor for n=4 vs n=4 = 2/C(8,4)). Five cells reach it, the most of the three AA contrasts, yet still 0 FDR hits.
- Per site: site A min p_adj 0.2857; site P min p_adj 0.2857; site E min p_adj 0.5714.

## Methods
Columns present: `site`, `amino_acid`, `median_day_5`, `median_day_0`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing day_5 vs day_0 occupancy per (site, amino acid) at n=4 vs n=4; user confirmed. The `U_stat` column and the paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's 20-feature family, per `merge_global_occupancy_analysis.py`). The test compares the two timepoint distributions per cell; it does not separate the BWM and control reps pooled into each n=4 arm, and at this N FDR<0.05 needs ~12+ features tied at the floor at once.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — the n=4-vs-n=4 two-sided MW floor is exactly 0.02857; with a per-site BH family of 20, FDR<0.05 needs k > 11.4 cells tied at the floor. Five cells reach the floor here, still far short.
- **condition-pooled-confound** (family-wide) — each n=4 arm pools 2 BWM + 2 control reps; divergent BWM-vs-control day_5-vs-day_0 responses would be masked by pooling.
- **n=4-low-power** (family-wide) — n=4 vs n=4 MW has very low power; this null is weakly informative.
- **p-floor-aware-headline** (family-wide) — the headline states min raw p ≈ 0.0286 and why "no FDR hits" is uninformative here, not a biological negative.

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs to reconcile: the codon resolution of this contrast (`codon_wilcoxon_timepoint_d5_vs_d0.csv`) and the other two contrasts; see the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- This contrast (and its codon sister) carries the most floor cells in the family, leaning toward depletions (P-site G, A-site P) and enrichments (A-site M, P-site S). Falsifier: do these same cells move with consistent direction and additive magnitude in `*_d10_vs_d0` (= d5_vs_d0 + d10_vs_d5), or is the apparent day_5-vs-day_0 structure non-additive / discreteness-driven?
- Falsifier on the within-condition Fisher: do P-site G depletion and A-site M enrichment reappear at FDR there (where N is large), or only here at the n=4 floor? Consistency across designs would let Chumeng decide whether to elevate; appearance only in the large-N Fisher would point to large-N amplification of a baseline difference.
