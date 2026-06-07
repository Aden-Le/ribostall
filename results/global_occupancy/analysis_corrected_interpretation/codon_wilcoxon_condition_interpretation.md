---
input_csv: results/global_occupancy/analysis_corrected/codon_wilcoxon_condition.csv
family: between_condition_wilcoxon
test_type: Mann-Whitney U (two-sample Wilcoxon rank-sum), BWM vs control
test_type_source: user-confirmed
n_tests: 186
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 1.0
p_floor: 0.00216
pseudoreplicated: false
caveats:
  - {label: "timepoint-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "mw-floor-tight", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "n=6-modest-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "bh-per-site", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
caveats_considered: []
headline: "0 of 186 codon tests significant at FDR<0.05 (and none at raw p<0.05); min p_adj = 1.0, min raw p = 0.1797 — a clean coordinated null at codon resolution; largest-magnitude cells are the stop codon TGA (A-site +0.8516, E-site +0.6565, both p_adj 1.0 and likely low-count noise) and, among sense codons, A-site TTC (-0.1693) and E-site GAG (-0.1926)."
user_directives:
  - "(triage) 'Confirm test type? Filename wilcoxon; columns median_BWM/median_control/log2_FC/U_stat/p_value/p_adj (n=6 vs 6).' -> 'Confirm Mann-Whitney U'"
  - "(triage) 'How firmly should this read? 0 hits at FDR and raw-p.' -> 'Mixed' (firm-null headline, note 1-2 largest-magnitude cells as non-significant leads)"
  - "(triage) 'Any CSV-specific caveat beyond the 4 family caveats?' -> 'None — family caveats suffice'"
  - "(triage) 'Spotlight any site/feature/group?' -> 'No spotlight'"
---

# Interpretation — codon_wilcoxon_condition

> Source: `results/global_occupancy/analysis_corrected/codon_wilcoxon_condition.csv`
> Family: `between_condition_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U (two-sample Wilcoxon rank-sum), BWM vs control (source: user-confirmed)

## User directives
- (triage) "Confirm test type? Filename suggests wilcoxon; columns are median_BWM, median_control, log2_FC, U_stat, p_value, p_adj (BWM vs control, n=6 vs 6 pooled across timepoints)." → "Confirm Mann-Whitney U"
- (triage) "How firmly should this read? 0 hits at FDR<0.05 and at raw p<0.05 (min raw p = 0.1797)." → "Mixed" (firm-null headline, note 1-2 largest-magnitude cells as non-significant leads)
- (triage) "Any caveat unique to this file beyond the 4 locked family caveats?" → "None — family caveats suffice"
- (triage) "Spotlight any site/feature/group?" → "No spotlight"

## Headline
0 of 186 codon tests are significant at FDR<0.05, and none clear raw p<0.05 (min raw p = 0.1797); minimum adjusted p is 1.0. This is a coordinated null at codon resolution. As non-significant leads only (Mixed framing): the largest-magnitude cells are the stop codon TGA (A-site `log2_FC` +0.8516; E-site +0.6565; both p_adj 1.0000), which carry low footprint counts and most plausibly reflect noise rather than biology; among sense codons the largest cells are A-site TTC (-0.1693), E-site GAG (-0.1926), and P-site GAG (-0.1768).

## Top hits

Tables are split per site (decoding-site order A / P / E) with the largest-magnitude enriched and depleted codons in each. No row reaches FDR<0.05 or raw p<0.05, so every cell below is a non-significant lead; `flag` is left blank because no per-row floor / nominal-only condition is met. The stop codon TGA is retained as the literal largest-magnitude cell but is flagged inline as likely low-count noise.

### Site A

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TGA (stop; likely low-count noise) | +0.8516 | 1.0000 |  |
| enriched | CCG | +0.2279 | 1.0000 |  |
| enriched | CTA | +0.1246 | 1.0000 |  |
| enriched | GGG | +0.0811 | 1.0000 |  |
| enriched | GTC | +0.0353 | 1.0000 |  |
| depleted | TTC | -0.1693 | 1.0000 |  |
| depleted | GAG | -0.1561 | 1.0000 |  |
| depleted | GTG | -0.1376 | 1.0000 |  |
| depleted | GAA | -0.1343 | 1.0000 |  |
| depleted | GAC | -0.0889 | 1.0000 |  |

<details>
<summary>Site P and Site E</summary>

### Site P

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | CGG | +0.1259 | 1.0000 |  |
| enriched | TTA | +0.1207 | 1.0000 |  |
| enriched | AGG | +0.1081 | 1.0000 |  |
| enriched | CTG | +0.0947 | 1.0000 |  |
| enriched | GGC | +0.0620 | 1.0000 |  |
| depleted | GAG | -0.1768 | 1.0000 |  |
| depleted | GTG | -0.1057 | 1.0000 |  |
| depleted | AAC | -0.1019 | 1.0000 |  |
| depleted | TGC | -0.0691 | 1.0000 |  |
| depleted | AGC | -0.0370 | 1.0000 |  |

### Site E

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TGA (stop; likely low-count noise) | +0.6565 | 1.0000 |  |
| enriched | GTC | +0.1044 | 1.0000 |  |
| enriched | TGC | +0.0736 | 1.0000 |  |
| enriched | TGT | +0.0735 | 1.0000 |  |
| enriched | GCA | +0.0320 | 1.0000 |  |
| depleted | GAG | -0.1926 | 1.0000 |  |
| depleted | AAG | -0.1407 | 1.0000 |  |
| depleted | AGA | -0.1068 | 1.0000 |  |
| depleted | AAC | -0.0968 | 1.0000 |  |
| depleted | CAG | -0.0862 | 1.0000 |  |

</details>

## Numbers at a glance
- `n_tests`: 186 (62 per site, E/P/A)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 1.0
- `min raw p`: 0.1797
- `p_floor`: 0.00216 (theoretical two-sided Mann-Whitney floor for n=6 vs n=6). Observed min raw p (0.1797) sits far above this floor, so the null here is signal/power-limited, not floor-limited.
- Per site: every site min p_adj = 1.0000.

## Methods
Columns present: `site`, `codon`, `median_BWM`, `median_control`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing BWM vs control occupancy per (site, codon); user confirmed. The `U_stat` column and the paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's ~62-codon family, per `merge_global_occupancy_analysis.py`). The test compares the two condition distributions per cell; it does not address timepoint structure (the 6 reps per condition span day_0/day_5/day_10) and does not separate synonymous-codon effects from amino-acid-level effects.

## Caveats
### Confirmed
- **timepoint-pooled-confound** (family-wide) — n=6 per condition treats the 6 reps across day_0/day_5/day_10 as replicates of one condition; if the BWM effect varies across timepoints, pooling can mask signal. Applies to both family members.
- **mw-floor-tight** (family-wide) — the theoretical two-sided MW floor for n=6 vs n=6 is ~0.00216; per-site BH families of ~62 make FDR<0.05 feasible but tight. Here the data does not approach the floor (min raw p 0.1797), so the constraint is not binding for this file.
- **n=6-modest-power** (family-wide) — Mann-Whitney with n=6 vs n=6 has modest power; this null is weakly informative.
- **bh-per-site** (family-wide) — `p_adj` is BH-corrected within each site's ~62-codon family, not across the merged E/P/A file.

## For Chumeng (joint-reading hooks)
- Family: `between_condition_wilcoxon` — sister CSV to reconcile: `aa_wilcoxon_condition.csv` (amino-acid resolution of the same BWM-vs-control contrast).
- Is the codon-level null an aggregate of synonyms moving together, or are any synonyms split (one codon enriched while its sister depletes at the same site)? GAG depletes at all three sites here (A -0.1561, P -0.1768, E -0.1926, all p_adj 1.0) — does this consistent-direction E/P/A pattern reappear in `per_timepoint_fisher_codon` at any timepoint, or is it sampling coherence with no significance behind it?
- Does the stop codon TGA's large +log2_FC at A and E reflect anything beyond low-count instability? Falsifier: does TGA show a comparable extreme in any large-N Fisher file, or only here where counts are smallest?
- Where this codon MW says null, does the aa-level sister file agree (it does — both 0 FDR hits), and do both nulls coexist with per-timepoint Fisher signal that the timepoint-pooled MW would have washed out?
