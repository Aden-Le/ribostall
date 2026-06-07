---
input_csv: results/global_occupancy/analysis_corrected/codon_wilcoxon_timepoint_d5_vs_d0.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_5 vs day_0
test_type_source: user-confirmed
n_tests: 186
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.3543
p_floor: 0.02857
pseudoreplicated: false
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "stop-codon-instability", proposed_by: dylan, status: confirmed, why: "Codon set is data-driven (global_codon_occ.py builds ordered_codons from observed transcriptome codons, not SENSE_CODONS), so 62 codons/site = 61 sense + the in-frame stop TGA; the codon counting path does not drop stop windows (the AA path does, hence 60 vs 186 tests). Terminal stops are trimmed (run used trim_start=20, trim_stop=10), so this TGA is an in-frame stop (likely selenocysteine recoding / readthrough) with very low support and unstable log2_FC. User-confirmed."}
caveats_considered: []
headline: "0 of 186 day_5-vs-day_0 codon MW tests significant at FDR<0.05 (min p_adj 0.3543); 10 cells reach raw p<0.05, all at the n=4 MW floor (0.02857) and all depletions (day_5 < day_0), concentrated at the P site (5 of 10). Coordinated floor-limited null, the absence of FDR hits is floor-blocking not a biological negative. Largest floor depletions P-site CCA (-0.3507), P-site GGA (-0.3254); CCA/GGA/GCC recur as floor depletions across sites. Largest enriched swings are the unstable in-frame stop TGA (P-site +1.2366, E-site +0.9454)."
user_directives:
  - "(triage, family-batched) 'Confirm test type? Columns median_day_5/median_day_0/log2_FC/U_stat/p_value/p_adj at n=4 vs 4; my read two-sample Wilcoxon rank-sum (Mann-Whitney U), day_5 vs day_0.' -> 'Confirm Mann-Whitney U (all 6)'"
  - "(triage, family-batched) 'How firmly should these read? 0 FDR hits; raw-p hits only at the n=4 floor.' -> 'Mixed' (firm coordinated-floor-blocked-null headline; floor-level cells reported as nominal-only exploratory leads, symmetric on enriched/depleted)"
  - "(triage, family-batched) 'Per-CSV flags beyond the 4 family caveats?' -> 'Inline stop annotation AND formal stop-codon-instability caveat on the codon files' (both applied here)"
  - "(triage, family-batched) 'How do we have stop codons, I thought we did trimming?' -> Dylan answer (verified from code): trimming IS applied (run_global_codon_occ.sh trim_start=20, trim_stop=10), which removes the terminal stop; the TGA seen is an in-frame stop retained because the codon set is data-driven and the codon counting path does not drop stop windows. Captured as the stop-codon-instability caveat."
  - "(triage, family-batched) 'Spotlight any site/feature/group?' -> 'No spotlight'"
---

# Interpretation — codon_wilcoxon_timepoint_d5_vs_d0

> Source: `results/global_occupancy/analysis_corrected/codon_wilcoxon_timepoint_d5_vs_d0.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_5 vs day_0 (source: user-confirmed)

## User directives
- (triage, family-batched) "Confirm test type? Columns are median_day_5, median_day_0, log2_FC, U_stat, p_value, p_adj (day_5 vs day_0, n=4 vs 4 pooling 2 BWM + 2 control reps per timepoint)." → "Confirm Mann-Whitney U (all 6)"
- (triage, family-batched) "How firmly should these read? 0 FDR hits across the family; raw-p hits only at the n=4 MW floor (0.0286)." → "Mixed" (firm coordinated-floor-blocked-null headline; note floor-level cells as nominal-only exploratory leads; symmetric enriched/depleted per A.2.2)
- (triage, family-batched) "Per-CSV flags beyond the 4 locked family caveats?" → "Inline stop annotation AND a formal `stop-codon-instability` caveat on the codon files." Both applied here.
- (triage, family-batched) "How do we have stop codons, I thought we did trimming?" → Dylan answer, verified from code: trimming IS applied (`run_global_codon_occ.sh`: `trim_start=20, trim_stop=10`), removing the terminal stop. The TGA present is an in-frame stop retained because the codon set is data-driven (`global_codon_occ.py:205`) and the codon counting path does not drop stop windows. Recorded as `stop-codon-instability`.
- (triage, family-batched) "Spotlight any site/feature/group?" → "No spotlight"

## Headline
0 of 186 day_5-vs-day_0 codon tests are significant at FDR<0.05; minimum adjusted p is 0.3543. Ten cells reach raw p<0.05, all sitting exactly at the n=4-vs-n=4 two-sided Mann-Whitney floor (0.02857), and all ten are depletions (day_5 < day_0), with five of the ten at the P site. The absence of FDR hits is floor-blocking, not a biological negative. This is a coordinated floor-limited null — the family member with the most floor cells. As nominal-only exploratory leads (Mixed framing): the largest-magnitude floor depletions are P-site CCA (`log2_FC` -0.3507), P-site GGA (-0.3254) and E-site GGA (-0.3125); CCA, GGA and GCC recur as floor-level depletions across multiple sites. No enriched cell reaches the floor; the largest enriched raw swings are the unstable in-frame stop codon TGA (P-site +1.2366, E-site +0.9454; see caveat). The largest-magnitude sense codon overall, E-site CCA (-0.4795), sits just above the floor (raw p 0.0571).

## Top hits

Tables are split per site (decoding-site order A / P / E), each showing the top-5 enriched and top-5 depleted codons by raw p (no FDR threshold is cleared, so selection is by raw p), displayed by `|log2_FC|` descending. Every cell flagged `nominal-only` sits exactly at the n=4 two-sided MW floor (raw p 0.02857): raw significance there is the maximum the test can produce at this N. All floor cells in this file are depletions. The in-frame stop codon TGA is annotated inline where it appears; see the stop-codon caveat.

### Site A

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TGA (in-frame stop; unstable) | +0.4589 | 0.8857 |  |
| enriched | AAT | +0.2142 | 0.9126 |  |
| enriched | TGT | +0.1479 | 0.9126 |  |
| enriched | CGG | +0.1288 | 0.9242 |  |
| enriched | AGT | +0.1219 | 0.8857 |  |
| depleted | GCC | -0.4356 | 0.8267 |  |
| depleted | CCA | -0.3251 | 0.8267 | nominal-only |
| depleted | GTG | -0.2953 | 0.8267 |  |
| depleted | GGT | -0.2902 | 0.8267 |  |
| depleted | GGA | -0.2663 | 0.8267 | nominal-only |

<details>
<summary>Site P and Site E</summary>

### Site P

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TGA (in-frame stop; unstable) | +1.2366 | 1.0000 |  |
| enriched | TTA | +0.1570 | 1.0000 |  |
| enriched | GCG | +0.1242 | 1.0000 |  |
| enriched | ATA | +0.1167 | 1.0000 |  |
| enriched | CGG | +0.0981 | 1.0000 |  |
| depleted | CCA | -0.3507 | 0.3543 | nominal-only |
| depleted | GGA | -0.3254 | 0.3543 | nominal-only |
| depleted | AAG | -0.3021 | 0.3543 | nominal-only |
| depleted | GCC | -0.2565 | 0.3543 | nominal-only |
| depleted | AGA | -0.2283 | 0.3543 | nominal-only |

### Site E

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TGA (in-frame stop; unstable) | +0.9454 | 0.8604 |  |
| enriched | ATA | +0.1537 | 0.8604 |  |
| enriched | TCG | +0.1456 | 0.8604 |  |
| enriched | ACG | +0.1132 | 0.8604 |  |
| enriched | CGA | +0.0864 | 0.8604 |  |
| depleted | CCA | -0.4795 | 0.7086 |  |
| depleted | GGA | -0.3125 | 0.5905 | nominal-only |
| depleted | GCC | -0.2934 | 0.5905 | nominal-only |
| depleted | ACC | -0.2303 | 0.5905 | nominal-only |
| depleted | ATC | -0.1101 | 0.7086 |  |

</details>

## Numbers at a glance
- `n_tests`: 186 (62 per site, E/P/A; 62 = 61 sense codons + the in-frame stop TGA)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.3543 (site P, where 5 codons tie at the floor → 0.02857 × 62/5 ≈ 0.354)
- `min raw p`: 0.0286 (= the n=4 floor; 10 cells tie here, all depletions)
- `p_floor`: 0.02857 (theoretical two-sided Mann-Whitney floor for n=4 vs n=4 = 2/C(8,4)). Ten cells reach it — the most of any family member — yet a ~62-codon BH family still blocks FDR<0.05.
- Per site: site A min p_adj 0.8267; site P min p_adj 0.3543; site E min p_adj 0.5905.

## Methods
Columns present: `site`, `codon`, `median_day_5`, `median_day_0`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing day_5 vs day_0 occupancy per (site, codon) at n=4 vs n=4; user confirmed. The `U_stat` column and paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's ~62-codon family, per `merge_global_occupancy_analysis.py`). The test compares the two timepoint distributions per cell; it does not separate synonymous-codon effects from amino-acid-level effects, nor the pooled BWM/control reps, and the ~62-codon family makes FDR<0.05 require dozens of floor-tied cells at once.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — the n=4-vs-n=4 two-sided MW floor is 0.02857; the per-site BH family here is ~62 codons, so FDR<0.05 needs k > 35 cells at the floor at once. Ten cells reach the floor (5 at the P site), the most in the family, but still far short.
- **condition-pooled-confound** (family-wide) — each n=4 arm pools 2 BWM + 2 control reps; divergent BWM-vs-control day_5-vs-day_0 responses would be masked.
- **n=4-low-power** (family-wide) — n=4 vs n=4 MW has very low power; this null is weakly informative.
- **p-floor-aware-headline** (family-wide) — the headline states min raw p ≈ 0.0286 and that the 10 floor cells are all depletions, so "no FDR hits" is floor-blocking, not a biological negative.
- **stop-codon-instability** (per-CSV) — the 62 codons per site are 61 sense codons + the single in-frame stop **TGA** (TAA/TAG do not appear). The codon set is data-driven (`global_codon_occ.py:205`) and the codon counter does not drop stop windows (the AA path does, hence 60 vs 186 tests). Terminal stops are trimmed (`trim_start=20, trim_stop=10`), so TGA is an in-frame stop (likely selenocysteine recoding / readthrough) with very low support. Its `log2_FC` here is the largest enriched swing at every site (A +0.4589, P +1.2366, E +0.9454), which reflects that low support, not a day_5-vs-day_0 effect; all three rows are non-significant (raw p ≥ 0.34).

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs to reconcile: the AA resolution of this contrast (`aa_wilcoxon_timepoint_d5_vs_d0.csv`) and the other two contrasts; see the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- The floor depletions are dominated by a recurring set (CCA, GGA, GCC depleting across A/P/E) — all Pro/Gly/Ala-family codons. Falsifier: do CCA/GGA/GCC depletions reappear with consistent direction in the within-condition Fisher d5_vs_d0 at large N, or as a synonymous split (one codon depleting while a sister enriches)? A codon-cluster floor pattern with no Fisher echo is most parsimoniously a discreteness coincidence, not a day_5-vs-day_0 signal.
- The AA sister (`aa_wilcoxon_timepoint_d5_vs_d0`) shows P-site G depletion at the floor — does the codon-level GGA/GGT depletion here account for that AA-level Gly signal, and does either survive in the Fisher? Falsifier for whether the day_5-vs-day_0 Gly read is codon-specific or amino-acid-wide.
- Falsifier for stop-codon-instability: does TGA reach a comparable +log2_FC extreme in any large-N Fisher file, or only here where its low support inflates the ratio?
