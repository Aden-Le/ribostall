---
input_csv: results/global_occupancy/analysis_corrected/codon_wilcoxon_timepoint_d10_vs_d5.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_10 vs day_5
test_type_source: user-confirmed
n_tests: 186
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.9984
p_floor: 0.02857
pseudoreplicated: false
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "stop-codon-instability", proposed_by: dylan, status: confirmed, why: "Codon set is data-driven (global_codon_occ.py builds ordered_codons from observed transcriptome codons, not SENSE_CODONS), so 62 codons/site = 61 sense + the in-frame stop TGA; the codon counting path does not drop stop windows (the AA path does, hence 60 vs 186 tests). Terminal stops are trimmed (run used trim_start=20, trim_stop=10), so this TGA is an in-frame stop (likely selenocysteine recoding / readthrough) with very low support and unstable log2_FC. User-confirmed."}
caveats_considered: []
headline: "0 of 186 day_10-vs-day_5 codon MW tests significant at FDR<0.05 (min p_adj 0.9984); no cell reaches even raw p<0.05 (min raw p 0.0571) — the flattest file in the family, no cell achieving the n=4 floor (0.02857). Coordinated null with no nominally-significant cells. Largest raw swings are the in-frame stop codon TGA (A-site -1.2306, E-site -0.8135; unstable low-support); among sense codons the largest are A-site GGG (+0.3535) and P-site CGG (+0.3039)."
user_directives:
  - "(triage, family-batched) 'Confirm test type? Columns median_day_10/median_day_5/log2_FC/U_stat/p_value/p_adj at n=4 vs 4; my read two-sample Wilcoxon rank-sum (Mann-Whitney U), day_10 vs day_5.' -> 'Confirm Mann-Whitney U (all 6)'"
  - "(triage, family-batched) 'How firmly should these read? 0 FDR hits; this file does not even reach the n=4 floor.' -> 'Mixed' (firm coordinated-null headline; flag the flattest-of-family status; no nominal-only cells to surface)"
  - "(triage, family-batched) 'Per-CSV flags beyond the 4 family caveats?' -> 'Inline stop annotation AND formal stop-codon-instability caveat on the codon files' (both applied here). The offered `sub-floor-flat` note was not selected as a caveat; the sub-floor fact is reported as a plain number, not a confirmed caveat."
  - "(triage, family-batched) 'How do we have stop codons, I thought we did trimming?' -> Dylan answer (verified from code): trimming IS applied (run_global_codon_occ.sh trim_start=20, trim_stop=10), which removes the terminal stop; the TGA seen is an in-frame stop retained because the codon set is data-driven and the codon counting path does not drop stop windows. Captured as the stop-codon-instability caveat."
  - "(triage, family-batched) 'Spotlight any site/feature/group?' -> 'No spotlight'"
---

# Interpretation — codon_wilcoxon_timepoint_d10_vs_d5

> Source: `results/global_occupancy/analysis_corrected/codon_wilcoxon_timepoint_d10_vs_d5.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_10 vs day_5 (source: user-confirmed)

## User directives
- (triage, family-batched) "Confirm test type? Columns are median_day_10, median_day_5, log2_FC, U_stat, p_value, p_adj (day_10 vs day_5, n=4 vs 4 pooling 2 BWM + 2 control reps per timepoint)." → "Confirm Mann-Whitney U (all 6)"
- (triage, family-batched) "How firmly should these read? 0 FDR hits; this file does not even reach the n=4 floor (min raw p 0.0571)." → "Mixed" (firm coordinated-null headline; flag the flattest-of-family status; there are no nominal-only cells to surface)
- (triage, family-batched) "Per-CSV flags beyond the 4 locked family caveats?" → "Inline stop annotation AND a formal `stop-codon-instability` caveat on the codon files." Both applied. The separately-offered `sub-floor-flat` note was not selected as a caveat; the sub-floor fact (min raw p 0.0571 > 0.02857) is reported as a plain number, not a confirmed caveat.
- (triage, family-batched) "How do we have stop codons, I thought we did trimming?" → Dylan answer, verified from code: trimming IS applied (`run_global_codon_occ.sh`: `trim_start=20, trim_stop=10`), removing the terminal stop. The TGA present is an in-frame stop retained because the codon set is data-driven (`global_codon_occ.py:205`) and the codon counting path does not drop stop windows. Recorded as `stop-codon-instability`.
- (triage, family-batched) "Spotlight any site/feature/group?" → "No spotlight"

## Headline
0 of 186 day_10-vs-day_5 codon tests are significant at FDR<0.05; minimum adjusted p is 0.9984. No cell reaches even raw p<0.05 (min raw p = 0.0571): this is the flattest file in the family — not a single cell achieves the n=4-vs-n=4 perfect-rank-separation floor (0.02857). It is a coordinated null with no nominally-significant cells. As leads (Mixed framing): the largest raw swings are the in-frame stop codon TGA (A-site `log2_FC` -1.2306, E-site -0.8135), which are unstable low-support cells (see caveat); among sense codons the largest cells are A-site GGG (+0.3535) and P-site CGG (+0.3039), all with raw p ≥ 0.057.

## Top hits

Tables are split per site (decoding-site order A / P / E), each showing the top-5 enriched and top-5 depleted codons by raw p (no FDR threshold is cleared and no cell reaches the n=4 floor, so selection is by raw p), displayed by `|log2_FC|` descending. No cell here is `nominal-only` — none reaches raw p<0.05. The in-frame stop codon TGA is annotated inline where it appears; see the stop-codon caveat.

### Site A

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GGG | +0.3535 | 1.0000 |  |
| enriched | CGG | +0.3436 | 1.0000 |  |
| enriched | GGC | +0.3001 | 1.0000 |  |
| enriched | TCG | +0.2491 | 1.0000 |  |
| enriched | CCG | +0.2255 | 1.0000 |  |
| depleted | TGA (in-frame stop; unstable) | -1.2306 | 1.0000 |  |
| depleted | CAA | -0.1101 | 1.0000 |  |
| depleted | AAC | -0.0786 | 1.0000 |  |
| depleted | GTT | -0.0295 | 1.0000 |  |
| depleted | AAG | -0.0173 | 1.0000 |  |

<details>
<summary>Site P and Site E</summary>

### Site P

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | CGG | +0.3039 | 0.9984 |  |
| enriched | AGG | +0.2967 | 0.9984 |  |
| enriched | GGG | +0.2902 | 0.9984 |  |
| enriched | CCT | +0.2760 | 0.9984 |  |
| enriched | GCA | +0.2267 | 0.9984 |  |
| depleted | CTT | -0.1592 | 0.9984 |  |
| depleted | TTG | -0.1027 | 0.9984 |  |
| depleted | ATG | -0.0986 | 0.9984 |  |
| depleted | ATC | -0.0641 | 0.9984 |  |
| depleted | ATA | -0.0640 | 0.9984 |  |

### Site E

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | CCT | +0.2950 | 1.0000 |  |
| enriched | CTC | +0.2516 | 1.0000 |  |
| enriched | CCG | +0.2393 | 1.0000 |  |
| enriched | AGT | +0.1335 | 1.0000 |  |
| enriched | CCC | +0.0998 | 1.0000 |  |
| depleted | TGA (in-frame stop; unstable) | -0.8135 | 1.0000 |  |
| depleted | GAG | -0.1997 | 1.0000 |  |
| depleted | ATT | -0.0618 | 1.0000 |  |
| depleted | TAT | -0.0239 | 1.0000 |  |
| depleted | TTG | -0.0218 | 1.0000 |  |

</details>

## Numbers at a glance
- `n_tests`: 186 (62 per site, E/P/A; 62 = 61 sense codons + the in-frame stop TGA)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.9984 (site P; sites A and E are 1.0000)
- `min raw p`: 0.0571 — the only family member whose minimum raw p sits *above* the n=4 floor (0.02857). No cell achieves perfect 4-vs-4 rank separation.
- `p_floor`: 0.02857 (theoretical two-sided Mann-Whitney floor for n=4 vs n=4 = 2/C(8,4)); not reached in this file.
- Per site: site A min p_adj 1.0000; site P min p_adj 0.9984; site E min p_adj 1.0000.

## Methods
Columns present: `site`, `codon`, `median_day_10`, `median_day_5`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing day_10 vs day_5 occupancy per (site, codon) at n=4 vs n=4; user confirmed. The `U_stat` column and paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's ~62-codon family, per `merge_global_occupancy_analysis.py`). The test compares the two timepoint distributions per cell; here no cell even reaches the discreteness floor, so the file carries no nominal signal at all.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — the n=4-vs-n=4 two-sided MW floor is 0.02857; in this file no cell even reaches the floor, so the floor is not the binding constraint here — the data are simply flat.
- **condition-pooled-confound** (family-wide) — each n=4 arm pools 2 BWM + 2 control reps; divergent responses would be masked.
- **n=4-low-power** (family-wide) — n=4 vs n=4 MW has very low power; this null is weakly informative.
- **p-floor-aware-headline** (family-wide) — the headline states min raw p = 0.0571 and notes this is the only family member not to reach the floor.
- **stop-codon-instability** (per-CSV) — the 62 codons per site are 61 sense codons + the single in-frame stop **TGA** (TAA/TAG do not appear). The codon set is data-driven (`global_codon_occ.py:205`), and the codon counter does not drop stop windows (the AA path does, hence 60 vs 186 tests). Terminal stops are trimmed (`trim_start=20, trim_stop=10`), so TGA is an in-frame stop (likely selenocysteine recoding / readthrough) with very low support. Its `log2_FC` here is the largest in the file (A-site -1.2306, E-site -0.8135), which reflects that low support, not a day_10-vs-day_5 effect; both rows are non-significant (p_adj 1.0).

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs to reconcile: the AA resolution of this contrast (`aa_wilcoxon_timepoint_d10_vs_d5.csv`) and the other two contrasts; see the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- This is the flattest file in the family at both resolutions (the AA sister is also the flattest AA contrast). Falsifier: does the within-condition Fisher d10_vs_d5 contrast also come out flattest of the three day-pairs, supporting a genuinely small day_10-vs-day_5 occupancy difference, or does large-N Fisher surface structure the n=4 MW could not?
- The two largest swings are the in-frame stop TGA at A and E. Falsifier for stop-codon-instability: does TGA reach a comparable extreme in any large-N Fisher file, or only here where its low support inflates `log2_FC`?
