---
input_csv: results/global_occupancy/analysis_corrected/codon_wilcoxon_timepoint_d10_vs_d0.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_10 vs day_0
test_type_source: user-confirmed
n_tests: 186
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.7873
p_floor: 0.02857
pseudoreplicated: false
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "stop-codon-instability", proposed_by: dylan, status: confirmed, why: "Codon set is data-driven (global_codon_occ.py builds ordered_codons from observed transcriptome codons, not SENSE_CODONS), so 62 codons/site = 61 sense + the in-frame stop TGA; the codon counting path does not drop stop windows (the AA path does, hence 60 vs 186 tests). Terminal stops are trimmed (run used trim_start=20, trim_stop=10), so this TGA is an in-frame stop (likely selenocysteine recoding / readthrough) with very low support and unstable log2_FC. User-confirmed."}
caveats_considered: []
headline: "0 of 186 day_10-vs-day_0 codon MW tests significant at FDR<0.05 (min p_adj 0.7873); 2 cells reach raw p<0.05, both at the n=4 MW floor (0.02857) and both E-site depletions (AGA -0.1710, AAG -0.1574); coordinated floor-limited null, the absence of FDR hits is floor-blocking not a biological negative. The in-frame stop codon TGA shows the largest raw swing (A-site -0.7716, raw p 0.6857) but is an unstable low-support cell."
user_directives:
  - "(triage, family-batched) 'Confirm test type? Columns median_day_10/median_day_0/log2_FC/U_stat/p_value/p_adj at n=4 vs 4; my read two-sample Wilcoxon rank-sum (Mann-Whitney U), day_10 vs day_0.' -> 'Confirm Mann-Whitney U (all 6)'"
  - "(triage, family-batched) 'How firmly should these read? 0 FDR hits; raw-p hits only at the n=4 floor.' -> 'Mixed' (firm coordinated-floor-blocked-null headline; floor-level cells reported as nominal-only exploratory leads, symmetric on enriched/depleted)"
  - "(triage, family-batched) 'Per-CSV flags beyond the 4 family caveats?' -> 'Inline stop annotation AND formal stop-codon-instability caveat on the codon files' (both applied here)"
  - "(triage, family-batched) 'How do we have stop codons, I thought we did trimming?' -> Dylan answer (verified from code): trimming IS applied (run_global_codon_occ.sh trim_start=20, trim_stop=10), which removes the terminal stop; the TGA seen is an in-frame stop retained because the codon set is data-driven and the codon counting path does not drop stop windows. Captured as the stop-codon-instability caveat."
  - "(triage, family-batched) 'Spotlight any site/feature/group?' -> 'No spotlight'"
---

# Interpretation — codon_wilcoxon_timepoint_d10_vs_d0

> Source: `results/global_occupancy/analysis_corrected/codon_wilcoxon_timepoint_d10_vs_d0.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_10 vs day_0 (source: user-confirmed)

## User directives
- (triage, family-batched) "Confirm test type? Columns are median_day_10, median_day_0, log2_FC, U_stat, p_value, p_adj (day_10 vs day_0, n=4 vs 4 pooling 2 BWM + 2 control reps per timepoint)." → "Confirm Mann-Whitney U (all 6)"
- (triage, family-batched) "How firmly should these read? 0 FDR hits across the family; raw-p hits only at the n=4 MW floor (0.0286)." → "Mixed" (firm coordinated-floor-blocked-null headline; note floor-level cells as nominal-only exploratory leads; symmetric enriched/depleted per A.2.2)
- (triage, family-batched) "Per-CSV flags beyond the 4 locked family caveats?" → "Inline stop annotation AND a formal `stop-codon-instability` caveat on the codon files." Both applied here.
- (triage, family-batched) "How do we have stop codons, I thought we did trimming?" → Dylan answer, verified from code: trimming IS applied (`run_global_codon_occ.sh`: `trim_start=20, trim_stop=10`), removing the terminal stop. The TGA present is an in-frame stop retained because the codon set is built data-driven (`global_codon_occ.py:205`) and the codon counting path (`iter_trimmed_site_counts`) does not drop stop windows. Recorded as the `stop-codon-instability` caveat.
- (triage, family-batched) "Spotlight any site/feature/group?" → "No spotlight"

## Headline
0 of 186 day_10-vs-day_0 codon tests are significant at FDR<0.05; minimum adjusted p is 0.7873. Two cells reach raw p<0.05, both sitting exactly at the n=4-vs-n=4 two-sided Mann-Whitney floor (0.02857), and both are E-site depletions (AGA `log2_FC` -0.1710, AAG -0.1574). The absence of FDR hits is floor-blocking, not a biological negative. This is a coordinated floor-limited null. As nominal-only exploratory leads (Mixed framing), the two E-site floor cells are the only nominally-significant cells; the largest-magnitude cells overall (P-site AAG -0.3690, P-site CGG +0.4020) sit above the floor (raw p ≥ 0.057). The in-frame stop codon TGA shows the single largest raw swing in the file (A-site -0.7716, raw p 0.6857) but is an unstable low-support cell — see caveat; it does not rank into the raw-p tables below.

## Top hits

Tables are split per site (decoding-site order A / P / E), each showing the top-5 enriched and top-5 depleted codons by raw p (no FDR threshold is cleared, so selection is by raw p), displayed by `|log2_FC|` descending. Cells flagged `nominal-only` sit exactly at the n=4 two-sided MW floor (raw p 0.02857): raw significance there is the maximum the test can produce at this N, not a strong effect. The in-frame stop codon TGA does not appear in any sub-table here (its raw p is high); see the stop-codon caveat.

### Site A

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TGT | +0.2712 | 0.9806 |  |
| enriched | TCG | +0.2337 | 0.9806 |  |
| enriched | GGG | +0.2325 | 0.9806 |  |
| enriched | AGT | +0.2250 | 0.9806 |  |
| enriched | GAT | +0.1721 | 0.9806 |  |
| depleted | GCC | -0.3546 | 0.9806 |  |
| depleted | CTC | -0.2327 | 0.9806 |  |
| depleted | GGT | -0.2130 | 0.9806 |  |
| depleted | AAG | -0.1804 | 0.9806 |  |
| depleted | CAA | -0.1541 | 0.9806 |  |

<details>
<summary>Site P and Site E</summary>

### Site P

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | CGG | +0.4020 | 0.7873 |  |
| enriched | CCT | +0.3172 | 0.7925 |  |
| enriched | CCG | +0.3146 | 0.7925 |  |
| enriched | CCC | +0.1990 | 0.7925 |  |
| enriched | GCG | +0.1691 | 0.7925 |  |
| depleted | AAG | -0.3690 | 0.7873 |  |
| depleted | AGA | -0.2401 | 0.7873 |  |
| depleted | CCA | -0.2365 | 0.7873 |  |
| depleted | CTT | -0.1739 | 0.7873 |  |
| depleted | ATC | -0.1259 | 0.7873 |  |

### Site E

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TCG | +0.2659 | 0.9242 |  |
| enriched | ACG | +0.2424 | 0.9242 |  |
| enriched | CTG | +0.1889 | 0.9242 |  |
| enriched | CGG | +0.1761 | 0.9242 |  |
| enriched | AGT | +0.1412 | 0.9242 |  |
| depleted | GAG | -0.3067 | 0.9242 |  |
| depleted | GGA | -0.3024 | 0.9242 |  |
| depleted | CCA | -0.2881 | 0.9242 |  |
| depleted | AGA | -0.1710 | 0.8857 | nominal-only |
| depleted | AAG | -0.1574 | 0.8857 | nominal-only |

</details>

## Numbers at a glance
- `n_tests`: 186 (62 per site, E/P/A; 62 = 61 sense codons + the in-frame stop TGA)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.7873 (site P)
- `min raw p`: 0.0286 (= the n=4 floor; 2 cells, both E-site)
- `p_floor`: 0.02857 (theoretical two-sided Mann-Whitney floor for n=4 vs n=4 = 2/C(8,4)). Two cells reach it; with a ~62-codon BH family the bar is even higher than the AA files'.
- Per site: site A min p_adj 0.9806; site P min p_adj 0.7873; site E min p_adj 0.8857.

## Methods
Columns present: `site`, `codon`, `median_day_10`, `median_day_0`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing day_10 vs day_0 occupancy per (site, codon) at n=4 vs n=4; user confirmed. The `U_stat` column and the paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's ~62-codon family, per `merge_global_occupancy_analysis.py`). The test compares the two timepoint distributions per cell; it does not separate synonymous-codon effects from amino-acid-level effects, does not separate the pooled BWM/control reps, and at this N a ~62-codon family makes FDR<0.05 require dozens of cells tied at the floor at once.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — the n=4-vs-n=4 two-sided MW floor is exactly 0.02857; the per-site BH family here is ~62 codons, so FDR<0.05 needs p_adj = 0.02857 × 62/k < 0.05, i.e. k > 35 cells at the floor at once. Two cells reach the floor, so FDR<0.05 is unreachable.
- **condition-pooled-confound** (family-wide) — each n=4 arm pools 2 BWM + 2 control reps; divergent responses would be masked.
- **n=4-low-power** (family-wide) — n=4 vs n=4 MW has very low power; this null is weakly informative.
- **p-floor-aware-headline** (family-wide) — the headline states min raw p ≈ 0.0286 and why "no FDR hits" is uninformative here.
- **stop-codon-instability** (per-CSV) — the 62 codons per site are 61 sense codons + the single in-frame stop **TGA** (TAA/TAG do not appear). The codon set is data-driven (`ordered_codons = sorted(set(transcriptome_codon_counts))`, `global_codon_occ.py:205`), and the codon counter `iter_trimmed_site_counts` does not drop stop windows (the AA path's `aggregate_to_aa` does, which is why the AA files carry 60 tests vs 186 here). Terminal stops are removed by trimming (`run_global_codon_occ.sh`: `trim_start=20, trim_stop=10`), so TGA here is an **in-frame** stop within the elongation body — most plausibly selenocysteine recoding or stop-codon readthrough in a small set of transcripts. Its footprint support is therefore very low, so its `log2_FC` (A-site -0.7716, P-site -0.0242, E-site +0.1319 in this file) is unstable and should be read as noise, not biology. TGA is non-significant here (all three site rows have raw p ≥ 0.69) and does not enter the Top-hits tables.

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs to reconcile: the AA resolution of this contrast (`aa_wilcoxon_timepoint_d10_vs_d0.csv`) and the other two contrasts; see the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- The two E-site floor depletions are AGA and AAG (both Arg/Lys-family codons). Falsifier: is an AA-level signal at E-site R or K visible in the AA sister (it is not — both AA files are floor-blocked nulls), and does either codon recur as a depletion in the within-condition Fisher d10_vs_d0 at large N? A codon-only floor cell with no AA-level echo and no Fisher echo is most parsimoniously a synonymous-discreteness artefact.
- Does the in-frame stop TGA show a comparable extreme in any large-N Fisher file, or only in these low-N Wilcoxon files where its low support inflates `log2_FC`? Falsifier for the stop-codon-instability reading.
