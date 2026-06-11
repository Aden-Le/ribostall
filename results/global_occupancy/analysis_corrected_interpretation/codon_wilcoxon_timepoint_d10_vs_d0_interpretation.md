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
synced_from_olive_qmd: 2026-06-07
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "stop-codon-instability", proposed_by: dylan, status: confirmed, why: "Codon set is data-driven (global_codon_occ.py builds ordered_codons from observed transcriptome codons, not SENSE_CODONS), so 62 codons/site = 61 sense + the in-frame stop TGA; the codon counting path does not drop stop windows (the AA path does, hence 60 vs 186 tests). Terminal stops are trimmed (run used trim_start=20, trim_stop=10), so this TGA is an in-frame stop (likely selenocysteine recoding / readthrough) with very low support and unstable log2_FC. User-confirmed."}
caveats_considered:
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: denied, why: "scipy's Mann-Whitney could fall back to the asymptotic Z approximation under tied per-replicate occupancy values, shifting raw p off the exact discrete bound. Ruled out empirically: the 186 tests collapse to 9 distinct p-values, U_stat is integer-valued (0-15), and all 9 fall on the n=4-vs-n=4 exact two-sided MW grid (k/70) -- the exact-branch signature with no ties. The identical n=4 design audited on the per-replicate sister data: 4 of 183 tests carry rank ties (all rare codons, far from raw p<0.05), method='auto' chose exact for 179 / asymptotic for 4, recomputed p matches the pipeline to ~1e-16 (max 8.3e-17), and forcing method='asymptotic' shifts raw p by at most 0.1021 (median 0.0061), flips 0 raw-p<0.05 decisions, and leaves the per-site BH conclusion unchanged (0 FDR hits either branch).", user_note: "Adopted into Dylan at Stage-7 readback; user-approved family-wide (mirrors CSV 1 + CSV 3)."}
headline: "0 of 186 day_10-vs-day_0 codon MW tests significant at FDR<0.05 (min p_adj 0.7873); 2 cells reach raw p<0.05, both at the n=4 MW floor (0.02857) and both E-site depletions (AGA -0.1710, AAG -0.1574); coordinated floor-limited null, the absence of FDR hits is floor-blocking not a biological negative. The in-frame stop codon TGA shows the largest raw swing (A-site -0.7716, raw p 0.6857) but is an unstable low-support cell."
user_directives:
  - "(triage, family-batched) 'Confirm test type? Columns median_day_10/median_day_0/log2_FC/U_stat/p_value/p_adj at n=4 vs 4; my read two-sample Wilcoxon rank-sum (Mann-Whitney U), day_10 vs day_0.' -> 'Confirm Mann-Whitney U (all 6)'"
  - "(triage, family-batched) 'How firmly should these read? 0 FDR hits; raw-p hits only at the n=4 floor.' -> 'Mixed' (firm coordinated-floor-blocked-null headline; floor-level cells reported as nominal-only exploratory leads, symmetric on enriched/depleted)"
  - "(triage, family-batched) 'Per-CSV flags beyond the 4 family caveats?' -> 'Inline stop annotation AND formal stop-codon-instability caveat on the codon files' (both applied here)"
  - "(triage, family-batched) 'How do we have stop codons, I thought we did trimming?' -> Dylan answer (verified from code): trimming IS applied (run_global_codon_occ.sh trim_start=20, trim_stop=10), which removes the terminal stop; the TGA seen is an in-frame stop retained because the codon set is data-driven and the codon counting path does not drop stop windows. Captured as the stop-codon-instability caveat."
  - "(triage, family-batched) 'Spotlight any site/feature/group?' -> 'No spotlight'"
  - "(readback) 'Reconciled shared content from the corrected .qmd on 2026-06-07' -> 'Adopted Olive 6 per-direction Top-hits sub-tables + raw p_value column (re-sorted raw-p asc, membership unchanged, <details> collapse of P/E kept); Headline largest-magnitude sense cells corrected to add A-site CGG +0.4724 and reworded nominal-only-exploratory-leads -> exploratory-leads-only; Numbers site order E/P/A -> A/P/E; stop-codon caveat bound raw p >= 0.69 -> >= 0.68; adopted the asymptotic-with-ties Considered-but-not-applicable entry (user-approved family-wide). Olive-only Biological interpretation/composite/plots not imported.'"
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
- (readback) "Reconciled shared content from the corrected `.qmd` on 2026-06-07." → Adopted Olive's six per-direction Top-hits sub-tables + the raw `p_value` column (re-sorted raw-p ascending, membership unchanged, `<details>` collapse of P/E kept); corrected the Headline largest-magnitude **sense** cells to include A-site CGG +0.4724 and reworded "nominal-only exploratory leads" → "exploratory leads only"; aligned the Numbers site order to A/P/E; fixed the stop-codon caveat bound "raw p ≥ 0.69" → "raw p ≥ 0.68"; adopted the `asymptotic-with-ties` Considered-but-not-applicable entry (user-approved family-wide, mirrors CSV 1 + CSV 3). Olive-only sections (Biological interpretation, composite figure, plots) were not imported.

## Headline
0 of 186 day_10-vs-day_0 codon tests are significant at FDR<0.05; minimum adjusted p is 0.7873. Two cells reach raw p<0.05, both sitting exactly at the n=4-vs-n=4 two-sided Mann-Whitney floor (0.02857), and both are E-site depletions (AGA `log2_FC` -0.1710, AAG -0.1574). The absence of FDR hits is floor-blocking, not a biological negative. This is a coordinated floor-limited null. As exploratory leads only (Mixed framing), the two E-site floor cells are the only nominally-significant cells; the largest-magnitude sense cells are A-site CGG (+0.4724) and P-site CGG (+0.4020) enriched and P-site AAG (-0.3690) depleted, all above the floor (raw p ≥ 0.057). The in-frame stop codon TGA shows the single largest raw swing in the file (A-site -0.7716, raw p 0.6857) but is an unstable low-support cell — see caveat; it does not rank into the raw-p tables below.

## Top hits

Tables are split per (site, direction) in decoding-site order A / P / E, each showing up to 5 enriched / 5 depleted codons ranked by raw `p_value` ascending with `|log2_FC|` descending as the tiebreaker (no FDR threshold is cleared, so selection is by raw p). Both p columns are shown: `p_value` is the raw Mann-Whitney p; `p_adj` is BH-corrected within each site's ~62-codon family. The `aa` column is the single-letter amino-acid translation of the codon. Cells flagged `nominal-only` sit exactly at the n=4 two-sided MW floor (raw p 0.02857): raw significance there is the maximum the test can produce at this N, not a strong effect. The in-frame stop codon TGA does not appear in any sub-table here (its raw p is high); see the stop-codon caveat.

### A site — Enriched (day_10 > day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GGG | G | +0.2325 | 0.2000 | 0.9806 |  |
| TGT | C | +0.2712 | 0.3429 | 0.9806 |  |
| TCG | S | +0.2337 | 0.3429 | 0.9806 |  |
| AGT | S | +0.2250 | 0.3429 | 0.9806 |  |
| GAT | D | +0.1721 | 0.3429 | 0.9806 |  |

### A site — Depleted (day_10 < day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CAA | Q | -0.1541 | 0.0571 | 0.9806 |  |
| GGT | G | -0.2130 | 0.1143 | 0.9806 |  |
| AAG | K | -0.1804 | 0.2000 | 0.9806 |  |
| GCC | A | -0.3546 | 0.3429 | 0.9806 |  |
| CTC | L | -0.2327 | 0.3429 | 0.9806 |  |

<details>
<summary>Site P and Site E</summary>

### P site — Enriched (day_10 > day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CGG | R | +0.4020 | 0.0571 | 0.7873 |  |
| GCG | A | +0.1691 | 0.2000 | 0.7925 |  |
| CCT | P | +0.3172 | 0.3429 | 0.7925 |  |
| CCG | P | +0.3146 | 0.3429 | 0.7925 |  |
| CCC | P | +0.1990 | 0.3429 | 0.7925 |  |

### P site — Depleted (day_10 < day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CCA | P | -0.2365 | 0.0571 | 0.7873 |  |
| ATC | I | -0.1259 | 0.0571 | 0.7873 |  |
| AAG | K | -0.3690 | 0.1143 | 0.7873 |  |
| AGA | R | -0.2401 | 0.1143 | 0.7873 |  |
| CTT | L | -0.1739 | 0.1143 | 0.7873 |  |

### E site — Enriched (day_10 > day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| AGT | S | +0.1412 | 0.0571 | 0.9242 |  |
| ACG | T | +0.2424 | 0.1143 | 0.9242 |  |
| TCG | S | +0.2659 | 0.2000 | 0.9242 |  |
| CTG | L | +0.1889 | 0.3429 | 0.9242 |  |
| CGG | R | +0.1761 | 0.3429 | 0.9242 |  |

### E site — Depleted (day_10 < day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| AGA | R | -0.1710 | 0.0286 | 0.8857 | nominal-only |
| AAG | K | -0.1574 | 0.0286 | 0.8857 | nominal-only |
| GAG | E | -0.3067 | 0.1143 | 0.9242 |  |
| GGA | G | -0.3024 | 0.1143 | 0.9242 |  |
| CCA | P | -0.2881 | 0.2000 | 0.9242 |  |

</details>

## Numbers at a glance
- `n_tests`: 186 (62 per site, A/P/E; 62 = 61 sense codons + the in-frame stop TGA)
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
- **stop-codon-instability** (per-CSV) — the 62 codons per site are 61 sense codons + the single in-frame stop **TGA** (TAA/TAG do not appear). The codon set is data-driven (`ordered_codons = sorted(set(transcriptome_codon_counts))`, `global_codon_occ.py:205`), and the codon counter `iter_trimmed_site_counts` does not drop stop windows (the AA path's `aggregate_to_aa` does, which is why the AA files carry 60 tests vs 186 here). Terminal stops are removed by trimming (`run_global_codon_occ.sh`: `trim_start=20, trim_stop=10`), so TGA here is an **in-frame** stop within the elongation body — most plausibly selenocysteine recoding or stop-codon readthrough in a small set of transcripts. Its footprint support is therefore very low, so its `log2_FC` (A-site -0.7716, P-site -0.0242, E-site +0.1319 in this file) is unstable and should be read as noise, not biology. TGA is non-significant here (all three site rows have raw p ≥ 0.68) and does not enter the Top-hits tables.

### Considered but not applicable
- **asymptotic-with-ties** (per-CSV) — concern that scipy's Mann-Whitney falls back to the asymptotic Z approximation under tied per-replicate occupancy, shifting raw p off the exact discrete bound. Ruled out empirically: the 186 tests collapse to 9 distinct p-values, every `U_stat` is integer-valued (0-15), and all 9 fall on the n=4-vs-n=4 exact two-sided MW grid (k/70) — the exact-branch signature with no ties. The identical n=4 design was audited on the per-replicate sister data (day_10 vs day_0): 4 of 183 tests carry rank ties (all rare codons, far from raw p<0.05), `method='auto'` picked exact for 179 / asymptotic for 4, recomputed p matches the pipeline to ~1e-16 (max 8.3e-17), and forcing `method='asymptotic'` shifts raw p by at most 0.1021 (median 0.0061), flips zero raw-p<0.05 decisions, and leaves the per-site BH conclusion unchanged (0 FDR hits on either branch). Raised but ruled out — user-approved family-wide (mirrors CSV 1 + CSV 3).

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs to reconcile: the AA resolution of this contrast (`aa_wilcoxon_timepoint_d10_vs_d0.csv`) and the other two contrasts; see the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- The two E-site floor depletions are AGA and AAG (both Arg/Lys-family codons). Falsifier: is an AA-level signal at E-site R or K visible in the AA sister (it is not — both AA files are floor-blocked nulls), and does either codon recur as a depletion in the within-condition Fisher d10_vs_d0 at large N? A codon-only floor cell with no AA-level echo and no Fisher echo is most parsimoniously a synonymous-discreteness artefact.
- Does the in-frame stop TGA show a comparable extreme in any large-N Fisher file, or only in these low-N Wilcoxon files where its low support inflates `log2_FC`? Falsifier for the stop-codon-instability reading.
