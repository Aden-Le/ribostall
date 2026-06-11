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
synced_from_olive_qmd: 2026-06-08
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "stop-codon-instability", proposed_by: dylan, status: confirmed, why: "Codon set is data-driven (global_codon_occ.py builds ordered_codons from observed transcriptome codons, not SENSE_CODONS), so 62 codons/site = 61 sense + the in-frame stop TGA; the codon counting path does not drop stop windows (the AA path does, hence 60 vs 186 tests). Terminal stops are trimmed (run used trim_start=20, trim_stop=10), so this TGA is an in-frame stop (likely selenocysteine recoding / readthrough) with very low support and unstable log2_FC. User-confirmed."}
caveats_considered:
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: denied, why: "scipy's Mann-Whitney could fall back to the asymptotic Z approximation under tied per-replicate occupancy values, shifting raw p off the exact discrete bound. Ruled out empirically: the 186 tests collapse to just 8 distinct p-values, U_stat is integer-valued (4-15), and all 8 fall on the n=4-vs-n=4 exact two-sided MW grid (k/70: 4/70, 8/70, 14/70, 24/70, 34/70, 48/70, 62/70, 70/70) -- the exact-branch signature with no ties. The identical n=4 design audited on the per-replicate sister data (day_10 vs day_5): 2 of 183 tests carry rank ties (both rare codons, far from raw p<0.05), method='auto' chose exact for 181 / asymptotic for 2, recomputed p matches the pipeline to ~1e-16 (max 8.3e-17), and forcing method='asymptotic' shifts raw p by at most 0.0759 (median 0.0061), flips 0 raw-p<0.05 decisions, and leaves the per-site BH conclusion unchanged (0 FDR hits either branch).", user_note: "Adopted into Dylan at Stage-7 readback; user-approved family-wide (mirrors CSV 1 + CSV 3)."}
headline: "0 of 186 day_10-vs-day_5 codon MW tests significant at FDR<0.05 (min p_adj 0.9984); no cell reaches even raw p<0.05 (min raw p 0.0571) — the flattest file in the family, no cell achieving the n=4 floor (0.02857). Coordinated null with no nominally-significant cells. Largest raw swings are the in-frame stop codon TGA (P-site -1.2608, A-site -1.2306, E-site -0.8135; unstable low-support); among sense codons the largest are P-site CCC (+0.3614) and A-site GGG (+0.3535)."
user_directives:
  - "(triage, family-batched) 'Confirm test type? Columns median_day_10/median_day_5/log2_FC/U_stat/p_value/p_adj at n=4 vs 4; my read two-sample Wilcoxon rank-sum (Mann-Whitney U), day_10 vs day_5.' -> 'Confirm Mann-Whitney U (all 6)'"
  - "(triage, family-batched) 'How firmly should these read? 0 FDR hits; this file does not even reach the n=4 floor.' -> 'Mixed' (firm coordinated-null headline; flag the flattest-of-family status; no nominal-only cells to surface)"
  - "(triage, family-batched) 'Per-CSV flags beyond the 4 family caveats?' -> 'Inline stop annotation AND formal stop-codon-instability caveat on the codon files' (both applied here). The offered `sub-floor-flat` note was not selected as a caveat; the sub-floor fact is reported as a plain number, not a confirmed caveat."
  - "(triage, family-batched) 'How do we have stop codons, I thought we did trimming?' -> Dylan answer (verified from code): trimming IS applied (run_global_codon_occ.sh trim_start=20, trim_stop=10), which removes the terminal stop; the TGA seen is an in-frame stop retained because the codon set is data-driven and the codon counting path does not drop stop windows. Captured as the stop-codon-instability caveat."
  - "(triage, family-batched) 'Spotlight any site/feature/group?' -> 'No spotlight'"
  - "(readback) 'Reconciled shared content from the corrected .qmd on 2026-06-08' -> 'Adopted Olive 6 per-direction Top-hits sub-tables + raw p_value column (re-sorted raw-p asc, membership unchanged, <details> collapse of P/E kept); Headline largest raw swings now lead with P-site TGA -1.2608 and largest sense cells corrected to P-site CCC +0.3614 / A-site GGG +0.3535 (was A-GGG / P-CGG +0.3039); stop-codon caveat now names all three TGA cells (P -1.2608 added; both -> all three non-significant); Numbers site order E/P/A -> A/P/E; adopted the asymptotic-with-ties Considered-but-not-applicable entry (user-approved family-wide). Olive-only Biological interpretation/composite/plots not imported.'"
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
- (readback) "Reconciled shared content from the corrected `.qmd` on 2026-06-08." → Adopted Olive's six per-direction Top-hits sub-tables + the raw `p_value` column (re-sorted raw-p ascending, membership unchanged, `<details>` collapse of P/E kept); corrected the Headline so the largest raw swings overall lead with P-site TGA -1.2608 and the largest **sense** cells read P-site CCC +0.3614 / A-site GGG +0.3535 (was A-site GGG / P-site CGG +0.3039); updated the `stop-codon-instability` caveat to name all three TGA cells (added P-site -1.2608; "both" → "all three" non-significant); aligned the Numbers site order to A/P/E; adopted the `asymptotic-with-ties` Considered-but-not-applicable entry (user-approved family-wide, mirrors CSV 1 + CSV 3). Olive-only sections (Biological interpretation, composite figure, plots) were not imported.

## Headline
0 of 186 day_10-vs-day_5 codon tests are significant at FDR<0.05; minimum adjusted p is 0.9984. No cell reaches even raw p<0.05 (min raw p = 0.0571): this is the flattest file in the family — not a single cell achieves the n=4-vs-n=4 perfect-rank-separation floor (0.02857). It is a coordinated null with no nominally-significant cells. As leads (Mixed framing): the largest raw swings overall are the in-frame stop codon TGA (P-site `log2_FC` -1.2608, A-site -1.2306, E-site -0.8135), which are unstable low-support cells (see caveat); among sense codons the largest cells are P-site CCC (+0.3614) and A-site GGG (+0.3535), all with raw p ≥ 0.057.

## Top hits

Tables are split per (site, direction) in decoding-site order A / P / E, each showing up to 5 enriched / 5 depleted codons ranked by raw `p_value` ascending with `|log2_FC|` descending as the tiebreaker (no FDR threshold is cleared and no cell reaches even raw p<0.05, so selection is by raw p, mostly carried by the magnitude tiebreak). Both p columns are shown: `p_value` is the raw Mann-Whitney p; `p_adj` is BH-corrected within each site's ~62-codon family. The `aa` column is the single-letter amino-acid translation of the codon (the in-frame stop TGA shows as `*`). No cell is `nominal-only` — none reaches raw p<0.05. The in-frame stop codon TGA is annotated inline where it ranks into a sub-table (A-site and E-site Depleted); its large `log2_FC` reflects very low footprint support, not a day_10-vs-day_5 effect — see the stop-codon caveat.

### A site — Enriched (day_10 > day_5)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GGG | G | +0.3535 | 0.2000 | 1.0000 |  |
| CCG | P | +0.2255 | 0.2000 | 1.0000 |  |
| CGG | R | +0.3436 | 0.3429 | 1.0000 |  |
| GGC | G | +0.3001 | 0.3429 | 1.0000 |  |
| TCG | S | +0.2491 | 0.3429 | 1.0000 |  |

### A site — Depleted (day_10 < day_5)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CAA | Q | -0.1101 | 0.3429 | 1.0000 |  |
| TGA (in-frame stop; unstable) | * | -1.2306 | 0.4857 | 1.0000 |  |
| AAC | N | -0.0786 | 0.6857 | 1.0000 |  |
| AAG | K | -0.0173 | 0.6857 | 1.0000 |  |
| GTT | V | -0.0295 | 0.8857 | 1.0000 |  |

<details>
<summary>Site P and Site E</summary>

### P site — Enriched (day_10 > day_5)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| AGG | R | +0.2967 | 0.0571 | 0.9984 |  |
| GGG | G | +0.2902 | 0.1143 | 0.9984 |  |
| GCA | A | +0.2267 | 0.1143 | 0.9984 |  |
| CGG | R | +0.3039 | 0.2000 | 0.9984 |  |
| CCT | P | +0.2760 | 0.2000 | 0.9984 |  |

### P site — Depleted (day_10 < day_5)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CTT | L | -0.1592 | 0.3429 | 0.9984 |  |
| ATC | I | -0.0641 | 0.4857 | 0.9984 |  |
| TTG | L | -0.1027 | 0.6857 | 0.9984 |  |
| ATG | M | -0.0986 | 0.6857 | 0.9984 |  |
| ATA | I | -0.0640 | 0.6857 | 0.9984 |  |

### E site — Enriched (day_10 > day_5)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CCC | P | +0.0998 | 0.2000 | 1.0000 |  |
| CTC | L | +0.2516 | 0.3429 | 1.0000 |  |
| CCG | P | +0.2393 | 0.3429 | 1.0000 |  |
| CCT | P | +0.2950 | 0.4857 | 1.0000 |  |
| AGT | S | +0.1335 | 0.4857 | 1.0000 |  |

### E site — Depleted (day_10 < day_5)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GAG | E | -0.1997 | 0.6857 | 1.0000 |  |
| ATT | I | -0.0618 | 0.6857 | 1.0000 |  |
| TAT | Y | -0.0239 | 0.6857 | 1.0000 |  |
| TTG | L | -0.0218 | 0.6857 | 1.0000 |  |
| TGA (in-frame stop; unstable) | * | -0.8135 | 0.8857 | 1.0000 |  |

</details>

## Numbers at a glance
- `n_tests`: 186 (62 per site, A/P/E; 62 = 61 sense codons + the in-frame stop TGA)
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
- **stop-codon-instability** (per-CSV) — the 62 codons per site are 61 sense codons + the single in-frame stop **TGA** (TAA/TAG do not appear). The codon set is data-driven (`global_codon_occ.py:205`), and the codon counter does not drop stop windows (the AA path does, hence 60 vs 186 tests). Terminal stops are trimmed (`trim_start=20, trim_stop=10`), so TGA is an in-frame stop (likely selenocysteine recoding / readthrough) with very low support. Its `log2_FC` here is the largest in the file (P-site -1.2608, A-site -1.2306, E-site -0.8135), which reflects that low support, not a day_10-vs-day_5 effect; all three rows are non-significant (p_adj 1.0).

### Considered but not applicable
- **asymptotic-with-ties** (per-CSV) — concern that scipy's Mann-Whitney falls back to the asymptotic Z approximation under tied per-replicate occupancy, shifting raw p off the exact discrete bound. Ruled out empirically: the 186 tests collapse to just 8 distinct p-values, every `U_stat` is integer-valued (4-15), and all 8 fall on the n=4-vs-n=4 exact two-sided MW grid (k/70: 4/70, 8/70, 14/70, 24/70, 34/70, 48/70, 62/70, 70/70) — the exact-branch signature with no ties. The identical n=4 design was audited on the per-replicate sister data (day_10 vs day_5): 2 of 183 tests carry rank ties (both rare codons, far from raw p<0.05), `method='auto'` picked exact for 181 / asymptotic for 2, recomputed p matches the pipeline to ~1e-16 (max 8.3e-17), and forcing `method='asymptotic'` shifts raw p by at most 0.0759 (median 0.0061), flips zero raw-p<0.05 decisions, and leaves the per-site BH conclusion unchanged (0 FDR hits on either branch). Raised but ruled out — user-approved family-wide (mirrors CSV 1 + CSV 3).

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs to reconcile: the AA resolution of this contrast (`aa_wilcoxon_timepoint_d10_vs_d5.csv`) and the other two contrasts; see the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- This is the flattest file in the family at both resolutions (the AA sister is also the flattest AA contrast). Falsifier: does the within-condition Fisher d10_vs_d5 contrast also come out flattest of the three day-pairs, supporting a genuinely small day_10-vs-day_5 occupancy difference, or does large-N Fisher surface structure the n=4 MW could not?
- The largest swings in the file are the in-frame stop TGA at all three sites (P -1.2608, A -1.2306, E -0.8135). Falsifier for stop-codon-instability: does TGA reach a comparable extreme in any large-N Fisher file, or only here where its low support inflates `log2_FC`?
