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
synced_from_olive_qmd: 2026-06-11
caveats:
  - {label: "timepoint-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "mw-floor-tight", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "n=6-modest-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "bh-per-site", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
caveats_considered:
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: denied, why: "Concern that scipy.stats.mannwhitneyu could fall back to the asymptotic Z approximation under tied per-replicate occupancy values, shifting raw p away from the exact discrete bound (~0.00216 floor).", user_note: "Denied — ruled out empirically. This file directly: the 186 (site, codon) tests collapse to 10 distinct p-values, every U_stat is integer-valued (range 9-26), and all 10 distinct p-values fall on the exact n=6 vs 6 two-sided MW grid (exact branch, no ties); all 186 cells share p_adj 1.0000. The identical n=6 vs 6 design audited on the per-replicate sister data (_for_claude_mw_branch_audit.py): 4/183 rank ties (all rare codons, far from raw p<0.05), method='auto' picked exact for 179 / asymptotic for 4, recomputed raw p matches the pipeline to ~1e-16, and forcing method='asymptotic' shifts raw p by at most 0.0328 (median 0.0080), flips 0 raw-p<0.05 decisions, and leaves the per-site BH conclusion unchanged (0 hits at FDR<0.05 either way; asymptotic marginally more conservative). Audit-sourced, not CSV-verifiable."}
headline: "0 of 186 codon tests significant at FDR<0.05 (and none at raw p<0.05); min p_adj = 1.0, min raw p = 0.1797 — a clean coordinated null at codon resolution; largest-magnitude cells are the stop codon TGA (A-site +0.8516, E-site +0.6565, both p_adj 1.0 and likely low-count noise) and, among sense codons, E-site GAG (-0.1926) and A-site CCG (+0.2279)."
user_directives:
  - "(triage) 'Confirm test type? Filename wilcoxon; columns median_BWM/median_control/log2_FC/U_stat/p_value/p_adj (n=6 vs 6).' -> 'Confirm Mann-Whitney U'"
  - "(triage) 'How firmly should this read? 0 hits at FDR and raw-p.' -> 'Mixed' (firm-null headline, note 1-2 largest-magnitude cells as non-significant leads)"
  - "(triage) 'Any CSV-specific caveat beyond the 4 family caveats?' -> 'None — family caveats suffice'"
  - "(triage) 'Spotlight any site/feature/group?' -> 'No spotlight'"
  - "(readback 2026-06-11) 'Reconciled shared content from the corrected .qmd' -> 'Top hits adopted Olive 6 per-(site,direction) sub-tables + aa column + raw p_value column (re-sorted raw-p asc, <details> collapse of P/E kept, bare TGA|* with the low-count caution moved to headline/intro prose); headline sense leads corrected A-site TTC (-0.1693) -> E-site GAG (-0.1926)/P-site GAG (-0.1768)/A-site CCG (+0.2279); Numbers site order E/P/A -> A/P/E; adopted the asymptotic-with-ties Considered-but-not-applicable entry (user-approved family-wide, mirrors the aa sister). Olive-only sections not imported.'"
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
- (readback 2026-06-11) Top hits reconciled to Olive's `.qmd` table structure (six per-(site, direction) sub-tables A/P/E x Enriched/Depleted + an `aa` column + the raw `p_value` column, re-sorted by raw p ascending; the `<details>` collapse of P/E kept; the stop codon TGA shown as a bare `TGA | *` with no inline annotation, its low-count caution kept in the Headline and Top-hits intro prose); the Headline's sense-codon leads corrected from A-site TTC (-0.1693) to E-site GAG (-0.1926) / P-site GAG (-0.1768) / A-site CCG (+0.2279) to match the corrected `.qmd`; the Numbers site order aligned E/P/A → A/P/E; adopted the `asymptotic-with-ties` Considered-but-not-applicable entry (user-approved family-wide, mirrors the aa sister). Dylan conventions kept (terse headline, Methods provenance, Confirmed/Considered caveats); Olive-only sections (Composite, Overview, Biological interpretation, Plots) not imported. Provenance in front-matter `synced_from_olive_qmd`.

## Headline
0 of 186 codon tests are significant at FDR<0.05, and none clear raw p<0.05 (min raw p = 0.1797); minimum adjusted p is 1.0. This is a coordinated null at codon resolution. As non-significant leads only (Mixed framing): the largest-magnitude cells are the stop codon TGA (A-site `log2_FC` +0.8516; E-site +0.6565; both p_adj 1.0000), which carry low footprint counts and most plausibly reflect noise rather than biology; among sense codons the largest cells are E-site GAG (-0.1926), P-site GAG (-0.1768), and A-site CCG (+0.2279).

## Top hits

Tables are split per (site, direction) in decoding-site order A / P / E, each showing up to 5 enriched / 5 depleted codons ranked by raw `p_value` ascending with `|log2_FC|` descending as the tiebreaker. Both p columns are shown: `p_value` is the raw Mann-Whitney p; `p_adj` is BH-corrected within each site's ~62-codon family. The `aa` column is the single-letter amino-acid translation of each codon; the in-frame stop TGA shows as `*`, and its large `log2_FC` reflects very low footprint support, not a BWM-vs-control effect (see Headline). No row reaches FDR<0.05 or raw p<0.05 (min raw p 0.1797), so every cell is a non-significant lead; `flag` is blank because no per-row floor / nominal-only condition is met.

### A site — Enriched (BWM > control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| TGA | * | +0.8516 | 0.3095 | 1.0000 |  |
| CCG | P | +0.2279 | 0.5887 | 1.0000 |  |
| CTA | L | +0.1246 | 0.5887 | 1.0000 |  |
| GGG | G | +0.0811 | 0.5887 | 1.0000 |  |
| GTC | V | +0.0353 | 0.5887 | 1.0000 |  |

### A site — Depleted (BWM < control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GAG | E | -0.1561 | 0.2403 | 1.0000 |  |
| TTC | F | -0.1693 | 0.3095 | 1.0000 |  |
| GTG | V | -0.1376 | 0.3095 | 1.0000 |  |
| GAA | E | -0.1343 | 0.3095 | 1.0000 |  |
| GAC | D | -0.0889 | 0.3095 | 1.0000 |  |

<details>
<summary>Site P and Site E</summary>

### P site — Enriched (BWM > control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CGG | R | +0.1259 | 0.3939 | 1.0000 |  |
| AGG | R | +0.1081 | 0.3939 | 1.0000 |  |
| TTA | L | +0.1207 | 0.4848 | 1.0000 |  |
| CTG | L | +0.0947 | 0.4848 | 1.0000 |  |
| GGC | G | +0.0620 | 0.4848 | 1.0000 |  |

### P site — Depleted (BWM < control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GAG | E | -0.1768 | 0.2403 | 1.0000 |  |
| TGC | C | -0.0691 | 0.2403 | 1.0000 |  |
| AGC | S | -0.0370 | 0.3095 | 1.0000 |  |
| GTG | V | -0.1057 | 0.4848 | 1.0000 |  |
| AAC | N | -0.1019 | 0.4848 | 1.0000 |  |

### E site — Enriched (BWM > control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| TGA | * | +0.6565 | 0.2403 | 1.0000 |  |
| TGC | C | +0.0736 | 0.5887 | 1.0000 |  |
| TGT | C | +0.0735 | 0.6991 | 1.0000 |  |
| GCA | A | +0.0320 | 0.6991 | 1.0000 |  |
| GTC | V | +0.1044 | 0.8182 | 1.0000 |  |

### E site — Depleted (BWM < control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| AAC | N | -0.0968 | 0.1797 | 1.0000 |  |
| CAG | Q | -0.0862 | 0.2403 | 1.0000 |  |
| GAG | E | -0.1926 | 0.3095 | 1.0000 |  |
| AAG | K | -0.1407 | 0.3095 | 1.0000 |  |
| AGA | R | -0.1068 | 0.3095 | 1.0000 |  |

</details>

## Numbers at a glance
- `n_tests`: 186 (62 per site, A/P/E)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 1.0 (all 186 cells tied at this value)
- `min raw p`: 0.1797
- `p_floor`: 0.00216 (theoretical two-sided Mann-Whitney floor for n=6 vs n=6). Observed min raw p (0.1797) sits far above this floor, so the null here is signal/power-limited, not floor-limited.
- Per site: every site min p_adj = 1.0000 (after BH, all 62 cells within each site share that site's adjusted p of 1.0).

## Methods
Columns present: `site`, `codon`, `median_BWM`, `median_control`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing BWM vs control occupancy per (site, codon); user confirmed. The `U_stat` column and the paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's ~62-codon family, per `merge_global_occupancy_analysis.py`). The test compares the two condition distributions per cell; it does not address timepoint structure (the 6 reps per condition span day_0/day_5/day_10) and does not separate synonymous-codon effects from amino-acid-level effects.

## Caveats
### Confirmed
- **timepoint-pooled-confound** (family-wide) — n=6 per condition treats the 6 reps across day_0/day_5/day_10 as replicates of one condition; if the BWM effect varies across timepoints, pooling can mask signal. Applies to both family members.
- **mw-floor-tight** (family-wide) — the theoretical two-sided MW floor for n=6 vs n=6 is ~0.00216; per-site BH families of ~62 make FDR<0.05 feasible but tight. Here the data does not approach the floor (min raw p 0.1797), so the constraint is not binding for this file.
- **n=6-modest-power** (family-wide) — Mann-Whitney with n=6 vs n=6 has modest power; this null is weakly informative.
- **bh-per-site** (family-wide) — `p_adj` is BH-corrected within each site's ~62-codon family, not across the merged E/P/A file.

### Considered but not applicable
- **asymptotic-with-ties** (per-CSV) — flagged that scipy's Mann-Whitney could fall back to the asymptotic Z approximation under tied per-replicate occupancy values, shifting raw p off the exact discrete bound (~0.00216 floor); ruled out empirically. This file directly: the 186 (site, codon) tests collapse to 10 distinct p-values, every `U_stat` is integer-valued (range 9-26), and all 10 distinct p-values fall on the exact n=6 vs 6 two-sided MW grid — the exact-branch, no-ties signature; all 186 cells share `p_adj` 1.0000. The identical n=6 vs 6 design audited on the per-replicate sister data (`_for_claude_mw_branch_audit.py`): 4 of 183 tests carry rank ties (all rare codons, all far from raw p<0.05), `method='auto'` selected the exact branch for 179 tests and the asymptotic branch for those 4, recomputed raw p matches the pipeline to ~1e-16, and forcing `method='asymptotic'` for all tests shifts raw p by at most 0.0328 (median 0.0080), flips 0 raw-p<0.05 decisions, and leaves the per-site BH conclusion unchanged (0 hits at FDR<0.05 either way; the asymptotic branch is marginally more conservative). Audit-sourced (`_for_claude_mw_branch_audit.py`), not CSV-verifiable.

## For Chumeng (joint-reading hooks)
- Family: `between_condition_wilcoxon` — sister CSV to reconcile: `aa_wilcoxon_condition.csv` (amino-acid resolution of the same BWM-vs-control contrast).
- Is the codon-level null an aggregate of synonyms moving together, or are any synonyms split (one codon enriched while its sister depletes at the same site)? GAG depletes at all three sites here (A -0.1561, P -0.1768, E -0.1926, all p_adj 1.0) — does this consistent-direction E/P/A pattern reappear in `per_timepoint_fisher_codon` at any timepoint, or is it sampling coherence with no significance behind it?
- Does the stop codon TGA's large +log2_FC at A and E reflect anything beyond low-count instability? Falsifier: does TGA show a comparable extreme in any large-N Fisher file, or only here where counts are smallest?
- Where this codon MW says null, does the aa-level sister file agree (it does — both 0 FDR hits), and do both nulls coexist with per-timepoint Fisher signal that the timepoint-pooled MW would have washed out?
