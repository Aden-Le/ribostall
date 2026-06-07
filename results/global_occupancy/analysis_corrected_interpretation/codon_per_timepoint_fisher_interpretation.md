---
input_csv: results/global_occupancy/analysis_corrected/codon_per_timepoint_fisher.csv
family: per_timepoint_fisher
test_type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, codon); BH-FDR within each (timepoint, site) family of 62 codons (61 sense + in-frame stop TGA)
test_type_source: user-confirmed
n_tests: 558
n_significant_fdr05: 445
n_significant_fdr10: 466
min_p_adj: 0.0
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "bh-per-(timepoint,site)", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "imbalanced-N", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "Each per-(timepoint, site) BH family is 62 codons (61 sense + the in-frame stop TGA), ~3x the aa sister's 20-AA families. AA-level signals can split across synonyms below the per-(timepoint, site) FDR threshold here; a feature significant in the aa file but absent here is an aggregation effect, not a contradiction. User-confirmed."}
  - {label: "n-asymmetry-mild", proposed_by: dylan, status: confirmed, why: "Same near-balanced totals as the aa sister (ctrl:BWM 1.59 / 1.20 / 0.81 across day_0/day_5/day_10; BWM is the larger arm by day_10). Per-(timepoint,site) n_sig is high at every timepoint (38/62 to 57/62), with no day_0 power gradient. The locked imbalanced-N caveat is mild here. User-confirmed."}
  - {label: "flip-sig-large-N-artifact", proposed_by: dylan, status: confirmed, why: "114 of the 150 cross-timepoint direction-flip cells register at p_adj<0.05 on both opposite-sign rows. At whole-transcriptome pooled N almost any small per-cell sign change clears FDR, so the high sig-flip count is N-driven, not 114 biological reversals. Rank flip cells by |log2_OR| range, not by #sig. User-confirmed."}
  - {label: "stop-codon-instability", proposed_by: dylan, status: confirmed, why: "The 62 codons/site are 61 sense + the single in-frame stop TGA (TAA/TAG absent; codon set is data-driven and the codon counting path does not drop stop windows, unlike the AA path). Terminal stops are trimmed in the run, so this TGA is an in-frame stop (likely selenocysteine recoding / readthrough) with very low support. It surfaces as the largest |log2_OR| at day_10 in all 3 sites (A +2.1403, E +2.1611, P +1.6835) but on tiny counts; the apparent day_10 enrichment is low-count instability, not a sense-codon signal. Matches the done between_*_wilcoxon codon files in this folder. User-confirmed."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "Exactly 9 of 558 rows fall below BWM_count<100 OR control_count<100, and ALL 9 are the in-frame stop TGA (3 sites x 3 timepoints); every sense codon has k in the thousands. TGA's day_10 spike (k_BWM 79-103, k_ctrl 16-20) gives unstable OR/p; rows flagged `rare-codon` coincide exactly with `stop-codon-instability`. User-confirmed."}
caveats_considered:
  - {label: "OR-direction-anchor", proposed_by: dylan, status: not-adopted, why: "Proposed stating the OR>1=BWM-enriched / OR<1=BWM-depleted convention as a formal caveat.", user_note: "Folded into the log2_OR display directive (sign encodes direction); stated in Methods/Top-hits rather than carried as a standalone caveat."}
  - {label: "small-bh-family-discreteness", proposed_by: dylan, status: denied, why: "Sense codons have per-cell k in the thousands so Fisher p-values are effectively continuous; no coarse BH clustering. The only low-count codon (TGA) is covered by rare-codon-low-count / stop-codon-instability.", user_note: "Recorded as not applicable for the sense codons."}
headline: "558 BWM-vs-control Fisher tests (3 timepoints x 3 sites x 62 codons = 61 sense + the in-frame stop TGA): 445/558 clear FDR<0.05 (79.7%), 466/558 FDR<0.10; min p_adj underflows to 0.0 (3 cells), 253/558 cells at p_adj<1e-10. As in the aa sister, at whole-transcriptome N FDR significance is near-universal and p magnitude is uninformative; rank by |log2_OR|, which for sense codons is small-to-moderate (largest sense enrichment day_0 A:CGG +0.5100 ~1.42x; largest sense depletion day_5 E:CGG -0.3649). Per-(timepoint,site) n_sig is high at all timepoints (38/62 day_10 P to 57/62 day_0 A), no day_0 gradient (totals near-balanced 1.59/1.20/0.81). The in-frame stop TGA is the single largest |log2_OR| at day_10 in all 3 sites (A +2.1403, E +2.1611, P +1.6835) but is the only low-count codon (k_BWM 79-103, k_ctrl 16-20) — flagged rare-codon + stop, not a sense-codon signal. Largest sense divergences: day_0 A:CGG +0.5100, day_0 A:CCC +0.4797, day_0 A:TGG +0.4526, day_0 A:GGC +0.4348, day_5 A:CCC +0.4008, day_5 E:CGG -0.3649, day_0 A:ACC -0.3333. Largest cross-timepoint concordant cells: enriched A:GGG +0.178, P:CGG +0.166, A:GGT +0.161; depleted E:AAG -0.165, E:GAG -0.125, P:TAT -0.078. Of 186 (site,codon) cells, 36 are direction-concordant across 3 tp (25 enriched, 11 depleted), 150 show >=1 sign change; 114 of 150 register sig on both opposite-sign rows (N-driven; see flip-sig-large-N-artifact)."
user_directives:
  - "(per-CSV triage) 'Confirm test type? Feature=codon, 62 codons per (timepoint,site) family (61 sense + in-frame stop TGA; 558 = 62 x 3 sites x 3 tp); my read Fisher's exact 2x2 BWM vs control, BH within each (timepoint,site) family of 62 codons.' -> 'Confirm Fisher's exact 2x2'"
  - "(per-CSV triage) 'Which CSV-specific caveats?' -> confirmed all four: 'larger-bh-family', 'n-asymmetry-mild', 'flip-sig-large-N-artifact', and 'stop-codon-instability + rare-codon' (recorded as two caveats that coincide on TGA); declined small-bh-discreteness for sense codons (considered-not-applicable)."
  - "(per-CSV triage) Effect-display directive (carried from the aa pass) -> 'The data should be log2OR as dylan's table should return': effect column displayed as log2_OR (log2 of the CSV odds_ratio column), not bare odds_ratio."
  - "(per-CSV triage) 'How firmly should this read?' -> 'Mixed' (firm structural read; sense cells exploratory; TGA day_10 spike exploratory/unstable)."
  - "(per-CSV triage) 'Spotlight any site/codon/timepoint?' -> 'No spotlight' (rank by data alone)."
---

# Interpretation — codon_per_timepoint_fisher

> Source: `results/global_occupancy/analysis_corrected/codon_per_timepoint_fisher.csv`
> Family: `per_timepoint_fisher` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, codon); BH-FDR within each (timepoint, site) family of 62 codons (61 sense + in-frame stop TGA) (source: user-confirmed)

## User directives
- (per-CSV triage) "Confirm test type? Feature=codon, 62 codons per (timepoint, site) family (61 sense + the in-frame stop TGA; 558 = 62 x 3 sites x 3 tp); my read Fisher's exact 2x2 BWM vs control, BH within each (timepoint, site) family of 62 codons." → "Confirm Fisher's exact 2x2."
- (per-CSV triage) "Which CSV-specific caveats?" → confirmed all four offered: `larger-bh-family`, `n-asymmetry-mild`, `flip-sig-large-N-artifact`, and `stop-codon-instability + rare-codon` (recorded as two caveats that coincide on TGA). Declined `small-bh-family-discreteness` for the sense codons (considered-not-applicable).
- (per-CSV triage) Effect-display directive (carried from the aa pass) → "The data should be log2OR as dylan's table should return." Effect shown as `log2_OR` (log2 of the CSV `odds_ratio` column), not bare `odds_ratio`. Direction convention stated in Methods.
- (per-CSV triage) "How firmly should this read?" → "Mixed."
- (per-CSV triage) "Spotlight any site/codon/timepoint?" → "No spotlight." Headline ranks by data alone (A.2.3).

## Headline
558 tests (3 timepoints x 3 sites x 62 codons; the 62 are 61 sense codons + the single in-frame stop TGA). 445 clear FDR<0.05 (79.7%) and 466 clear FDR<0.10; the minimum adjusted p underflows to 0.0 (3 cells) and 253 of 558 cells sit at p_adj<1e-10. As in the aa sister, at whole-transcriptome pooled N (BWM/control totals 1.3M-3.4M per timepoint) FDR significance is near-universal and p magnitude is uninformative; rank by `|log2_OR|`, which for sense codons is small-to-moderate (largest sense enrichment day_0 A:CGG +0.5100, ~1.42x odds ratio; largest sense depletion day_5 E:CGG -0.3649). Per-(timepoint, site) n_sig is high at every timepoint, ranging from 38/62 (day_10 P) to 57/62 (day_0 A), with no day_0-dominated gradient (totals near-balanced, ctrl:BWM 1.59 / 1.20 / 0.81).

The in-frame stop TGA carries the single largest `|log2_OR|` at day_10 in all three sites (A +2.1403, E +2.1611, P +1.6835) but is the only codon below the count threshold (k_BWM 79-103, k_ctrl 16-20); it is flagged `rare-codon` + `stop` and is a low-count instability, not a sense-codon biological signal. Largest-magnitude sense-codon divergences, by `|log2_OR|`, mixing directions: day_0 A:CGG +0.5100, day_0 A:CCC +0.4797, day_0 A:TGG +0.4526, day_0 A:GGC +0.4348, day_0 A:CTA +0.4118, day_5 A:CCC +0.4008, day_5 A:CCG +0.3941, day_5 E:CGG -0.3649, day_0 A:ACC -0.3333, day_10 A:CCC -0.3267. Largest-magnitude cells in the cross-timepoint concordant set (same OR direction at all three timepoints; equal billing per A.2.2): enriched A:GGG +0.178 (mean log2OR), P:CGG +0.166, A:GGT +0.161, E:CTT +0.122, E:TTT +0.120; depleted E:AAG -0.165, E:GAG -0.125, P:TAT -0.078, E:AGA -0.077, E:AAC -0.068. Of the 186 (site, codon) cells, 36 are direction-concordant across all three timepoints (25 enriched, 11 depleted) and 150 show at least one sign change; 114 of those 150 register the change at p_adj<0.05 on both opposite-sign rows, which at this N is N-driven (see `flip-sig-large-N-artifact`), not 114 biological reversals.

## Top hits

The effect column is `log2_OR` (the log2 of the `odds_ratio` column), per the triage directive. Direction is fixed by the BWM-vs-control contingency layout: positive `log2_OR` = BWM-enriched at that (timepoint, site, codon); negative = BWM-depleted. `p_value` is the raw two-sided Fisher's exact p; `p_adj` is BH-corrected within each (timepoint, site) family of 62 codons.

Selection is the standard top-5 enriched + top-5 depleted by `|log2_OR|` within each (timepoint, site) group (every sense row shown clears FDR<0.05). The `large-N` flag marks every row with p_adj<1e-10 (most rows; rank by `|log2_OR|`, not p). `rare-codon` marks the in-frame stop TGA (the only sub-threshold codon); TGA is annotated inline. The first (timepoint, site) group (day_0, site A) is shown; the remaining eight are collapsed below. A cross-timepoint summary follows.

### day_0, site A

| direction | codon | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | CGG | +0.5100 | 5.33e-111 | 2.36e-110 | large-N |
| enriched | CCC | +0.4797 | 2.69e-94 | 8.34e-94 | large-N |
| enriched | TGG | +0.4526 | 0.00e+00 | 0.00e+00 | large-N |
| enriched | GGC | +0.4348 | 3.94e-129 | 2.22e-128 | large-N |
| enriched | CTA | +0.4118 | 6.11e-102 | 2.23e-101 | large-N |
| depleted | ACC | -0.3333 | 2.26e-198 | 4.68e-197 | large-N |
| depleted | GTC | -0.2571 | 4.60e-165 | 3.57e-164 | large-N |
| depleted | GTT | -0.2533 | 7.93e-187 | 1.23e-185 | large-N |
| depleted | GCC | -0.2324 | 3.57e-119 | 1.70e-118 | large-N |
| depleted | ATC | -0.2243 | 4.96e-166 | 4.39e-165 | large-N |

<details>
<summary>Remaining (timepoint, site) groups: day_0 P/E, day_5 A/P/E, day_10 A/P/E</summary>

### day_0, site P

| direction | codon | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | CGA | +0.4252 | 3.38e-199 | 2.10e-197 | large-N |
| enriched | CGG | +0.2271 | 4.85e-16 | 1.43e-15 | large-N |
| enriched | GGT | +0.2269 | 5.02e-94 | 6.23e-93 | large-N |
| enriched | GGC | +0.2032 | 7.72e-28 | 2.82e-27 | large-N |
| enriched | GGG | +0.2000 | 3.32e-13 | 7.92e-13 | large-N |
| depleted | ACC | -0.2365 | 1.71e-109 | 2.65e-108 | large-N |
| depleted | GAG | -0.1843 | 2.69e-156 | 8.34e-155 | large-N |
| depleted | TTG | -0.1523 | 2.90e-53 | 2.00e-52 | large-N |
| depleted | TTA | -0.1480 | 3.32e-13 | 7.92e-13 | large-N |
| depleted | ATG | -0.1279 | 3.08e-46 | 1.91e-45 | large-N |

### day_0, site E

| direction | codon | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | CCT | +0.3438 | 6.19e-88 | 1.28e-86 | large-N |
| enriched | AGG | +0.2380 | 4.13e-16 | 9.86e-16 | large-N |
| enriched | TGT | +0.1880 | 4.57e-29 | 1.42e-28 | large-N |
| enriched | CCA | +0.1814 | 5.10e-170 | 3.16e-168 | large-N |
| enriched | GTG | +0.1784 | 1.06e-49 | 9.39e-49 | large-N |
| depleted | ACC | -0.1992 | 2.52e-84 | 3.91e-83 | large-N |
| depleted | CAC | -0.1663 | 7.73e-41 | 3.42e-40 | large-N |
| depleted | TCC | -0.1557 | 6.50e-36 | 2.52e-35 | large-N |
| depleted | GTC | -0.1510 | 6.37e-68 | 6.59e-67 | large-N |
| depleted | GAG | -0.1440 | 1.23e-103 | 3.82e-102 | large-N |

### day_5, site A

| direction | codon | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | CCC | +0.4008 | 1.16e-45 | 4.22e-45 | large-N |
| enriched | CCG | +0.3941 | 2.68e-109 | 2.07e-108 | large-N |
| enriched | CTC | +0.3340 | 4.59e-195 | 7.11e-194 | large-N |
| enriched | CCT | +0.3269 | 2.09e-63 | 9.25e-63 | large-N |
| enriched | CCA | +0.2984 | 4.92e-264 | 1.52e-262 | large-N |
| depleted | GAA | -0.2613 | 0.00e+00 | 0.00e+00 | large-N |
| depleted | AAA | -0.2530 | 3.77e-217 | 7.78e-216 | large-N |
| depleted | ATT | -0.2094 | 6.34e-145 | 6.55e-144 | large-N |
| depleted | AAT | -0.1724 | 6.68e-94 | 4.14e-93 | large-N |
| depleted | AGA | -0.1687 | 2.34e-57 | 9.08e-57 | large-N |

### day_5, site P

| direction | codon | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | CAC | +0.2550 | 9.46e-71 | 8.38e-70 | large-N |
| enriched | CAA | +0.2213 | 1.03e-128 | 3.19e-127 | large-N |
| enriched | TGG | +0.2206 | 1.52e-42 | 7.85e-42 | large-N |
| enriched | TCC | +0.2015 | 3.23e-43 | 2.00e-42 | large-N |
| enriched | AAG | +0.1947 | 6.18e-177 | 3.83e-175 | large-N |
| depleted | TTT | -0.2030 | 1.87e-81 | 2.32e-80 | large-N |
| depleted | GTT | -0.1599 | 3.57e-90 | 5.54e-89 | large-N |
| depleted | TTA | -0.1508 | 1.80e-14 | 4.46e-14 | large-N |
| depleted | GAT | -0.1329 | 7.61e-111 | 1.57e-109 | large-N |
| depleted | GTA | -0.1272 | 2.02e-13 | 4.64e-13 | large-N |

### day_5, site E

| direction | codon | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | TCC | +0.3522 | 2.12e-146 | 2.19e-145 | large-N |
| enriched | TAC | +0.3173 | 4.39e-177 | 6.80e-176 | large-N |
| enriched | GCC | +0.3088 | 4.26e-156 | 5.28e-155 | large-N |
| enriched | CCA | +0.2854 | 2.30e-265 | 7.13e-264 | large-N |
| enriched | ACC | +0.2802 | 7.78e-122 | 4.83e-121 | large-N |
| depleted | CGG | -0.3649 | 1.26e-42 | 3.13e-42 | large-N |
| depleted | AGG | -0.2928 | 1.04e-21 | 1.84e-21 | large-N |
| depleted | AAG | -0.2926 | 0.00e+00 | 0.00e+00 | large-N |
| depleted | GGG | -0.2767 | 1.65e-20 | 2.76e-20 | large-N |
| depleted | AGT | -0.2521 | 3.69e-52 | 1.09e-51 | large-N |

### day_10, site A

| direction | codon | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | TGA (in-frame stop; unstable) | +2.1403 | 7.78e-12 | 2.41e-11 | rare-codon; stop |
| enriched | AAC | +0.2019 | 1.80e-57 | 3.72e-56 | large-N |
| enriched | AAT | +0.1990 | 4.80e-71 | 1.49e-69 | large-N |
| enriched | GGT | +0.1788 | 6.12e-21 | 3.45e-20 | large-N |
| enriched | CAA | +0.1347 | 1.09e-32 | 1.13e-31 | large-N |
| depleted | CCC | -0.3267 | 1.70e-22 | 1.05e-21 | large-N |
| depleted | TAC | -0.2675 | 9.60e-104 | 5.95e-102 | large-N |
| depleted | TGG | -0.2209 | 3.76e-44 | 5.83e-43 | large-N |
| depleted | GTG | -0.1611 | 1.13e-23 | 7.80e-23 | large-N |
| depleted | TTG | -0.1577 | 2.68e-39 | 3.32e-38 | large-N |

### day_10, site P

| direction | codon | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | TGA (in-frame stop; unstable) | +1.6835 | 4.29e-07 | 1.27e-06 | rare-codon; stop |
| enriched | TTA | +0.3136 | 6.28e-33 | 1.95e-31 | large-N |
| enriched | GTC | +0.1625 | 4.52e-37 | 2.80e-35 | large-N |
| enriched | CTG | +0.1600 | 1.57e-13 | 1.39e-12 | large-N |
| enriched | TGG | +0.1428 | 6.86e-13 | 4.72e-12 | large-N |
| depleted | TAT | -0.1405 | 8.07e-28 | 1.67e-26 | large-N |
| depleted | AAT | -0.0921 | 1.59e-21 | 1.97e-20 | large-N |
| depleted | AGA | -0.0843 | 3.61e-09 | 1.32e-08 |  |
| depleted | TAC | -0.0841 | 7.96e-10 | 3.29e-09 |  |
| depleted | ACC | -0.0751 | 7.20e-07 | 2.03e-06 |  |

### day_10, site E

| direction | codon | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | TGA (in-frame stop; unstable) | +2.1611 | 2.69e-10 | 1.27e-09 | rare-codon; stop |
| enriched | TGT | +0.1782 | 7.00e-16 | 5.42e-15 | large-N |
| enriched | TCT | +0.1354 | 1.28e-19 | 1.59e-18 | large-N |
| enriched | GGT | +0.1326 | 6.44e-14 | 3.99e-13 | large-N |
| enriched | TGC | +0.1314 | 2.31e-09 | 9.53e-09 |  |
| depleted | ACC | -0.1233 | 3.34e-17 | 3.45e-16 | large-N |
| depleted | CCC | -0.1230 | 4.17e-04 | 8.34e-04 |  |
| depleted | CCA | -0.1146 | 8.38e-30 | 2.60e-28 | large-N |
| depleted | AAG | -0.1079 | 1.91e-46 | 1.19e-44 | large-N |
| depleted | AAA | -0.0966 | 1.45e-27 | 3.00e-26 | large-N |

</details>

### Cross-timepoint summary

Per (site, codon) cell, how the BWM-vs-control OR direction behaves across the three timepoints. Descriptive per-cell tables, not a trajectory claim. `#sig` is the count (of 3 timepoints) at p_adj<0.05; `*` marks each per-timepoint `log2_OR` at p_adj<0.05; the per-tp triple is chronological (day_0, day_5, day_10).

#### Direction concordance (36 of 186 cells: same OR direction at all 3 timepoints)

25 enriched, 11 depleted. Sorted by `#sig` desc, then `|mean log2_OR|` desc. Top 15 enriched shown (10 more below the displayed cutoff); all 11 depleted shown.

##### Concordance — BWM-enriched (mean log2_OR > 0)

| codon | mean log2_OR | per-tp log2_OR | #sig | min p_adj | flag |
| --- | --- | --- | --- | --- | --- |
| A:GGG | +0.178 | +0.27, +0.16, +0.11 | 3/3 | 1.59e-23 | large-N |
| P:CGG | +0.166 | +0.23, +0.13, +0.14 | 3/3 | 1.43e-15 | large-N |
| A:GGT | +0.161 | +0.21, +0.09, +0.18 | 3/3 | 1.07e-57 | large-N |
| E:CTT | +0.122 | +0.12, +0.17, +0.07 | 3/3 | 4.11e-79 | large-N |
| E:TTT | +0.120 | +0.16, +0.11, +0.09 | 3/3 | 3.89e-42 | large-N |
| E:CTA | +0.119 | +0.15, +0.10, +0.11 | 3/3 | 6.77e-11 | large-N |
| E:GCT | +0.112 | +0.07, +0.23, +0.04 | 3/3 | 2.76e-122 | large-N |
| P:CTG | +0.106 | +0.05, +0.10, +0.16 | 3/3 | 1.39e-12 | large-N |
| E:TTC | +0.104 | +0.03, +0.25, +0.03 | 3/3 | 1.45e-180 | large-N |
| P:CAA | +0.100 | +0.03, +0.22, +0.05 | 3/3 | 3.18e-127 | large-N |
| A:CAG | +0.100 | +0.06, +0.11, +0.13 | 3/3 | 1.42e-20 | large-N |
| E:GCA | +0.092 | +0.16, +0.08, +0.04 | 3/3 | 2.94e-44 | large-N |
| E:GTT | +0.078 | +0.10, +0.09, +0.05 | 3/3 | 4.84e-35 | large-N |
| A:CAA | +0.077 | +0.06, +0.04, +0.13 | 3/3 | 1.13e-31 | large-N |
| E:TAT | +0.074 | +0.07, +0.12, +0.03 | 3/3 | 8.14e-22 | large-N |

(15 of 25 enriched concordant cells shown; 10 more below the cutoff `(#sig, |mean log2_OR|) = (3/3, 0.074)`.)

##### Concordance — BWM-depleted (mean log2_OR < 0)

| codon | mean log2_OR | per-tp log2_OR | #sig | min p_adj | flag |
| --- | --- | --- | --- | --- | --- |
| E:AAG | -0.165 | -0.09, -0.29, -0.11 | 3/3 | 0.00e+00 | large-N |
| E:GAG | -0.125 | -0.14, -0.17, -0.06 | 3/3 | 6.94e-133 | large-N |
| P:TAT | -0.078 | -0.04, -0.06, -0.14 | 3/3 | 1.67e-26 | large-N |
| P:GAA | -0.061 | -0.04, -0.12, -0.02 | 3/3 | 3.85e-62 | large-N |
| E:AGA | -0.077 | -0.01, -0.17, -0.05 | 2/3 | 7.59e-63 | large-N |
| E:AAC | -0.068 | -0.11, -0.00, -0.09 | 2/3 | 5.11e-44 | large-N |
| E:GGA | -0.065 | -0.08, -0.10, -0.01 | 2/3 | 7.04e-49 | large-N |
| E:GAA | -0.035 | -0.00, -0.07, -0.04 | 2/3 | 1.63e-21 | large-N |
| A:GGA | -0.026 | -0.04, -0.03, -0.01 | 2/3 | 1.93e-12 | large-N |
| P:AAA | -0.023 | -0.03, -0.03, -0.01 | 2/3 | 4.47e-05 |  |
| P:AGC | -0.026 | -0.03, -0.03, -0.02 | 0/3 | 5.10e-02 |  |

#### Direction-flip cells (150 of 186 cells: >=1 sign change across the 3 timepoints)

114 of the 150 register the change at p_adj<0.05 on both opposite-sign rows. Per `flip-sig-large-N-artifact`, this count is N-driven; rank by `|log2_OR|` range, not by `#sig`. Top 15 by `#sig` desc then max `|log2_OR|` desc; 135 more below the cutoff `(#sig, max |log2_OR|) = (3/3, 0.285)`.

| codon | log2_OR range | per-tp log2_OR | #sig | flag |
| --- | --- | --- | --- | --- |
| A:CGG | [-0.087, +0.510] | +0.51\*, +0.24\*, -0.09\* | 3/3 | large-N |
| A:CCC | [-0.327, +0.480] | +0.48\*, +0.40\*, -0.33\* | 3/3 | large-N |
| A:TGG | [-0.221, +0.453] | +0.45\*, +0.17\*, -0.22\* | 3/3 | large-N |
| A:GGC | [-0.112, +0.435] | +0.43\*, +0.21\*, -0.11\* | 3/3 | large-N |
| A:CCG | [-0.070, +0.394] | +0.33\*, +0.39\*, -0.07\* | 3/3 | large-N |
| A:CCT | [-0.074, +0.370] | +0.37\*, +0.33\*, -0.07\* | 3/3 | large-N |
| A:CTG | [-0.155, +0.364] | +0.36\*, +0.16\*, -0.16\* | 3/3 | large-N |
| A:GCG | [-0.126, +0.353] | +0.35\*, +0.16\*, -0.13\* | 3/3 | large-N |
| E:TCC | [-0.156, +0.352] | -0.16\*, +0.35\*, +0.05\* | 3/3 | large-N |
| E:CCT | [-0.065, +0.344] | +0.34\*, +0.13\*, -0.07\* | 3/3 | large-N |
| P:TTA | [-0.151, +0.314] | -0.15\*, -0.15\*, +0.31\* | 3/3 | large-N |
| E:GCC | [-0.129, +0.309] | -0.13\*, +0.31\*, -0.07\* | 3/3 | large-N |
| A:CCA | [-0.123, +0.298] | -0.02\*, +0.30\*, -0.12\* | 3/3 | large-N |
| A:GCA | [-0.151, +0.287] | +0.29\*, +0.07\*, -0.15\* | 3/3 | large-N |
| E:CCA | [-0.115, +0.285] | +0.18\*, +0.29\*, -0.11\* | 3/3 | large-N |

### Flag glossary
- `large-N` — the row's `p_adj` (or, for cross-timepoint cells, `min p_adj`) is below 1e-10. At whole-transcriptome pooled N Fisher's exact returns vanishing p for tiny absolute deviations; rank by `|log2_OR|`, not by p. Applied to every qualifying row for symmetry (A.2.4). A blank flag means FDR-significant but p_adj >= 1e-10.
- `rare-codon` — `BWM_count` < 100 or `control_count` < 100. The only such codon in this file is the in-frame stop TGA (9 rows); OR/p are unstable at small k.
- `stop` — the in-frame stop codon TGA (alternative explanation for its day_10 p_adj<1e-10 row per A.2.6: low-count instability of an in-frame stop, not a sense-codon signal). See `stop-codon-instability`.

## Numbers at a glance
- `n_tests`: 558 (3 timepoints x 3 sites x 62 codons; 62 = 61 sense + the in-frame stop TGA)
- `n_significant` (adjusted-p < 0.05): 445 (79.7%)
- `n_significant` (adjusted-p < 0.10): 466 (83.5%)
- `min adjusted-p`: 0.0 (underflow; 3 cells at exactly 0.0 to double precision). Smallest non-zero p_adj is 7.13e-264 (day_5 E:CCA, log2OR=+0.2854).
- Cells with p_adj < 1e-10: 253 of 558
- `p_floor`: n/a — Fisher's exact has no analytic floor; the relevant discipline is large-N anti-conservatism for the sense codons and small-k instability for the in-frame stop TGA (see caveats).
- Cross-(timepoint, site) BH families and their n_sig at p_adj<0.05:

| timepoint | A | P | E | BWM_total | control_total | ctrl:BWM ratio |
| --- | --- | --- | --- | --- | --- | --- |
| day_0 | 57/62 | 48/62 | 50/62 | 2,110,390 | 3,365,097 | 1.59 |
| day_5 | 53/62 | 52/62 | 56/62 | 2,014,868 | 2,414,993 | 1.20 |
| day_10 | 45/62 | 38/62 | 46/62 | 1,555,576 | 1,264,946 | 0.81 |

- Cross-timepoint (site, codon) cells, all 3 timepoints same OR direction: 36/186 (25 BWM-enriched, 11 BWM-depleted)
- Cross-timepoint (site, codon) cells with at least one sign change: 150/186
- Cells with sign change at p_adj<0.05 on both opposite-sign rows: 114/150 (N-driven; see `flip-sig-large-N-artifact`)

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 contingency `BWM_count` vs `control_count` for each (timepoint, site, codon), with BH-FDR within each (timepoint, site) family of 62 codons; user confirmed. The effect is reported as `log2_OR` (the log2 of the CSV `odds_ratio` column) per the triage directive; `log2_OR` > 0 = BWM-enriched, < 0 = BWM-depleted at that cell. The p-correction column is `p_adj` (BH per (timepoint, site) family, applied by `merge_global_occupancy_analysis.py`). The 62 codons per family are 61 sense codons + the in-frame stop TGA; the codon counting path (unlike the AA path, which yields 20 features) does not drop the in-frame stop window. The test answers "is the BWM-vs-control codon composition at this site different at this timepoint?". It does not separate synonymous-codon effects from amino-acid-level effects (compare the `aa` sister), does not test enrichment vs transcriptomic background (that is `within_condition_binomial`), and does not test whether the BWM-vs-control difference shifts across timepoints (the cross-timepoint summary above is descriptive only).

## Caveats
### Confirmed
- **pseudorep** (family-wide) — the 2x2 contingencies pool the 2 biological replicates per (condition, timepoint) before Fisher's exact; per-replicate variation is not in the test statistic, so p-values are anti-conservative. Inherited from family `per_timepoint_fisher`.
- **large-N-Fisher-anticonservative** (family-wide) — pooled totals are whole-transcriptome footprint counts (1.3M-3.4M per timepoint); Fisher's exact returns vanishing p (underflow to 0.0 for 3 cells; 253/558 below 1e-10) for tiny absolute deviations. `log2_OR` is the primary effect column; p magnitude is not a ranking axis. Inherited from family `per_timepoint_fisher`.
- **bh-per-(timepoint,site)** (family-wide) — BH is applied independently within each of the 9 (timepoint, site) families of 62 codons, not across the 558-test grid; cross-(timepoint, site) p_adj rankings are not directly commensurable. Inherited from family `per_timepoint_fisher`.
- **imbalanced-N** (family-wide) — the two arms differ in total per timepoint; Fisher handles imbalance correctly but interpretation should not over-read effect where one cell is small. Here the imbalance is mild (see `n-asymmetry-mild`). Inherited from family `per_timepoint_fisher`.
- **larger-bh-family** (per-CSV) — each per-(timepoint, site) BH family is 62 codons, ~3x the aa sister's 20-AA families. AA-level signals can split across synonyms below the per-(timepoint, site) FDR threshold; an AA hit absent here is an aggregation effect, not a contradiction.
- **n-asymmetry-mild** (per-CSV) — totals are near-balanced (ctrl:BWM 1.59 / 1.20 / 0.81; BWM is the larger arm by day_10), far milder than the stall-sites sister. Per-(timepoint, site) n_sig is high at every timepoint (38/62 to 57/62) with no day_0 gradient, so cross-timepoint n_sig differences here are not N-modulated the way they are in the stall-sites set.
- **flip-sig-large-N-artifact** (per-CSV) — 114 of the 150 cross-timepoint direction-flip cells register at p_adj<0.05 on both opposite-sign rows. At this N almost any small per-cell sign change clears FDR; the count reflects power, not 114 biological reversals. Rank flip cells by `|log2_OR|` range, not by `#sig`.
- **stop-codon-instability** (per-CSV) — the 62 codons/site are 61 sense + the single in-frame stop TGA (TAA/TAG absent; the codon set is data-driven and the codon path does not drop stop windows). Terminal stops are trimmed in the run, so this TGA is an in-frame stop (likely selenocysteine recoding / readthrough) with very low support. It is the largest `|log2_OR|` at day_10 in all three sites (A +2.1403, E +2.1611, P +1.6835) but on tiny counts; the day_10 spike is low-count instability, not a sense-codon signal. Matches the done between_*_wilcoxon codon files in this folder.
- **rare-codon-low-count** (per-CSV) — exactly 9 of 558 rows fall below BWM_count<100 or control_count<100, and all 9 are the in-frame stop TGA (3 sites x 3 timepoints); every sense codon has k in the thousands. The `rare-codon` flag therefore coincides exactly with `stop-codon-instability`. TGA's day_10 spike has k_BWM 79-103, k_ctrl 16-20.

### Considered but not applicable
- **OR-direction-anchor** — proposed as a standalone direction caveat; folded into the `log2_OR` display directive (sign encodes direction), so stated in Methods/Top hits rather than carried separately.
- **small-bh-family-discreteness** — the sense codons have per-cell k in the thousands, so Fisher p-values are effectively continuous with no coarse BH clustering. The only low-count codon (TGA) is covered by `rare-codon-low-count` / `stop-codon-instability`.

## For Chumeng (joint-reading hooks)
- Family: `per_timepoint_fisher` — sister CSV to reconcile: `aa_per_timepoint_fisher.csv` (AA resolution, same design, same family-wide caveats). See the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- Synonym-coherence falsifier: the day_0 A and day_5 A enrichments are dominated by Pro codons (CCC, CCG, CCT, CCA) and Arg codons (CGG, CGA at day_0 A/P). Do an amino acid's synonyms move in the same direction (synonym-coherent → AA aggregation preserves it) or split (one synonym enriched while a sister depletes → codon-usage signal AA aggregation would dilute)? The aa sister is the place to test; e.g. does aa-level P at day_5 A carry the +0.4 magnitude its codons show here.
- Falsifier vs the aa sister's concordant set: do the concordant codon cells (enriched A:GGG, P:CGG, A:GGT; depleted E:AAG, E:GAG) aggregate to a concordant cell at their amino acid (Gly for GGG/GGT/GGA, Arg for CGG, Lys for AAG, Glu for GAG)? A codon-level concordant cell that aggregates to a flat AA-level cell means the synonym carries the signal.
- Falsifier for `flip-sig-large-N-artifact`: do the highest-`#sig` flip cells (A:CGG, A:CCC, A:TGG, P:TTA) reappear as flips in the aa sister and in the within-condition Fisher contrasts (`timepoint_fisher_within_condition`)? A flip reproducing across designs is a candidate for elevation; a flip seen only here is most parsimoniously N-amplified per-cell noise.
- Falsifier for the in-frame stop TGA: does TGA reach a comparable +log2_OR extreme at day_10 in any other large-N file, or only here where its low day_10 count inflates the ratio? In the between_*_wilcoxon codon files TGA was always the largest swing but never significant; here it reaches p_adj<1e-10 at day_10 A purely on a 103-vs-19 count split. Treat as low-support, not a day_10 readthrough finding, unless a high-count file corroborates.
- Falsifier on resolution: per-(timepoint, site) n_sig is uniformly high here as in the aa sister. Does the codon gradient match the aa gradient quantitatively, or does codon resolution show relatively fewer day_0 / more day_5-10 hits (a resolution-dependent effect)?
