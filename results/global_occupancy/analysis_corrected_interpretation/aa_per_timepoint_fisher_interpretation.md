---
input_csv: results/global_occupancy/analysis_corrected/aa_per_timepoint_fisher.csv
family: per_timepoint_fisher
test_type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, aa); BH-FDR within each (timepoint, site) family of 20 AAs
test_type_source: user-confirmed
n_tests: 180
n_significant_fdr05: 159
n_significant_fdr10: 164
min_p_adj: 0.0
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "bh-per-(timepoint,site)", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "imbalanced-N", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "n-asymmetry-mild", proposed_by: dylan, status: confirmed, why: "Global-occupancy totals are near-balanced (ctrl:BWM 1.59 at day_0, 1.20 at day_5, 0.81 at day_10 — BWM is the larger arm by day_10), much milder than the stall-sites sister's 4.55x at day_0. Per-(timepoint,site) n_sig is uniformly high across all timepoints (15/20 to 20/20), with no day_0-dominated power gradient. The locked imbalanced-N caveat therefore has little teeth here; per-cell n_sig differences across timepoints are not power-modulated the way they are in the stall-sites set. User-confirmed."}
  - {label: "flip-sig-large-N-artifact", proposed_by: dylan, status: confirmed, why: "36 of the 42 cross-timepoint direction-flip cells register at p_adj<0.05 on both opposite-sign rows. At whole-transcriptome pooled N (totals 1.3M-3.4M) almost any small per-cell sign change clears FDR, so the high sig-flip count is N-driven, not 36 biological reversals. Rank flip cells by |log2_OR| range, not by the count of sig timepoints. User-confirmed."}
caveats_considered:
  - {label: "OR-direction-anchor", proposed_by: dylan, status: not-adopted, why: "Proposed stating the OR>1=BWM-enriched / OR<1=BWM-depleted convention as a formal caveat.", user_note: "User redirected to display the effect as log2_OR (the checker's column); sign of log2_OR encodes direction, so the convention is stated in Methods/Top-hits rather than carried as a standalone caveat."}
  - {label: "rare-aa-low-count", proposed_by: dylan, status: denied, why: "0 of 180 rows fall below BWM_count<100 OR control_count<100; at whole-transcriptome N every amino acid has high counts in both arms.", user_note: "Recorded as not applicable (unlike the stall-sites sister where rare AAs W/M/C had low-k cells)."}
  - {label: "small-bh-family-discreteness", proposed_by: dylan, status: denied, why: "The stall-sites discreteness came from small per-cell k giving discrete Fisher p-values clustering at coarse BH levels. Here per-cell k is in the thousands so p-values are effectively continuous and do not cluster.", user_note: "Recorded as not applicable at global-occupancy N."}
headline: "180 BWM-vs-control Fisher tests (3 timepoints x 3 sites x 20 AAs): 159/180 clear FDR<0.05 (88.3%), 164/180 FDR<0.10; min p_adj underflows to 0.0 (4 cells), 106/180 cells at p_adj<1e-10. At whole-transcriptome pooled N (totals 1.3M-3.4M per timepoint) FDR significance is near-universal and p magnitude is uninformative; the informative axis is effect size, which is small (largest single-cell |log2_OR| is day_5 A:P +0.3300, ~1.26x). Per-(timepoint,site) n_sig is uniformly high (15/20 day_10 A to 20/20 day_0 A and day_5 E) with no day_0 power gradient (totals near-balanced, ctrl:BWM 1.59/1.20/0.81). Largest divergences (mixing directions): day_5 A:P +0.3300, day_5 E:Y +0.2272, day_5 E:K -0.2260, day_10 A:W -0.2209, day_0 E:P +0.1974, day_0 A:D -0.1905. Largest cross-timepoint concordant cells (same direction all 3 tp, equal billing): enriched P:W +0.122 mean, E:F +0.110, E:C +0.107; depleted E:K -0.126, A:E -0.084, E:E -0.082. Of 60 (site,aa) cells, 18 are direction-concordant across all 3 timepoints (12 enriched, 6 depleted), 42 show >=1 sign change; 36 of the 42 register the change at p_adj<0.05 on both opposite-sign rows, but at this N a significant flip is N-driven, not a biological reversal."
user_directives:
  - "(per-CSV triage) 'Confirm test type? Columns site/timepoint/amino_acid/odds_ratio/p_value/BWM_count/BWM_total/control_count/control_total/p_adj; my read Fisher's exact 2x2 BWM vs control per (timepoint,site,aa), BH within each (timepoint,site) family of 20 AAs.' -> 'Confirm Fisher's exact 2x2'"
  - "(per-CSV triage) 'Which CSV-specific caveats?' -> confirmed 'n-asymmetry-mild' and 'flip-sig-large-N-artifact'; declined rare-aa (0/180) and small-bh-discreteness (recorded considered-not-applicable)."
  - "(per-CSV triage) Effect-display directive -> 'The data should be log2OR as dylan's table should return': effect column is displayed as log2_OR (log2 of the CSV odds_ratio column), matching the checker output, not bare odds_ratio."
  - "(per-CSV triage) 'How firmly should this read?' -> 'Mixed' (firm on the structural read that near-universal FDR significance is large-N not biology and effects are small; individual cells exploratory pending cross-test corroboration)."
  - "(per-CSV triage) 'Spotlight any site/AA/timepoint?' -> 'No spotlight' (rank by data alone)."
---

# Interpretation — aa_per_timepoint_fisher

> Source: `results/global_occupancy/analysis_corrected/aa_per_timepoint_fisher.csv`
> Family: `per_timepoint_fisher` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, aa); BH-FDR within each (timepoint, site) family of 20 AAs (source: user-confirmed)

## User directives
- (per-CSV triage) "Confirm test type? Columns `site`, `timepoint`, `amino_acid`, `odds_ratio`, `p_value`, `BWM_count`, `BWM_total`, `control_count`, `control_total`, `p_adj`; my read Fisher's exact 2x2 BWM vs control per (timepoint, site, aa), BH within each (timepoint, site) family of 20 AAs." → "Confirm Fisher's exact 2x2."
- (per-CSV triage) "Which CSV-specific caveats?" → confirmed `n-asymmetry-mild` and `flip-sig-large-N-artifact`; declined `rare-aa-low-count` (0/180 rows below k<100) and `small-bh-family-discreteness` (huge per-cell counts → continuous p). Both recorded as considered-not-applicable.
- (per-CSV triage) Effect-display directive → "The data should be log2OR as dylan's table should return." The effect column is shown as `log2_OR` (log2 of the CSV `odds_ratio` column), matching the checker output, not bare `odds_ratio`. Direction convention stated in Methods.
- (per-CSV triage) "How firmly should this read?" → "Mixed" (firm structural read; individual cells exploratory).
- (per-CSV triage) "Spotlight any site/AA/timepoint?" → "No spotlight." Headline ranks by data alone (A.2.3).

## Headline
180 tests (3 timepoints x 3 sites x 20 AAs). 159 clear FDR<0.05 (88.3%) and 164 clear FDR<0.10; the minimum adjusted p underflows to 0.0 (4 cells) and 106 of 180 cells sit at p_adj<1e-10. At whole-transcriptome pooled N (BWM/control totals 1.3M-3.4M per timepoint) FDR significance is near-universal, so p magnitude carries essentially no information here; the informative axis is effect size, and effect sizes are small — the largest single-cell `|log2_OR|` in the file is day_5 A:P +0.3300 (about a 1.26x odds ratio). Per-(timepoint, site) n_sig at FDR<0.05 is uniformly high, ranging only from 15/20 (day_10 A) to 20/20 (day_0 A and day_5 E), with no day_0-dominated power gradient — totals are near-balanced (ctrl:BWM 1.59 / 1.20 / 0.81 across day_0/day_5/day_10).

Largest-magnitude single-cell BWM-vs-control divergences, by `|log2_OR|`, mixing directions: day_5 A:P +0.3300, day_5 E:Y +0.2272, day_5 E:K -0.2260, day_10 A:W -0.2209, day_5 P:W +0.2206, day_10 A:N +0.2048, day_0 E:P +0.1974, day_5 A:E -0.1905 / day_0 A:D -0.1905. Largest-magnitude cells in the cross-timepoint concordant set (same OR direction at all three timepoints; reported at equal billing with the divergences per A.2.2): enriched P:W +0.122 (mean log2OR), E:F +0.110, E:C +0.107, A:Q +0.086, P:Q +0.085; depleted E:K -0.126, A:E -0.084, E:E -0.082, P:E -0.081, E:N -0.066. Of the 60 (site, aa) cells, 18 are direction-concordant across all three timepoints (12 enriched, 6 depleted) and 42 show at least one sign change; 36 of those 42 register the change at p_adj<0.05 on both opposite-sign rows, but at this N a "significant flip" is N-driven (see `flip-sig-large-N-artifact`), not evidence of 36 biological reversals.

## Top hits

The effect column is `log2_OR` (the log2 of the `odds_ratio` column), per the triage directive. Direction is fixed by the BWM-vs-control contingency layout: positive `log2_OR` = BWM-enriched relative to control at that (timepoint, site, aa); negative = BWM-depleted. `p_value` is the raw two-sided Fisher's exact p; `p_adj` is BH-corrected within each (timepoint, site) family of 20 AAs.

Selection is the standard top-5 enriched + top-5 depleted by `|log2_OR|` within each (timepoint, site) group (every row shown clears FDR<0.05). Because p magnitude is uninformative at this N, the `large-N` flag marks every row with p_adj<1e-10 — most rows carry it; rank by `|log2_OR|`, not by p. A blank flag means the row is FDR-significant but its p_adj is above 1e-10 (genuinely weaker, not extreme). The first (timepoint, site) group (day_0, site A) is shown; the remaining eight are collapsed below. A cross-timepoint summary (direction concordance + direction-flip) follows the per-group tables.

### day_0, site A

| direction | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | W | +0.4526 | 0.00e+00 | 0.00e+00 | large-N |
| enriched | C | +0.1858 | 1.20e-75 | 3.42e-75 | large-N |
| enriched | L | +0.1580 | 2.22e-252 | 1.48e-251 | large-N |
| enriched | P | +0.1047 | 3.67e-69 | 9.16e-69 | large-N |
| enriched | Y | +0.0881 | 9.59e-44 | 2.13e-43 | large-N |
| depleted | D | -0.1905 | 2.21e-290 | 2.21e-289 | large-N |
| depleted | I | -0.1772 | 8.18e-227 | 4.09e-226 | large-N |
| depleted | N | -0.1743 | 6.07e-167 | 2.43e-166 | large-N |
| depleted | V | -0.1246 | 5.57e-121 | 1.86e-120 | large-N |
| depleted | M | -0.0622 | 6.80e-13 | 9.07e-13 | large-N |

<details>
<summary>Remaining (timepoint, site) groups: day_0 P/E, day_5 A/P/E, day_10 A/P/E</summary>

### day_0, site P

| direction | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | G | +0.1439 | 1.13e-200 | 2.25e-199 | large-N |
| enriched | D | +0.1283 | 6.44e-161 | 6.44e-160 | large-N |
| enriched | F | +0.0728 | 8.17e-32 | 2.34e-31 | large-N |
| enriched | R | +0.0480 | 4.30e-18 | 7.81e-18 | large-N |
| enriched | A | +0.0367 | 1.15e-11 | 1.92e-11 | large-N |
| depleted | M | -0.1279 | 3.08e-46 | 1.23e-45 | large-N |
| depleted | E | -0.1163 | 3.49e-121 | 2.33e-120 | large-N |
| depleted | K | -0.0812 | 4.51e-67 | 2.26e-66 | large-N |
| depleted | Y | -0.0743 | 1.63e-27 | 4.07e-27 | large-N |
| depleted | L | -0.0694 | 2.95e-43 | 9.82e-43 | large-N |

### day_0, site E

| direction | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | P | +0.1974 | 7.79e-269 | 1.56e-267 | large-N |
| enriched | C | +0.0975 | 1.05e-16 | 2.63e-16 | large-N |
| enriched | F | +0.0739 | 1.31e-28 | 5.22e-28 | large-N |
| enriched | L | +0.0660 | 4.02e-42 | 2.68e-41 | large-N |
| enriched | A | +0.0378 | 2.66e-12 | 4.43e-12 | large-N |
| depleted | H | -0.1132 | 5.13e-38 | 2.57e-37 | large-N |
| depleted | E | -0.0745 | 1.87e-55 | 1.87e-54 | large-N |
| depleted | N | -0.0495 | 3.33e-18 | 9.51e-18 | large-N |
| depleted | M | -0.0476 | 1.80e-08 | 2.57e-08 |  |
| depleted | K | -0.0450 | 1.10e-26 | 3.65e-26 | large-N |

### day_5, site A

| direction | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | P | +0.3300 | 0.00e+00 | 0.00e+00 | large-N |
| enriched | A | +0.1925 | 1.35e-219 | 5.39e-219 | large-N |
| enriched | L | +0.1764 | 1.78e-249 | 8.91e-249 | large-N |
| enriched | W | +0.1688 | 5.64e-37 | 1.41e-36 | large-N |
| enriched | R | +0.0720 | 1.40e-34 | 3.11e-34 | large-N |
| depleted | E | -0.1905 | 0.00e+00 | 0.00e+00 | large-N |
| depleted | K | -0.1748 | 1.33e-263 | 8.85e-263 | large-N |
| depleted | I | -0.1612 | 1.13e-179 | 3.76e-179 | large-N |
| depleted | N | -0.1499 | 2.23e-122 | 6.36e-122 | large-N |
| depleted | M | -0.1062 | 1.05e-33 | 2.11e-33 | large-N |

### day_5, site P

| direction | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | W | +0.2206 | 1.64e-42 | 2.98e-42 | large-N |
| enriched | Q | +0.2040 | 3.34e-158 | 6.68e-157 | large-N |
| enriched | H | +0.1711 | 1.49e-78 | 5.96e-78 | large-N |
| enriched | P | +0.1111 | 1.96e-57 | 5.61e-57 | large-N |
| enriched | K | +0.0994 | 9.58e-80 | 4.79e-79 | large-N |
| depleted | V | -0.1283 | 3.45e-121 | 3.45e-120 | large-N |
| depleted | E | -0.1157 | 1.48e-105 | 9.84e-105 | large-N |
| depleted | F | -0.0997 | 2.84e-53 | 7.09e-53 | large-N |
| depleted | D | -0.0876 | 3.58e-66 | 1.19e-65 | large-N |
| depleted | I | -0.0789 | 1.55e-44 | 3.09e-44 | large-N |

### day_5, site E

| direction | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | Y | +0.2272 | 4.54e-167 | 1.81e-166 | large-N |
| enriched | P | +0.2109 | 1.04e-210 | 1.04e-209 | large-N |
| enriched | F | +0.2074 | 5.64e-180 | 2.82e-179 | large-N |
| enriched | A | +0.1845 | 1.42e-200 | 9.43e-200 | large-N |
| enriched | T | +0.1497 | 4.96e-123 | 1.42e-122 | large-N |
| depleted | K | -0.2260 | 0.00e+00 | 0.00e+00 | large-N |
| depleted | M | -0.1999 | 3.40e-116 | 8.49e-116 | large-N |
| depleted | W | -0.1760 | 1.18e-36 | 1.57e-36 | large-N |
| depleted | G | -0.1333 | 1.15e-113 | 2.55e-113 | large-N |
| depleted | E | -0.1237 | 3.28e-133 | 1.09e-132 | large-N |

### day_10, site A

| direction | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | N | +0.2048 | 1.47e-129 | 2.94e-128 | large-N |
| enriched | Q | +0.1352 | 3.37e-49 | 2.25e-48 | large-N |
| enriched | I | +0.0491 | 1.61e-11 | 3.57e-11 | large-N |
| enriched | M | +0.0489 | 1.32e-05 | 2.20e-05 |  |
| enriched | K | +0.0474 | 7.35e-13 | 1.84e-12 | large-N |
| depleted | W | -0.2209 | 3.77e-44 | 1.51e-43 | large-N |
| depleted | Y | -0.1909 | 8.70e-100 | 8.70e-99 | large-N |
| depleted | P | -0.1234 | 1.77e-48 | 8.85e-48 | large-N |
| depleted | A | -0.0984 | 3.91e-41 | 1.30e-40 | large-N |
| depleted | L | -0.0676 | 2.08e-25 | 5.94e-25 | large-N |

### day_10, site P

| direction | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | W | +0.1428 | 6.86e-13 | 2.29e-12 | large-N |
| enriched | L | +0.1272 | 1.54e-76 | 3.08e-75 | large-N |
| enriched | V | +0.0863 | 1.65e-34 | 1.10e-33 | large-N |
| enriched | S | +0.0579 | 2.38e-15 | 9.51e-15 | large-N |
| enriched | M | +0.0417 | 5.88e-04 | 7.84e-04 |  |
| depleted | Y | -0.1160 | 1.18e-34 | 1.10e-33 | large-N |
| depleted | N | -0.0849 | 2.55e-30 | 1.28e-29 | large-N |
| depleted | C | -0.0519 | 1.25e-04 | 1.92e-04 |  |
| depleted | R | -0.0416 | 3.28e-08 | 9.37e-08 |  |
| depleted | P | -0.0385 | 4.42e-06 | 9.81e-06 |  |

### day_10, site E

| direction | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| enriched | C | +0.1557 | 2.15e-23 | 8.59e-23 | large-N |
| enriched | W | +0.1170 | 2.31e-11 | 6.59e-11 | large-N |
| enriched | S | +0.0888 | 1.39e-35 | 6.93e-35 | large-N |
| enriched | L | +0.0827 | 1.58e-37 | 1.58e-36 | large-N |
| enriched | F | +0.0474 | 1.86e-07 | 4.13e-07 |  |
| depleted | K | -0.1084 | 1.65e-75 | 3.29e-74 | large-N |
| depleted | P | -0.1042 | 2.87e-36 | 1.91e-35 | large-N |
| depleted | E | -0.0481 | 3.71e-14 | 1.24e-13 | large-N |
| depleted | N | -0.0403 | 1.78e-07 | 4.13e-07 |  |
| depleted | I | -0.0254 | 6.17e-04 | 1.10e-03 |  |

</details>

### Cross-timepoint summary

These two tables describe, per (site, aa) cell, how the BWM-vs-control OR direction behaves across the three timepoints. They are per-cell descriptive tables, not a trajectory claim. `#sig` is the count (of 3 timepoints) at p_adj<0.05; `*` marks each per-timepoint `log2_OR` at p_adj<0.05; the per-tp triple is in chronological order (day_0, day_5, day_10).

#### Direction concordance (18 of 60 cells: same OR direction at all 3 timepoints)

12 enriched, 6 depleted. Sorted by `#sig` desc, then `|mean log2_OR|` desc.

| cell | mean log2_OR | per-tp log2_OR | #sig | min p_adj | flag |
| --- | --- | --- | --- | --- | --- |
| E:F | +0.110 | +0.07, +0.21, +0.05 | 3/3 | 2.82e-179 | large-N |
| E:C | +0.107 | +0.10, +0.07, +0.16 | 3/3 | 8.59e-23 | large-N |
| A:Q | +0.086 | +0.06, +0.06, +0.14 | 3/3 | 2.25e-48 | large-N |
| P:Q | +0.085 | +0.02, +0.20, +0.03 | 3/3 | 6.68e-157 | large-N |
| E:L | +0.076 | +0.07, +0.08, +0.08 | 3/3 | 3.41e-51 | large-N |
| E:V | +0.042 | +0.04, +0.08, +0.01 | 3/3 | 8.76e-41 | large-N |
| A:S | +0.042 | +0.06, +0.03, +0.04 | 3/3 | 6.25e-26 | large-N |
| P:W | +0.122 | +0.00, +0.22, +0.14 | 2/3 | 2.98e-42 | large-N |
| E:A | +0.079 | +0.04, +0.18, +0.01 | 2/3 | 9.43e-200 | large-N |
| A:R | +0.048 | +0.07, +0.07, +0.00 | 2/3 | 1.42e-37 | large-N |
| P:S | +0.028 | +0.02, +0.01, +0.06 | 2/3 | 9.51e-15 | large-N |
| A:G | +0.022 | +0.04, +0.01, +0.01 | 2/3 | 7.05e-17 | large-N |
| E:K | -0.126 | -0.05, -0.23, -0.11 | 3/3 | 0.00e+00 | large-N |
| E:E | -0.082 | -0.07, -0.12, -0.05 | 3/3 | 1.09e-132 | large-N |
| E:N | -0.066 | -0.05, -0.11, -0.04 | 3/3 | 6.65e-73 | large-N |
| E:D | -0.033 | -0.03, -0.05, -0.02 | 3/3 | 2.37e-19 | large-N |
| A:E | -0.084 | -0.06, -0.19, -0.00 | 2/3 | 0.00e+00 | large-N |
| P:E | -0.081 | -0.12, -0.12, -0.01 | 2/3 | 2.33e-120 | large-N |

#### Direction-flip cells (42 of 60 cells: >=1 sign change across the 3 timepoints)

36 of the 42 register the change at p_adj<0.05 on both opposite-sign rows. Per `flip-sig-large-N-artifact`, this count is N-driven; rank by `|log2_OR|` range, not by `#sig`. Top 15 by `#sig` desc then max `|log2_OR|` desc; 27 more below the cutoff `(#sig, |log2_OR|) = (3/3, 0.128)`.

| cell | log2_OR range | per-tp log2_OR | #sig | flag |
| --- | --- | --- | --- | --- |
| A:W | [-0.221, +0.453] | +0.45\*, +0.17\*, -0.22\* | 3/3 | large-N |
| A:P | [-0.123, +0.330] | +0.10\*, +0.33\*, -0.12\* | 3/3 | large-N |
| E:P | [-0.104, +0.211] | +0.20\*, +0.21\*, -0.10\* | 3/3 | large-N |
| A:N | [-0.174, +0.205] | -0.17\*, -0.15\*, +0.20\* | 3/3 | large-N |
| E:M | [-0.200, +0.034] | -0.05\*, -0.20\*, +0.03\* | 3/3 | large-N |
| A:A | [-0.098, +0.193] | +0.04\*, +0.19\*, -0.10\* | 3/3 | large-N |
| A:Y | [-0.191, +0.088] | +0.09\*, -0.06\*, -0.19\* | 3/3 | large-N |
| A:I | [-0.177, +0.049] | -0.18\*, -0.16\*, +0.05\* | 3/3 | large-N |
| A:L | [-0.068, +0.176] | +0.16\*, +0.18\*, -0.07\* | 3/3 | large-N |
| E:W | [-0.176, +0.117] | +0.03\*, -0.18\*, +0.12\* | 3/3 | large-N |
| A:K | [-0.175, +0.047] | -0.02\*, -0.17\*, +0.05\* | 3/3 | large-N |
| E:T | [-0.031, +0.150] | -0.03\*, +0.15\*, -0.02\* | 3/3 | large-N |
| E:G | [-0.133, +0.023] | -0.04\*, -0.13\*, +0.02\* | 3/3 | large-N |
| P:V | [-0.128, +0.086] | +0.02\*, -0.13\*, +0.09\* | 3/3 | large-N |
| P:D | [-0.088, +0.128] | +0.13\*, -0.09\*, -0.03\* | 3/3 | large-N |

### Flag glossary
- `large-N` — the row's `p_adj` is below 1e-10 (or for cross-timepoint cells, its `min p_adj` is). At whole-transcriptome pooled N Fisher's exact returns vanishing p for tiny absolute deviations, so `-log10(p_adj)` is inflated; rank by `|log2_OR|`, not by p. Applied to every qualifying row for symmetry (A.2.4). A blank flag means FDR-significant but p_adj >= 1e-10.

## Numbers at a glance
- `n_tests`: 180 (3 timepoints x 3 sites x 20 AAs)
- `n_significant` (adjusted-p < 0.05): 159 (88.3%)
- `n_significant` (adjusted-p < 0.10): 164 (91.1%)
- `min adjusted-p`: 0.0 (underflow; 4 cells at exactly 0.0 to double precision). Smallest non-zero p_adj is 2.21e-289 (day_0 A:D, log2OR=-0.1905).
- Cells with p_adj < 1e-10: 106 of 180
- `p_floor`: n/a — Fisher's exact has no analytic floor; the relevant discipline here is large-N anti-conservatism (see caveats), not a discreteness floor (per-cell counts are in the thousands).
- Cross-(timepoint, site) BH families and their n_sig at p_adj<0.05:

| timepoint | A | P | E | BWM_total | control_total | ctrl:BWM ratio |
| --- | --- | --- | --- | --- | --- | --- |
| day_0 | 20/20 | 18/20 | 18/20 | 2,110,322 | 3,364,978 | 1.59 |
| day_5 | 16/20 | 17/20 | 20/20 | 2,014,776 | 2,414,875 | 1.20 |
| day_10 | 15/20 | 17/20 | 18/20 | 1,555,473 | 1,264,927 | 0.81 |

- Cross-timepoint (site, aa) cells, all 3 timepoints same OR direction: 18/60 (12 BWM-enriched, 6 BWM-depleted)
- Cross-timepoint (site, aa) cells with at least one sign change: 42/60
- Cells with sign change at p_adj<0.05 on both opposite-sign rows: 36/42 (N-driven; see `flip-sig-large-N-artifact`)

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 contingency `BWM_count` vs `control_count` for each (timepoint, site, aa), with BH-FDR within each (timepoint, site) family of 20 AAs; user confirmed. The effect is reported as `log2_OR` (the log2 of the CSV `odds_ratio` column) per the triage directive; `log2_OR` > 0 means BWM-enriched relative to control at that cell, < 0 means BWM-depleted. The p-correction column is `p_adj` (BH per (timepoint, site) family, applied by `merge_global_occupancy_analysis.py`). The test answers "is the BWM-vs-control AA composition at this site different at this timepoint?". It does not answer "is each condition's per-aa occupancy enriched vs its transcriptomic background" (that is the `within_condition_binomial` family) and does not test "is the BWM-vs-control difference itself shifting across timepoints" (that needs an explicit between-timepoint contrast; the cross-timepoint summary above is descriptive only).

## Caveats
### Confirmed
- **pseudorep** (family-wide) — the 2x2 contingencies pool the 2 biological replicates per (condition, timepoint) before Fisher's exact; per-replicate variation is not in the test statistic, so p-values are anti-conservative. Inherited from family `per_timepoint_fisher`.
- **large-N-Fisher-anticonservative** (family-wide) — pooled totals are whole-transcriptome footprint counts (1.3M-3.4M per timepoint); Fisher's exact returns vanishing p (here, underflow to 0.0 for 4 cells; 106/180 below 1e-10) for tiny absolute deviations. `log2_OR` is the primary effect column; p magnitude is not a ranking axis. Inherited from family `per_timepoint_fisher`.
- **bh-per-(timepoint,site)** (family-wide) — BH is applied independently within each of the 9 (timepoint, site) families of 20 AAs, not across the full 180-test grid; cross-(timepoint, site) p_adj rankings are not directly commensurable. Inherited from family `per_timepoint_fisher`.
- **imbalanced-N** (family-wide) — the two arms differ in total per timepoint; Fisher handles imbalance correctly but interpretation should not over-read effect where one cell is small. Here the imbalance is mild (see `n-asymmetry-mild`). Inherited from family `per_timepoint_fisher`.
- **n-asymmetry-mild** (per-CSV) — totals are near-balanced (ctrl:BWM 1.59 / 1.20 / 0.81; BWM is the larger arm by day_10), far milder than the stall-sites sister's 4.55x at day_0. Per-(timepoint, site) n_sig is uniformly high across all timepoints (15/20 to 20/20) with no day_0 power gradient, so cross-timepoint n_sig differences here are not N-modulated the way they are in the stall-sites set.
- **flip-sig-large-N-artifact** (per-CSV) — 36 of the 42 cross-timepoint direction-flip cells register at p_adj<0.05 on both opposite-sign rows. At this N almost any small per-cell sign change clears FDR, so the high sig-flip count reflects power, not 36 biological reversals. Rank flip cells by `|log2_OR|` range, not by `#sig`.

### Considered but not applicable
- **OR-direction-anchor** — proposed as a standalone direction caveat; the user redirected to displaying the effect as `log2_OR` (sign encodes direction), so the convention is stated in Methods/Top hits rather than carried as a separate caveat.
- **rare-aa-low-count** — 0 of 180 rows fall below BWM_count<100 or control_count<100; every amino acid has high counts in both arms at whole-transcriptome N (unlike the stall-sites sister where W/M/C had low-k cells).
- **small-bh-family-discreteness** — at per-cell k in the thousands the Fisher p-values are effectively continuous and do not cluster at coarse BH levels, so the discreteness concern that applies in the small-k stall-sites file does not apply here.

## For Chumeng (joint-reading hooks)
- Family: `per_timepoint_fisher` — sister CSV to reconcile: `codon_per_timepoint_fisher.csv` (codon resolution, same design, same family-wide caveats). See the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- Falsifier: the largest divergences here are small (`|log2_OR|` <= 0.33) yet uniformly FDR-significant. Do the same (site, aa) cells carry comparable `|log2_OR|` in the codon sister aggregated to AA, and does any AA-level cell exceed the codon-level signal (then AA is the carrier) or fall below it (then a synonym carries it)?
- Falsifier for `flip-sig-large-N-artifact`: do the high-`#sig` flip cells (A:W, A:P, E:P, A:N) reappear as flips in the codon sister, and do they also flip in the within-condition Fisher contrasts (`timepoint_fisher_within_condition`)? A flip that reproduces across designs is a candidate for elevation; a flip seen only here is most parsimoniously N-amplified per-cell noise.
- Falsifier for the concordant set: do the all-3-timepoints same-direction cells (enriched E:F, E:C, A:Q; depleted E:K, A:E, E:E) appear as stable baselines in `within_condition_binomial` (then the BWM-vs-control concordance rides on a shared baseline) or as group-variable cells (then it carries condition-specific information)?
- Falsifier on power: per-(timepoint, site) n_sig is uniformly high here (no day_0 gradient). Does the codon sister show the same uniform-across-timepoints pattern? If both do, the absence of a gradient is a property of the balanced global-occupancy N; if the codon file shows a gradient, that flags a resolution-dependent effect.
