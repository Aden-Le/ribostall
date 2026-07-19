# Within-Condition Binomial Enrichment (A1)

**Pipeline:** stall_sites_consensus_intersection (C. elegans)
**Test:** Two-sided binomial test (`scipy.stats.binomtest`) of the observed codon/amino-acid frequency at each E/P/A stall site against that group's background frequency, pooled within the group (`ribostall.stats_core.binom_row`, wrapped by `ribostall.enrichment.within_condition_enrichment`). Null hypothesis: the stall-site frequency of the feature equals genome background. Positive `log2_enrichment` means enriched relative to background; negative means depleted.
**Source data:** `analysis/within_condition_binomial_aa.csv`, `analysis/within_condition_binomial_codon.csv`

## Key Data — Amino Acid level

- Tests run: **360** · Significant (p_adj < 0.05): **122** (33.9%)
- Direction split (significant only): **55** favor **enriched**, **67** favor **depleted**

**Most significant (top 10 by p_adj)**

| Site | Group | Feature | log2_enrichment | Obs/Total | Obs freq | Bg freq | p_value | p_adj | Flags |
|---|---|---|---|---|---|---|---|---|---|
| E | BWM_day_0 | K | 1.179 | 183/785 | 0.2331 | 0.1030 | 7.09e-26 | 1.42e-24 |  |
| E | control_day_5 | K | 1.113 | 166/745 | 0.2228 | 0.1030 | 1.86e-21 | 3.73e-20 |  |
| E | control_day_0 | K | 1.083 | 132/605 | 0.2182 | 0.1030 | 1.26e-16 | 2.52e-15 |  |
| E | BWM_day_5 | K | 1.029 | 141/671 | 0.2101 | 0.1030 | 3.37e-16 | 6.73e-15 |  |
| P | control_day_10 | D | 1.288 | 89/732 | 0.1216 | 0.0498 | 2.15e-14 | 4.3e-13 |  |
| P | BWM_day_10 | D | 1.226 | 97/833 | 0.1164 | 0.0498 | 2.51e-14 | 5.01e-13 |  |
| A | BWM_day_10 | A | -1.794 | 20/833 | 0.0240 | 0.0833 | 1.1e-12 | 2.2e-11 | low-count |
| A | control_day_10 | E | 1.114 | 96/732 | 0.1311 | 0.0606 | 2e-12 | 4e-11 |  |
| E | BWM_day_10 | K | 0.835 | 153/833 | 0.1837 | 0.1030 | 3.02e-12 | 6.04e-11 |  |
| E | control_day_10 | K | 0.862 | 137/732 | 0.1872 | 0.1030 | 8.41e-12 | 1.68e-10 |  |

**Largest effect (top 10 by \|effect\|, all rows)**

| Site | Group | Feature | log2_enrichment | Obs/Total | Obs freq | Bg freq | p_value | p_adj | Flags |
|---|---|---|---|---|---|---|---|---|---|
| E | control_day_5 | C | -3.225 | 1/745 | 0.0013 | 0.0125 | 0.0015 | 0.00501 | low-count |
| E | BWM_day_5 | C | -3.074 | 1/671 | 0.0015 | 0.0125 | 0.00454 | 0.0182 | low-count |
| E | control_day_0 | C | -2.924 | 1/605 | 0.0017 | 0.0125 | 0.00904 | 0.0265 | low-count |
| P | control_day_10 | W | -2.664 | 1/732 | 0.0014 | 0.0087 | 0.0257 | 0.0513 | low-count |
| E | BWM_day_0 | C | -2.300 | 2/785 | 0.0025 | 0.0125 | 0.00575 | 0.023 | low-count |
| E | BWM_day_5 | A | -1.897 | 15/671 | 0.0224 | 0.0833 | 4.37e-11 | 4.37e-10 | low-count |
| P | control_day_0 | A | -1.847 | 14/605 | 0.0231 | 0.0833 | 9.23e-10 | 1.85e-08 | low-count |
| A | BWM_day_5 | A | -1.804 | 16/671 | 0.0238 | 0.0833 | 1.82e-10 | 3.64e-09 | low-count |
| P | BWM_day_5 | A | -1.804 | 16/671 | 0.0238 | 0.0833 | 1.82e-10 | 3.64e-09 | low-count |
| A | BWM_day_10 | A | -1.794 | 20/833 | 0.0240 | 0.0833 | 1.1e-12 | 2.2e-11 | low-count |

## Key Data — Codon level

- Tests run: **1098** · Significant (p_adj < 0.05): **163** (14.8%)
- Direction split (significant only): **79** favor **enriched**, **82** favor **depleted**

**Most significant (top 10 by p_adj)**

| Site | Group | Feature | log2_enrichment | Obs/Total | Obs freq | Bg freq | p_value | p_adj | Flags |
|---|---|---|---|---|---|---|---|---|---|
| E | BWM_day_0 | AAG | 1.133 | 165/785 | 0.2102 | 0.0958 | 9.76e-22 | 5.95e-20 |  |
| E | control_day_5 | AAG | 1.109 | 154/745 | 0.2067 | 0.0958 | 1.39e-19 | 8.46e-18 |  |
| P | control_day_10 | GAT | 1.918 | 62/732 | 0.0847 | 0.0224 | 1.59e-18 | 9.67e-17 |  |
| E | control_day_0 | AAG | 1.062 | 121/605 | 0.2000 | 0.0958 | 9.23e-15 | 5.63e-13 |  |
| P | BWM_day_10 | GAT | 1.636 | 58/833 | 0.0696 | 0.0224 | 9.89e-14 | 6.03e-12 |  |
| E | BWM_day_5 | AAG | 0.971 | 126/671 | 0.1878 | 0.0958 | 3.64e-13 | 2.22e-11 |  |
| E | BWM_day_10 | AAG | 0.851 | 144/833 | 0.1729 | 0.0958 | 5.17e-12 | 3.16e-10 |  |
| P | control_day_5 | GAT | 1.611 | 51/745 | 0.0685 | 0.0224 | 5.33e-12 | 3.25e-10 |  |
| P | BWM_day_0 | GAT | 1.448 | 48/785 | 0.0611 | 0.0224 | 9.35e-10 | 5.7e-08 | low-count |
| E | control_day_10 | AAG | 0.810 | 123/732 | 0.1680 | 0.0958 | 1.14e-09 | 6.95e-08 |  |

**Largest effect (top 10 by \|effect\|, all rows)**

| Site | Group | Feature | log2_enrichment | Obs/Total | Obs freq | Bg freq | p_value | p_adj | Flags |
|---|---|---|---|---|---|---|---|---|---|
| P | control_day_0 | ATA | 3.623 | 2/605 | 0.0033 | 0.0003 | 0.0118 | 0.0601 | low-count |
| P | BWM_day_5 | ATA | 3.473 | 2/671 | 0.0030 | 0.0003 | 0.0144 | 0.0626 | low-count |
| P | BWM_day_10 | TCC | -3.358 | 2/833 | 0.0024 | 0.0246 | 4.8e-07 | 1.46e-05 | low-count |
| P | control_day_5 | ATA | 3.322 | 2/745 | 0.0027 | 0.0003 | 0.0175 | 0.0762 | low-count |
| P | BWM_day_0 | ATA | 3.247 | 2/785 | 0.0025 | 0.0003 | 0.0193 | 0.0905 | low-count |
| P | BWM_day_5 | CAG | -3.206 | 1/671 | 0.0015 | 0.0138 | 0.00225 | 0.0125 | low-count |
| P | BWM_day_10 | ATA | 3.161 | 2/833 | 0.0024 | 0.0003 | 0.0215 | 0.0939 | low-count |
| E | BWM_day_0 | TGC | -3.039 | 1/785 | 0.0013 | 0.0105 | 0.0044 | 0.0298 | low-count |
| E | control_day_5 | TGC | -2.963 | 1/745 | 0.0013 | 0.0105 | 0.00624 | 0.0346 | low-count |
| A | control_day_0 | AGG | 2.815 | 2/605 | 0.0033 | 0.0005 | 0.0334 | 0.179 | low-count |

## Plots

**Amino Acid composites**

_Unweighted_

![BWM_volcano_grid](../plots/within_condition/composite/unweighted/BWM_volcano_grid.png)
![all_groups_volcano_grid](../plots/within_condition/composite/unweighted/all_groups_volcano_grid.png)
![control_volcano_grid](../plots/within_condition/composite/unweighted/control_volcano_grid.png)
![day_0_volcano_grid](../plots/within_condition/composite/unweighted/day_0_volcano_grid.png)
![day_10_volcano_grid](../plots/within_condition/composite/unweighted/day_10_volcano_grid.png)
![day_5_volcano_grid](../plots/within_condition/composite/unweighted/day_5_volcano_grid.png)

_Weighted_

![BWM_volcano_grid](../plots/within_condition/composite/weighted/BWM_volcano_grid.png)
![all_groups_volcano_grid](../plots/within_condition/composite/weighted/all_groups_volcano_grid.png)
![control_volcano_grid](../plots/within_condition/composite/weighted/control_volcano_grid.png)
![day_0_volcano_grid](../plots/within_condition/composite/weighted/day_0_volcano_grid.png)
![day_10_volcano_grid](../plots/within_condition/composite/weighted/day_10_volcano_grid.png)
![day_5_volcano_grid](../plots/within_condition/composite/weighted/day_5_volcano_grid.png)

Individual amino acid plots (36 files, not embedded): [`../plots/within_condition/individual`](../plots/within_condition/individual)

**Codon composites**

_Unweighted_

![BWM_volcano_grid](../plots/within_condition/codon/composite/unweighted/BWM_volcano_grid.png)
![all_groups_volcano_grid](../plots/within_condition/codon/composite/unweighted/all_groups_volcano_grid.png)
![control_volcano_grid](../plots/within_condition/codon/composite/unweighted/control_volcano_grid.png)
![day_0_volcano_grid](../plots/within_condition/codon/composite/unweighted/day_0_volcano_grid.png)
![day_10_volcano_grid](../plots/within_condition/codon/composite/unweighted/day_10_volcano_grid.png)
![day_5_volcano_grid](../plots/within_condition/codon/composite/unweighted/day_5_volcano_grid.png)

_Weighted_

![BWM_volcano_grid](../plots/within_condition/codon/composite/weighted/BWM_volcano_grid.png)
![all_groups_volcano_grid](../plots/within_condition/codon/composite/weighted/all_groups_volcano_grid.png)
![control_volcano_grid](../plots/within_condition/codon/composite/weighted/control_volcano_grid.png)
![day_0_volcano_grid](../plots/within_condition/codon/composite/weighted/day_0_volcano_grid.png)
![day_10_volcano_grid](../plots/within_condition/codon/composite/weighted/day_10_volcano_grid.png)
![day_5_volcano_grid](../plots/within_condition/codon/composite/weighted/day_5_volcano_grid.png)

Individual codon plots (36 files, not embedded): [`../plots/within_condition/codon/individual`](../plots/within_condition/codon/individual)

## Key Points

<!-- KEY_POINTS_START -->
- **Lysine, driven specifically by codon AAG, is enriched at the E-site in all six condition×timepoint groups** (BWM/control × day_0/5/10) — every one of the six group combinations appears in both the amino-acid and the codon "most significant" top-10 tables (AA log2_enrichment 0.84–1.18, codon log2 0.81–1.13; p_adj as low as 1.42e-24 for K, 5.95e-20 for AAG). This is the single most consistent, best-powered signal in the family, and it's specifically attributable to AAG — the other Lys codon (AAA) never appears in either top-10.

- **Aspartate / its codon GAT is enriched at the P-site.** The amino-acid table's top hits are strongest at day_10 in both conditions (control_day_10 log2=1.29, BWM_day_10 log2=1.23), but the codon-level table shows GAT recurring in 4 of the 6 groups (also control_day_5 and BWM_day_0) — so the P-site preference for Asp/GAT is not exclusive to day_10, it's just currently most significant there.

- **Alanine is recurrently depleted across all three sites (A, E, P), mostly at day_5/day_10, in both conditions** (e.g. BWM_day_10 A-site log2=−1.79 at obs 20/833; BWM_day_5 shows depletion at both A- and P-sites simultaneously, log2=−1.80 each). Every one of these rows is `low-count` (raw counts in the teens–20s), so the recurring direction across independent site/group combinations is more trustworthy than any single row's exact magnitude.

- **Cysteine is depleted at the E-site specifically at day_0 and day_5** (both conditions; log2 −2.30 to −3.23) but is absent from the day_10 top-10 — the depletion may ease by day_10, though that can't be confirmed without checking day_10's Cys rows directly (they don't clear either top-10 table). All four rows here are `low-count` (1–2 observed counts out of 600–800); at codon resolution the signal is driven by TGC specifically, and only the BWM_day_0 / control_day_5 TGC rows actually clear p_adj < 0.05.

- **The two most extreme depletions in the entire family aren't in the "largest effect" table at all**: codon TGC at the E-site and TCC at the P-site (both control_day_0) go to *zero* observed count against non-trivial backgrounds (1.05% and 2.46%), which is significant (p_adj = 0.040 and 1.6e-05) — but the upstream pipeline reports `log2_enrichment = 0.0` for any row where either the observed or background frequency is exactly 0, rather than −∞. Worth knowing about even though the magnitude ranking can't surface them.

- **ATA (Ile) at the P-site is nominally the single largest codon-level effect** (log2 up to 3.62, recurring in 5 of 6 groups) but none of those five rows clears p_adj < 0.05 (0.060–0.094) — a rare background codon (0.03%) means even 2 observed counts produces a huge log2 ratio without enough power to be significant. Read this as "possibly a real enrichment, not yet proven" rather than a confirmed hit.

- Roughly a third of amino-acid tests are significant (122/360, 33.9%) versus only 14.8% of codon tests (163/1098) — expected, since spreading the same stall-site counts across 61 codons instead of 20 amino acids shrinks per-feature counts and drives more rows into the `low-count` flag.
<!-- KEY_POINTS_END -->

## Caveats

- **FDR grouping:** p-values are Benjamini-Hochberg corrected per (group, site) — a row's `p_adj` is only comparable to other rows sharing that grouping.
- **Low-count threshold:** rows flagged `low-count` have a raw feature count below 50; treat their effect sizes as less reliable.

---
_Key Data, Plots, and Caveats are auto-generated by `result_interpretation_scripts/extract_key_data.py`
from `analysis/*.csv` and will be overwritten on the next run. Only the Key Points section (between
the KEY_POINTS markers above) is hand-authored and preserved across regenerations._
