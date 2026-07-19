# Within-Condition Binomial Enrichment (A1)

**Pipeline:** stall_sites_consensus_intersection (C. elegans)
**Test:** Two-sided binomial test (`scipy.stats.binomtest`) of the observed codon/amino-acid frequency at each E/P/A stall site against that group's background frequency, pooled within the group (`ribostall.stats_core.binom_row`, wrapped by `ribostall.enrichment.within_condition_enrichment`). Null hypothesis: the stall-site frequency of the feature equals genome background. Positive `log2_enrichment` means enriched relative to background; negative means depleted.
**Source data:** `analysis/within_condition_binomial_aa.csv`, `analysis/within_condition_binomial_codon.csv`

## Amino Acid level

### Plots

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

### Data

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

### Interpretation

<!-- INTERP_AA_START -->
- **Lysine (K) is enriched at the E-site across all six condition×timepoint groups** (BWM/control × day_0/5/10) — log2_enrichment 0.84–1.18, p_adj as low as 1.42e-24. The single most consistent, best-powered signal at this resolution.

- **Aspartate (D) is enriched at the P-site**, strongest at day_10 in both conditions (control_day_10 log2=1.29, BWM_day_10 log2=1.23).

- **Alanine is recurrently depleted across all three sites (A, E, P), mostly at day_5/day_10, in both conditions** (e.g. BWM_day_10 A-site log2=−1.79 at obs 20/833; BWM_day_5 shows depletion at both A- and P-sites simultaneously, log2=−1.80 each). Every one of these rows is `low-count` (raw counts in the teens–20s), so the recurring direction across independent site/group combinations is more trustworthy than any single row's exact magnitude.

- **Cysteine is depleted at the E-site specifically at day_0 and day_5** (both conditions; log2 −2.30 to −3.23) but is absent from the day_10 top-10 — the depletion may ease by day_10, though that can't be confirmed since neither top-10 table shows a day_10 Cys row. All four rows here are `low-count` (1–2 observed counts out of 600–800).
<!-- INTERP_AA_END -->

## Codon level

### Plots

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

### Data

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

### Interpretation

<!-- INTERP_CODON_START -->
- **AAG is the specific codon driving the E-site Lys enrichment** — it appears in all six groups' top-10 (log2 0.81–1.13, p_adj down to 5.95e-20), while the other Lys codon (AAA) never appears in either top-10. This rules out "just overall Lys usage" and points specifically at AAG.

- **GAT drives the P-site Asp enrichment and recurs in 4 of the 6 groups** (control_day_10, BWM_day_10, control_day_5, BWM_day_0) — broader than the amino-acid table's day_10-only top hits suggest, so this P-site preference isn't day_10-exclusive.

- **TGC (one of Cys's two codons) is what clears significance for the E-site Cys depletion** — only the BWM_day_0 and control_day_5 TGC rows reach p_adj < 0.05, consistent with the amino-acid-level day_0/day_5-only pattern.

- **The two most extreme depletions in this CSV aren't in the "largest effect" table at all**: TGC (E-site) and TCC (P-site), both in control_day_0, go to *zero* observed count against non-trivial backgrounds (1.05% and 2.46%) — genuinely significant (p_adj = 0.040 and 1.6e-05) but reported as `log2_enrichment = 0.0` by the upstream pipeline (guarded against log2(0) rather than returning −∞), so they don't surface in a magnitude-ranked table.

- **ATA (Ile) at the P-site is nominally the single largest codon-level effect** (log2 up to 3.62, recurring in 5 of 6 groups) but none of those five rows clears p_adj < 0.05 (0.060–0.094) — a rare background codon (0.03%) means even 2 observed counts produces a huge log2 ratio without enough power to be significant. Read as "possibly real, not yet proven."
<!-- INTERP_CODON_END -->

## Key Points

<!-- KEY_POINTS_START -->
- **The clearest, best-replicated finding in this family is E-site Lysine enrichment, and it holds at both resolutions**: amino-acid K and its specific codon AAG both show up in all six condition×timepoint groups (p_adj as low as 1.42e-24 for K, 5.95e-20 for AAG), while AAA (the other Lys codon) never appears — the codon view rules out "this is just overall Lys usage drift" and points specifically at AAG.

- **P-site Aspartate/GAT enrichment tells a similar but partial story**: strongest and most significant at day_10 in the amino-acid view, but the codon view shows GAT recurring in 4 of 6 groups — the preference likely isn't day_10-specific, day_10 just currently has the most power/signal.

- **Two of the largest-magnitude effects in this family turn out to be the least trustworthy or least visible.** ATA's huge log2 ratios (up to 3.62) are all non-significant — a rare-codon, low-count artifact. Meanwhile TGC/TCC's complete (zero-count) depletions are genuinely significant but invisible to the "largest effect" ranking, because the upstream pipeline reports `log2_enrichment = 0.0` (not −∞) whenever either frequency is exactly 0. Both are worth remembering as limits of a magnitude-only view — check significance and the underlying counts before trusting a "largest effect" row.

- Roughly a third of amino-acid tests are significant (122/360, 33.9%) versus only 14.8% of codon tests (163/1098) — expected, since spreading the same stall-site counts across 61 codons instead of 20 amino acids shrinks per-feature counts and drives more rows into the `low-count` flag.
<!-- KEY_POINTS_END -->

## Caveats

- **FDR grouping:** p-values are Benjamini-Hochberg corrected per (group, site) — a row's `p_adj` is only comparable to other rows sharing that grouping.
- **Low-count threshold:** rows flagged `low-count` have a raw feature count below 50; treat their effect sizes as less reliable.

---
_Plots, Data, and Caveats are auto-generated by `result_interpretation_scripts/extract_key_data.py`
from `analysis/*.csv` and will be overwritten on the next run. The Interpretation (per level) and
Key Points (overall, at the bottom) sections are hand-authored and preserved across regenerations._
