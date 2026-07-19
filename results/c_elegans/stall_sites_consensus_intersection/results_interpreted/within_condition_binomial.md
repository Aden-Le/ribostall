# Within-Condition Binomial Enrichment (A1)

**Pipeline:** stall_sites_consensus_intersection (C. elegans)
**Test:** Two-sided binomial test (`scipy.stats.binomtest`) of the observed codon/amino-acid frequency at each E/P/A stall site against that group's background frequency, pooled within the group (`ribostall.stats_core.binom_row`, wrapped by `ribostall.enrichment.within_condition_enrichment`). Null hypothesis: the stall-site frequency of the feature equals genome background. Positive `log2_enrichment` means enriched relative to background; negative means depleted.
**Source data:** `analysis/within_condition_binomial_aa.csv`, `analysis/within_condition_binomial_codon.csv`

## Amino Acid level

### Plots

_Unweighted_

![all_groups_volcano_grid](../plots/within_condition/composite/unweighted/all_groups_volcano_grid.png)

Individual amino acid plots (36 files, not embedded): [`../plots/within_condition/individual`](../plots/within_condition/individual)

### Data

- Tests run: **360** · Significant (p_adj < 0.05): **122** (33.9%)
- Direction split (significant only): **55** favor **enriched**, **67** favor **depleted**

**Most significant (top 5 per site by p_adj)**

| Site | Group | Feature | log2_enrichment | Obs/Total | Obs freq | Bg freq | p_value | p_adj | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | BWM_day_10 | A | -1.794 | 20/833 | 0.0240 | 0.0833 | 1.1e-12 | 2.2e-11 | low-count |
| A | control_day_10 | E | 1.114 | 96/732 | 0.1311 | 0.0606 | 2e-12 | 4e-11 |  |
| A | BWM_day_5 | A | -1.804 | 16/671 | 0.0238 | 0.0833 | 1.82e-10 | 3.64e-09 | low-count |
| A | control_day_5 | E | 1.012 | 91/745 | 0.1221 | 0.0606 | 4.12e-10 | 8.24e-09 |  |
| A | control_day_0 | Y | 1.492 | 45/605 | 0.0744 | 0.0264 | 1.06e-09 | 2.13e-08 | low-count |
| E | BWM_day_0 | K | 1.179 | 183/785 | 0.2331 | 0.1030 | 7.09e-26 | 1.42e-24 |  |
| E | control_day_5 | K | 1.113 | 166/745 | 0.2228 | 0.1030 | 1.86e-21 | 3.73e-20 |  |
| E | control_day_0 | K | 1.083 | 132/605 | 0.2182 | 0.1030 | 1.26e-16 | 2.52e-15 |  |
| E | BWM_day_5 | K | 1.029 | 141/671 | 0.2101 | 0.1030 | 3.37e-16 | 6.73e-15 |  |
| E | BWM_day_10 | K | 0.835 | 153/833 | 0.1837 | 0.1030 | 3.02e-12 | 6.04e-11 |  |
| P | control_day_10 | D | 1.288 | 89/732 | 0.1216 | 0.0498 | 2.15e-14 | 4.3e-13 |  |
| P | BWM_day_10 | D | 1.226 | 97/833 | 0.1164 | 0.0498 | 2.51e-14 | 5.01e-13 |  |
| P | BWM_day_5 | A | -1.804 | 16/671 | 0.0238 | 0.0833 | 1.82e-10 | 3.64e-09 | low-count |
| P | BWM_day_0 | A | -1.507 | 23/785 | 0.0293 | 0.0833 | 8.29e-10 | 1.66e-08 | low-count |
| P | control_day_0 | A | -1.847 | 14/605 | 0.0231 | 0.0833 | 9.23e-10 | 1.85e-08 | low-count |

**Largest effect (top 5 per site by \|effect\|, all rows)**

| Site | Group | Feature | log2_enrichment | Obs/Total | Obs freq | Bg freq | p_value | p_adj | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | BWM_day_5 | A | -1.804 | 16/671 | 0.0238 | 0.0833 | 1.82e-10 | 3.64e-09 | low-count |
| A | BWM_day_10 | A | -1.794 | 20/833 | 0.0240 | 0.0833 | 1.1e-12 | 2.2e-11 | low-count |
| A | BWM_day_10 | Q | -1.638 | 9/833 | 0.0108 | 0.0336 | 4.53e-05 | 0.000181 | low-count |
| A | control_day_0 | A | -1.567 | 17/605 | 0.0281 | 0.0833 | 3.63e-08 | 3.63e-07 | low-count |
| A | BWM_day_0 | N | -1.518 | 11/785 | 0.0140 | 0.0401 | 3.6e-05 | 0.000144 | low-count |
| E | control_day_5 | C | -3.225 | 1/745 | 0.0013 | 0.0125 | 0.0015 | 0.00501 | low-count |
| E | BWM_day_5 | C | -3.074 | 1/671 | 0.0015 | 0.0125 | 0.00454 | 0.0182 | low-count |
| E | control_day_0 | C | -2.924 | 1/605 | 0.0017 | 0.0125 | 0.00904 | 0.0265 | low-count |
| E | BWM_day_0 | C | -2.300 | 2/785 | 0.0025 | 0.0125 | 0.00575 | 0.023 | low-count |
| E | BWM_day_5 | A | -1.897 | 15/671 | 0.0224 | 0.0833 | 4.37e-11 | 4.37e-10 | low-count |
| P | control_day_10 | W | -2.664 | 1/732 | 0.0014 | 0.0087 | 0.0257 | 0.0513 | low-count |
| P | control_day_0 | A | -1.847 | 14/605 | 0.0231 | 0.0833 | 9.23e-10 | 1.85e-08 | low-count |
| P | BWM_day_5 | A | -1.804 | 16/671 | 0.0238 | 0.0833 | 1.82e-10 | 3.64e-09 | low-count |
| P | control_day_5 | W | -1.689 | 2/745 | 0.0027 | 0.0087 | 0.107 | 0.18 | low-count |
| P | BWM_day_0 | A | -1.507 | 23/785 | 0.0293 | 0.0833 | 8.29e-10 | 1.66e-08 | low-count |

### Interpretation

<!-- INTERP_AA_START -->
- **Lysine (K) enrichment at the E-site is significant in all six condition×timepoint groups** — log2_enrichment 0.84–1.18, p_adj as low as 1.42e-24. The E-site's own top-5 table only has room for 5 of them; the sixth (control_day_10, p_adj=1.68e-10) is confirmed significant directly against the CSV, just narrowly outside the display.

- **Aspartate (D) enrichment at the P-site turns out to be just as universal** — also significant in all six groups (day_10 strongest in both conditions, log2≈1.23–1.29; the one group not shown in the top-5, BWM_day_5, is confirmed significant at p_adj=9.2e-7).

- **Unlike the E- and P-sites, no single amino acid dominates the A-site** — its top-5 splits three ways: Alanine depletion (BWM only, day_10 and day_5, log2≈−1.8), Glutamate enrichment (control only, day_10 and day_5, log2≈1.0–1.1), and Tyrosine enrichment (control_day_0, log2=1.49, low-count).

- **Alanine depletion recurs at the P-site too** (BWM_day_5, BWM_day_0, control_day_0 all land in its top-5) — a pattern spanning both the A- and P-sites and both conditions. Every one of these rows is independently significant despite being `low-count` (raw counts in the teens–20s).

- **Cysteine depletion at the E-site (day_0 and day_5, both conditions, log2 −2.30 to −3.23) is significant in all four of those rows** despite counts of only 1–2 — low count doesn't automatically mean "not significant" here, since the comparison is against a correspondingly small background frequency (≈1.25%). No day_10 Cys row appears in the E-site's top-5 by either ranking.

- **The "largest effect" table has a different texture per site**: at the A-site, all 5 rows are both large *and* significant (three Ala + a Gln + an Asn depletion, p_adj 1.8e-4 to 3.6e-9); at the P-site, 3 of 5 (Ala) are significant but 2 (Trp depletion, control only) are not — day_10's Trp row (p_adj=0.051) narrowly misses.
<!-- INTERP_AA_END -->

## Codon level

### Plots

_Unweighted_

![all_groups_volcano_grid](../plots/within_condition/codon/composite/unweighted/all_groups_volcano_grid.png)

Individual codon plots (36 files, not embedded): [`../plots/within_condition/codon/individual`](../plots/within_condition/codon/individual)

### Data

- Tests run: **1098** · Significant (p_adj < 0.05): **163** (14.8%)
- Direction split (significant only): **79** favor **enriched**, **82** favor **depleted**

**Most significant (top 5 per site by p_adj)**

| Site | Group | Feature | log2_enrichment | Obs/Total | Obs freq | Bg freq | p_value | p_adj | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | control_day_10 | GAG | 1.108 | 70/732 | 0.0956 | 0.0444 | 4.4e-09 | 2.68e-07 |  |
| A | control_day_0 | TAC | 1.472 | 41/605 | 0.0678 | 0.0244 | 8.79e-09 | 5.36e-07 | low-count |
| A | BWM_day_5 | AAG | 0.748 | 108/671 | 0.1610 | 0.0958 | 1.21e-07 | 7.38e-06 |  |
| A | BWM_day_10 | GCC | -1.982 | 9/833 | 0.0108 | 0.0427 | 1.8e-07 | 1.1e-05 | low-count |
| A | BWM_day_10 | GGA | 0.770 | 94/833 | 0.1128 | 0.0662 | 6.08e-07 | 1.85e-05 |  |
| E | BWM_day_0 | AAG | 1.133 | 165/785 | 0.2102 | 0.0958 | 9.76e-22 | 5.95e-20 |  |
| E | control_day_5 | AAG | 1.109 | 154/745 | 0.2067 | 0.0958 | 1.39e-19 | 8.46e-18 |  |
| E | control_day_0 | AAG | 1.062 | 121/605 | 0.2000 | 0.0958 | 9.23e-15 | 5.63e-13 |  |
| E | BWM_day_5 | AAG | 0.971 | 126/671 | 0.1878 | 0.0958 | 3.64e-13 | 2.22e-11 |  |
| E | BWM_day_10 | AAG | 0.851 | 144/833 | 0.1729 | 0.0958 | 5.17e-12 | 3.16e-10 |  |
| P | control_day_10 | GAT | 1.918 | 62/732 | 0.0847 | 0.0224 | 1.59e-18 | 9.67e-17 |  |
| P | BWM_day_10 | GAT | 1.636 | 58/833 | 0.0696 | 0.0224 | 9.89e-14 | 6.03e-12 |  |
| P | control_day_5 | GAT | 1.611 | 51/745 | 0.0685 | 0.0224 | 5.33e-12 | 3.25e-10 |  |
| P | BWM_day_0 | GAT | 1.448 | 48/785 | 0.0611 | 0.0224 | 9.35e-10 | 5.7e-08 | low-count |
| P | control_day_0 | GAT | 1.561 | 40/605 | 0.0661 | 0.0224 | 2.54e-09 | 1.55e-07 | low-count |

**Largest effect (top 5 per site by \|effect\|, all rows)**

| Site | Group | Feature | log2_enrichment | Obs/Total | Obs freq | Bg freq | p_value | p_adj | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | control_day_0 | AGG | 2.815 | 2/605 | 0.0033 | 0.0005 | 0.0334 | 0.179 | low-count |
| A | control_day_0 | CTA | 2.623 | 1/605 | 0.0017 | 0.0003 | 0.15 | 0.435 | low-count |
| A | BWM_day_5 | CTA | 2.473 | 1/671 | 0.0015 | 0.0003 | 0.165 | 0.457 | low-count |
| A | BWM_day_5 | ATA | 2.473 | 1/671 | 0.0015 | 0.0003 | 0.165 | 0.457 | low-count |
| A | BWM_day_0 | CCC | 2.372 | 3/785 | 0.0038 | 0.0007 | 0.0211 | 0.117 | low-count |
| E | BWM_day_0 | TGC | -3.039 | 1/785 | 0.0013 | 0.0105 | 0.0044 | 0.0298 | low-count |
| E | control_day_5 | TGC | -2.963 | 1/745 | 0.0013 | 0.0105 | 0.00624 | 0.0346 | low-count |
| E | BWM_day_5 | TGC | -2.812 | 1/671 | 0.0015 | 0.0105 | 0.0126 | 0.107 | low-count |
| E | BWM_day_0 | CCC | 2.787 | 4/785 | 0.0051 | 0.0007 | 0.00295 | 0.0257 | low-count |
| E | BWM_day_5 | ACG | 2.599 | 3/671 | 0.0045 | 0.0007 | 0.014 | 0.107 | low-count |
| P | control_day_0 | ATA | 3.623 | 2/605 | 0.0033 | 0.0003 | 0.0118 | 0.0601 | low-count |
| P | BWM_day_5 | ATA | 3.473 | 2/671 | 0.0030 | 0.0003 | 0.0144 | 0.0626 | low-count |
| P | BWM_day_10 | TCC | -3.358 | 2/833 | 0.0024 | 0.0246 | 4.8e-07 | 1.46e-05 | low-count |
| P | control_day_5 | ATA | 3.322 | 2/745 | 0.0027 | 0.0003 | 0.0175 | 0.0762 | low-count |
| P | BWM_day_0 | ATA | 3.247 | 2/785 | 0.0025 | 0.0003 | 0.0193 | 0.0905 | low-count |

### Interpretation

<!-- INTERP_CODON_START -->
- **AAG is the specific codon driving the E-site Lys enrichment** (top-5 there mirrors the amino-acid pattern: 5 of 6 groups shown, control_day_10 confirmed significant but just outside). AAG also shows up, more weakly, as one of the **A-site's** top hits (BWM_day_5, log2=0.748, p_adj=7.38e-6) — Lys enrichment isn't purely an E-site phenomenon.

- **GAT drives the P-site Asp enrichment and is significant in all six groups** (5 shown directly in its top-5; BWM_day_5 confirmed significant at p_adj=9.2e-7) — as universal a signal as AAG/Lys, just for a different amino acid and site.

- **GCC (one of Ala's four codons) shows an even larger depletion at the A-site/BWM_day_10** (log2=−1.982) than the aggregated amino-acid signal (−1.794) — the other three Ala codons partly offset it at the amino-acid level.

- **Of Cysteine's two codons, TGC alone explains 3 of the 4 significant E-site Cys rows** (BWM_day_0, control_day_5, and the zero-count control_day_0 case where `log2_enrichment=0.0` masks what's actually a complete absence). The fourth (BWM_day_5) only reaches significance once pooled with TGT at the amino-acid level — TGC's own background frequency there is roughly half of Cys's combined background, so the same tiny count (1) looks less extreme judged against TGC alone.

- **TCC at the P-site/BWM_day_10 is a real, significant depletion** (log2=−3.358, p_adj=1.46e-5, obs 2/833 vs background 2.46%) — distinct from the ATA cluster shown alongside it (control_day_0/BWM_day_5/control_day_5/BWM_day_0, log2 3.16–3.62), which is large but **never significant** (p_adj 0.060–0.094): a rare background codon (0.03%) inflates the ratio without giving the test enough power.
<!-- INTERP_CODON_END -->

## Key Points

<!-- KEY_POINTS_START -->
- **Two enrichment signals are each significant in all six condition×timepoint groups: E-site Lysine/AAG and P-site Aspartate/GAT** (each confirmed directly against the raw CSV where the per-site top-5 table only had room for 5 of 6). These are the two most robust, best-replicated findings in the family, and the codon view confirms each is a genuine single-codon effect — AAG for Lys, GAT for Asp — not just aggregate amino-acid drift.

- **The A-site behaves differently from the E- and P-sites**: instead of one dominant feature, it splits between Alanine depletion (BWM), Glutamate enrichment (control), and Tyrosine enrichment (control_day_0) — and unlike the E-/P-site signals, none of these recurs across all six groups.

- **"Low-count" and "not significant" are not the same thing here.** Every A-site largest-effect row and all four E-site Cys rows are significant despite tiny raw counts (1–20), because they're compared against a correspondingly small background frequency — while the P-site's ATA cluster (counts of 1–2) and Trp cluster (counts of 1–4) are large but genuinely underpowered (p_adj 0.05–0.9). Check both count *and* background frequency before dismissing (or trusting) a `low-count` row.

- Roughly a third of amino-acid tests are significant (122/360, 33.9%) versus only 14.8% of codon tests (163/1098) — expected, since spreading the same stall-site counts across 61 codons instead of 20 amino acids shrinks per-feature counts (and their corresponding backgrounds) and drives more rows into the `low-count` flag.
<!-- KEY_POINTS_END -->

## Caveats

- **FDR grouping:** p-values are Benjamini-Hochberg corrected per (group, site) — a row's `p_adj` is only comparable to other rows sharing that grouping.
- **Low-count threshold:** rows flagged `low-count` have a raw feature count below 50; treat their effect sizes as less reliable.

---
_Plots, Data, and Caveats are auto-generated by `result_interpretation_scripts/extract_key_data.py`
from `analysis/*.csv` and will be overwritten on the next run. The Interpretation (per level) and
Key Points (overall, at the bottom) sections are hand-authored and preserved across regenerations._
