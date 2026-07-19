# Timepoint Fisher's Exact Within Condition — day 5 vs day 0 (A6)

**Pipeline:** stall_sites_consensus_intersection (C. elegans)
**Test:** Fisher's exact test (two-sided), day_5 vs day_0, within each condition independently, per E/P/A site (`ribostall.enrichment.between_timepoint_fisher_within_condition`). Null hypothesis: feature frequency at the stall site is independent of timepoint, holding condition fixed. Positive log2(odds ratio) favors the later timepoint (day_5). Fair under the *intersection* design for the same reason as the per-timepoint comparison.
**Source data:** `analysis/timepoint_fisher_within_condition_d5_vs_d0_aa.csv`, `analysis/timepoint_fisher_within_condition_d5_vs_d0_codon.csv`

## Amino Acid level

### Plots

![aa_fisher_composite](../plots/within_condition_timepoint_fisher/d5_vs_d0/composite/aa_fisher_composite.png)

Individual amino acid plots (6 files, not embedded): [`../plots/within_condition_timepoint_fisher/d5_vs_d0/individual`](../plots/within_condition_timepoint_fisher/d5_vs_d0/individual)

### Data

- Tests run: **120** · Significant (p_adj < 0.05): **0** (0.0%)
- Direction split (significant only): **0** favor **day_5**, **0** favor **day_0**

**Most significant (top 5 per site by p_adj)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_5 | day_0 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | control | V | 0.800 | 1.742 | 0.0102 | 0.118 | 70/745 | 34/605 | low-count |
| A | control | P | -1.013 | 0.496 | 0.0137 | 0.118 | 22/745 | 35/605 | low-count |
| A | control | Y | -0.840 | 0.559 | 0.0178 | 0.118 | 32/745 | 45/605 | low-count |
| A | control | I | 0.605 | 1.521 | 0.0857 | 0.428 | 53/745 | 29/605 | low-count |
| A | control | R | -0.487 | 0.714 | 0.134 | 0.495 | 45/745 | 50/605 | low-count |
| E | control | P | -1.286 | 0.410 | 0.00658 | 0.132 | 14/745 | 27/605 | low-count |
| E | control | Q | 1.022 | 2.031 | 0.0483 | 0.336 | 27/745 | 11/605 | low-count |
| E | control | E | 0.554 | 1.468 | 0.0504 | 0.336 | 82/745 | 47/605 | low-count |
| E | control | V | -0.536 | 0.689 | 0.116 | 0.581 | 40/745 | 46/605 | low-count |
| E | BWM | E | 0.472 | 1.387 | 0.111 | 0.91 | 59/671 | 51/785 |  |
| P | control | K | -0.426 | 0.744 | 0.11 | 0.977 | 70/745 | 74/605 |  |
| P | control | Y | 0.776 | 1.712 | 0.146 | 0.977 | 27/745 | 13/605 | low-count |
| P | control | C | 1.449 | 2.730 | 0.161 | 0.977 | 10/745 | 3/605 | low-count |
| P | control | A | 0.610 | 1.527 | 0.259 | 0.977 | 26/745 | 14/605 | low-count |
| P | control | P | 0.428 | 1.346 | 0.345 | 0.977 | 36/745 | 22/605 | low-count |

**Largest effect (top 5 per site by \|effect\|, all rows)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_5 | day_0 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | control | W | -1.160 | 0.447 | 0.179 | 0.495 | 5/745 | 9/605 | low-count |
| A | BWM | M | -1.040 | 0.486 | 0.118 | 0.502 | 8/671 | 19/785 | low-count |
| A | control | P | -1.013 | 0.496 | 0.0137 | 0.118 | 22/745 | 35/605 | low-count |
| A | control | Y | -0.840 | 0.559 | 0.0178 | 0.118 | 32/745 | 45/605 | low-count |
| A | control | M | 0.826 | 1.773 | 0.353 | 0.698 | 13/745 | 6/605 | low-count |
| E | control | P | -1.286 | 0.410 | 0.00658 | 0.132 | 14/745 | 27/605 | low-count |
| E | control | Q | 1.022 | 2.031 | 0.0483 | 0.336 | 27/745 | 11/605 | low-count |
| E | BWM | W | -0.777 | 0.584 | 0.693 | 1 | 2/671 | 4/785 | low-count |
| E | BWM | C | -0.775 | 0.584 | 1 | 1 | 1/671 | 2/785 | low-count |
| E | BWM | A | -0.639 | 0.642 | 0.209 | 0.91 | 15/671 | 27/785 | low-count |
| P | control | C | 1.449 | 2.730 | 0.161 | 0.977 | 10/745 | 3/605 | low-count |
| P | control | W | -0.889 | 0.540 | 0.662 | 0.977 | 2/745 | 3/605 | low-count |
| P | control | Y | 0.776 | 1.712 | 0.146 | 0.977 | 27/745 | 13/605 | low-count |
| P | BWM | C | 0.718 | 1.645 | 0.403 | 1 | 7/671 | 5/785 | low-count |
| P | BWM | Q | -0.703 | 0.614 | 0.321 | 1 | 9/671 | 17/785 | low-count |

### Interpretation

<!-- INTERP_AA_START -->
- **Zero significant hits (0/120)** — the weakest signal of the three temporal comparisons in this pipeline (both other `timepoint_fisher_within_condition_*` families had 2/120).

- **Nearly all of the weak signal that does exist is in control, not BWM**: 14 of the 15 rows across all three sites' top-5-by-p_adj tables are control (the one BWM row, at the E-site, is also the least significant of its group). This mirrors — in reverse — `timepoint_fisher_within_condition_d10_vs_d0`, where every one of the A-site's top-5 rows was BWM. Control appears to drift from its own day_0 baseline earlier (already showing some signal by day_5); BWM's divergence shows up mostly by day_10.

- **Checked directly against the CSV (beyond what these non-significant tables show): Lysine's frequency is lowest at day_10 in all 6 site×condition combinations**, confirming the pattern flagged in the other two temporal families — but the day_0-to-day_5 step is genuinely mixed, not a clean intermediate point: already declining by day_5 in 3 of 6 combinations (E/BWM, P/BWM, P/control), roughly flat in 2 (A/control, E/control), and actually slightly *higher* at day_5 than day_0 in the sixth (A/BWM: 14.8%→17.3%→11.5%). day_10 is uniformly the trough; the path there isn't a straight line.

- **The mirror image holds for Glycine, whose frequency is highest at day_10 in all 6 combinations** — control rises steadily across all three timepoints at every site (a genuine monotonic increase), while BWM dips slightly at day_5 before a sharper rise to day_10 at every site.
<!-- INTERP_AA_END -->

## Codon level

### Plots

![codon_fisher_composite](../plots/within_condition_timepoint_fisher/d5_vs_d0/codon/composite/codon_fisher_composite.png)

Individual codon plots (6 files, not embedded): [`../plots/within_condition_timepoint_fisher/d5_vs_d0/codon/individual`](../plots/within_condition_timepoint_fisher/d5_vs_d0/codon/individual)

### Data

- Tests run: **366** · Significant (p_adj < 0.05): **0** (0.0%)
- Direction split (significant only): **0** favor **day_5**, **0** favor **day_0**

**Most significant (top 5 per site by p_adj)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_5 | day_0 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | control | TAC | -0.896 | 0.537 | 0.0131 | 0.469 | 28/745 | 41/605 | low-count |
| A | control | CCA | -1.018 | 0.494 | 0.0154 | 0.469 | 20/745 | 32/605 | low-count |
| A | control | GTC | 0.998 | 1.997 | 0.0305 | 0.619 | 36/745 | 15/605 | low-count |
| A | control | CAT | 2.518 | 5.729 | 0.0813 | 1 | 7/745 | 1/605 | low-count |
| A | BWM | TAC | -0.618 | 0.652 | 0.108 | 1 | 25/671 | 44/785 | low-count |
| E | control | CCA | -1.109 | 0.464 | 0.0302 | 1 | 14/745 | 24/605 | low-count |
| E | control | GTG | -2.898 | 0.134 | 0.05 | 1 | 1/745 | 6/605 | low-count |
| E | control | GAG | 0.623 | 1.540 | 0.0633 | 1 | 59/745 | 32/605 | low-count |
| E | control | CAA | 1.302 | 2.466 | 0.111 | 1 | 15/745 | 5/605 | low-count |
| E | BWM | GAG | 0.577 | 1.492 | 0.115 | 1 | 40/671 | 32/785 | low-count |
| P | BWM | TTG | 1.095 | 2.136 | 0.0568 | 1 | 18/671 | 10/785 | low-count |
| P | control | ATT | -1.318 | 0.401 | 0.0923 | 1 | 6/745 | 12/605 | low-count |
| P | control | TAC | 0.894 | 1.858 | 0.101 | 1 | 27/745 | 12/605 | low-count |
| P | control | AAG | -0.461 | 0.726 | 0.103 | 1 | 58/745 | 63/605 |  |
| P | BWM | GAA | 1.075 | 2.106 | 0.104 | 1 | 16/671 | 9/785 | low-count |

**Largest effect (top 5 per site by \|effect\|, all rows)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_5 | day_0 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | control | CAT | 2.518 | 5.729 | 0.0813 | 1 | 7/745 | 1/605 | low-count |
| A | BWM | GGT | -1.365 | 0.388 | 0.3 | 1 | 2/671 | 6/785 | low-count |
| A | BWM | CCC | -1.362 | 0.389 | 0.629 | 1 | 1/671 | 3/785 | low-count |
| A | control | ACA | -1.303 | 0.405 | 0.59 | 1 | 1/745 | 2/605 | low-count |
| A | control | CCC | -1.303 | 0.405 | 0.59 | 1 | 1/745 | 2/605 | low-count |
| E | control | GTG | -2.898 | 0.134 | 0.05 | 1 | 1/745 | 6/605 | low-count |
| E | BWM | CCC | -1.779 | 0.291 | 0.382 | 1 | 1/671 | 4/785 | low-count |
| E | BWM | GTG | -1.362 | 0.389 | 0.629 | 1 | 1/671 | 3/785 | low-count |
| E | control | CAT | -1.303 | 0.405 | 0.59 | 1 | 1/745 | 2/605 | low-count |
| E | control | CAA | 1.302 | 2.466 | 0.111 | 1 | 15/745 | 5/605 | low-count |
| P | BWM | CAG | -2.103 | 0.233 | 0.226 | 1 | 1/671 | 5/785 | low-count |
| P | control | TCA | -1.891 | 0.270 | 0.331 | 1 | 1/745 | 3/605 | low-count |
| P | control | TGC | 1.710 | 3.273 | 0.2 | 1 | 8/745 | 2/605 | low-count |
| P | control | ATT | -1.318 | 0.401 | 0.0923 | 1 | 6/745 | 12/605 | low-count |
| P | control | GTG | 1.288 | 2.442 | 0.632 | 1 | 3/745 | 1/605 | low-count |

_81 row(s) have a fully separated 2x2 table (one arm's count is 0), giving an undefined/infinite odds ratio; excluded from the table above and always low-count-flagged._

### Interpretation

<!-- INTERP_CODON_START -->
- **AAG remains the dominant codon behind the P-site Lysine pattern** (control: log2(OR)=−0.461, closely tracking the amino-acid-level −0.426) — consistent with its role in both other temporal comparisons in this pipeline.

- **The codon-level picture here is uniformly non-significant and control-skewed**, mirroring the amino-acid level: only two low-count BWM rows (P-site TTG, GAA) appear anywhere in the top-5-by-p_adj tables across all three sites.

- **Cysteine's E-site trajectory dips slightly from day_0 to day_5 in both conditions** (day_5 < day_0, both low-count: BWM 0.15% vs 0.25%, control 0.13% vs 0.17%) before rising sharply to day_10 (per the other two temporal families) — a trough-then-peak shape, not a steady climb toward day_10.

- 81 of 366 rows (22%) have a fully separated 2×2 table, consistent with the other two temporal-comparison families (21–22% each).
<!-- INTERP_CODON_END -->

## Key Points

<!-- KEY_POINTS_START -->
- **This family completes the three-way day_0/day_5/day_10 picture for Lysine and Glycine, and resolves what was flagged as an open question in `timepoint_fisher_within_condition_d10_vs_d5`.** Across all three `timepoint_fisher_within_condition_*` families: Lysine's frequency is lowest at day_10, and Glycine's is highest at day_10, in *every one* of the 6 site×condition combinations — the single most robust, fully cross-validated pattern in this whole pipeline. But the expected "partial day_0-to-day_5 step" isn't a clean intermediate point: it's already mixed at day_5 (Lys declining in half the combinations, flat or even reversed in the rest). The honest description is "day_10 is uniformly the extreme," not "a steady monotonic trend" — the path there varies by site and condition.

- **A consistent asymmetry runs through all three temporal families: control tends to diverge from its own day_0 baseline earlier (visible already by day_5, as in this family's top hits), while BWM's divergence concentrates later (by day_10, as in `timepoint_fisher_within_condition_d10_vs_d0`'s all-BWM A-site top-5).** This timing difference between conditions isn't captured by any single pairwise test — it only shows up by comparing across all three families, and would be worth a dedicated look if pursued further.

- **This also connects back to `within_condition_binomial`**: that family found Lysine enriched vs. background most strongly at day_0 and most weakly at day_10 (E-site, all six groups) — consistent with, and now fully explained by, the direct day_10-is-the-trough finding here and in the other two temporal comparisons.

- **This is the sparsest family in the entire pipeline**: 0/120 amino-acid and 0/366 codon tests are significant — expected, since day_5 is temporally closer to day_0 than day_10 is to either baseline, leaving the smallest window for a within-condition difference to accumulate.
<!-- KEY_POINTS_END -->

## Caveats

- **FDR grouping:** p-values are Benjamini-Hochberg corrected per (condition, site) — a row's `p_adj` is only comparable to other rows sharing that grouping.
- **Low-count threshold:** rows flagged `low-count` have a raw feature count below 50; treat their effect sizes as less reliable.

---
_Plots, Data, and Caveats are auto-generated by `result_interpretation_scripts/extract_key_data.py`
from `analysis/*.csv` and will be overwritten on the next run. The Interpretation (per level) and
Key Points (overall, at the bottom) sections are hand-authored and preserved across regenerations._
