# Timepoint Fisher's Exact Within Condition — day 10 vs day 5 (A6)

**Pipeline:** stall_sites_consensus_intersection (C. elegans)
**Test:** Fisher's exact test (two-sided), day_10 vs day_5, within each condition independently, per E/P/A site (`ribostall.enrichment.between_timepoint_fisher_within_condition`). Null hypothesis: feature frequency at the stall site is independent of timepoint, holding condition fixed. Positive log2(odds ratio) favors the later timepoint (day_10). Fair under the *intersection* design for the same reason as the per-timepoint comparison.
**Source data:** `analysis/timepoint_fisher_within_condition_d10_vs_d5_aa.csv`, `analysis/timepoint_fisher_within_condition_d10_vs_d5_codon.csv`

## Amino Acid level

### Plots

![aa_fisher_composite](../plots/within_condition_timepoint_fisher/d10_vs_d5/composite/aa_fisher_composite.png)

Individual amino acid plots (6 files, not embedded): [`../plots/within_condition_timepoint_fisher/d10_vs_d5/individual`](../plots/within_condition_timepoint_fisher/d10_vs_d5/individual)

### Data

- Tests run: **120** · Significant (p_adj < 0.05): **2** (1.7%)
- Direction split (significant only): **1** favor **day_10**, **1** favor **day_5**

**Most significant (top 5 per site by p_adj)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_10 | day_5 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | BWM | G | 0.989 | 1.985 | 0.000178 | 0.00356 | 104/833 | 45/671 | low-count |
| A | BWM | K | -0.682 | 0.623 | 0.0017 | 0.017 | 96/833 | 116/671 |  |
| A | control | K | -0.518 | 0.698 | 0.0214 | 0.281 | 82/732 | 114/745 |  |
| A | control | A | 0.860 | 1.815 | 0.0281 | 0.281 | 40/732 | 23/745 | low-count |
| A | BWM | Q | -1.251 | 0.420 | 0.0447 | 0.298 | 9/833 | 17/671 | low-count |
| E | control | R | -0.811 | 0.570 | 0.0123 | 0.246 | 33/732 | 57/745 | low-count |
| E | BWM | I | -0.845 | 0.557 | 0.0125 | 0.251 | 35/833 | 49/671 | low-count |
| E | BWM | G | 0.634 | 1.552 | 0.0357 | 0.288 | 71/833 | 38/671 | low-count |
| E | BWM | H | 1.235 | 2.353 | 0.0432 | 0.288 | 23/833 | 8/671 | low-count |
| E | BWM | A | 0.851 | 1.804 | 0.0756 | 0.378 | 33/833 | 15/671 | low-count |
| P | control | G | 0.649 | 1.568 | 0.00704 | 0.0788 | 101/732 | 69/745 |  |
| P | control | V | -0.878 | 0.544 | 0.00788 | 0.0788 | 31/732 | 56/745 | low-count |
| P | BWM | K | -0.559 | 0.679 | 0.0253 | 0.505 | 71/833 | 81/671 |  |
| P | BWM | G | 0.413 | 1.332 | 0.0963 | 0.717 | 101/833 | 63/671 |  |
| P | BWM | N | -0.429 | 0.743 | 0.17 | 0.717 | 47/833 | 50/671 | low-count |

**Largest effect (top 5 per site by \|effect\|, all rows)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_10 | day_5 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | BWM | Q | -1.251 | 0.420 | 0.0447 | 0.298 | 9/833 | 17/671 | low-count |
| A | control | H | -0.990 | 0.503 | 0.149 | 0.991 | 8/732 | 16/745 | low-count |
| A | BWM | G | 0.989 | 1.985 | 0.000178 | 0.00356 | 104/833 | 45/671 | low-count |
| A | control | A | 0.860 | 1.815 | 0.0281 | 0.281 | 40/732 | 23/745 | low-count |
| A | BWM | M | 0.788 | 1.727 | 0.228 | 0.652 | 17/833 | 8/671 | low-count |
| E | BWM | C | 2.281 | 4.861 | 0.139 | 0.537 | 6/833 | 1/671 | low-count |
| E | control | C | 1.614 | 3.062 | 0.37 | 1 | 3/732 | 1/745 | low-count |
| E | BWM | W | 1.503 | 2.835 | 0.313 | 0.67 | 7/833 | 2/671 | low-count |
| E | BWM | H | 1.235 | 2.353 | 0.0432 | 0.288 | 23/833 | 8/671 | low-count |
| E | control | P | 0.943 | 1.923 | 0.0544 | 0.441 | 26/732 | 14/745 | low-count |
| P | control | W | -0.977 | 0.508 | 1 | 1 | 1/732 | 2/745 | low-count |
| P | control | V | -0.878 | 0.544 | 0.00788 | 0.0788 | 31/732 | 56/745 | low-count |
| P | BWM | W | -0.730 | 0.603 | 0.707 | 1 | 3/833 | 4/671 | low-count |
| P | BWM | A | 0.662 | 1.582 | 0.179 | 0.717 | 31/833 | 16/671 | low-count |
| P | control | G | 0.649 | 1.568 | 0.00704 | 0.0788 | 101/732 | 69/745 |  |

### Interpretation

<!-- INTERP_AA_START -->
- **Two significant hits, both at the A-site in BWM**: Glycine is enriched at day_10 (log2(OR)=0.989, day_10 104/833 vs day_5 45/671, p_adj=0.0036) and Lysine is depleted at day_10 (log2(OR)=−0.682, day_10 96/833 vs day_5 116/671, p_adj=0.017).

- **Both patterns hold across every site and both conditions, verified directly against the CSV** — not just the two significant A-site rows. Lysine is lower at day_10 than day_5 in **all 6** site×condition combinations (A/BWM −0.682\*, A/control −0.518, E/BWM −0.243, E/control −0.320, P/BWM −0.559, P/control −0.324); Glycine is higher at day_10 in **all 6** as well (A/BWM +0.989\*, A/control +0.263, E/BWM +0.635, E/control +0.058, P/BWM +0.413, P/control +0.649). Only the two starred A-site/BWM rows are individually significant, but the site- and condition-spanning consistency of both directions is itself a notable pattern.

- **Cysteine shows a small, non-significant day_10-enrichment at the E-site in both conditions** (log2 1.6–2.3, low-count) — the same direction and site seen in the day_10-vs-day_0 comparison (`timepoint_fisher_within_condition_d10_vs_d0`), suggesting Cys may be trending upward across the whole time course, though never with enough counts in any single pairwise test to be significant.
<!-- INTERP_AA_END -->

## Codon level

### Plots

![codon_fisher_composite](../plots/within_condition_timepoint_fisher/d10_vs_d5/codon/composite/codon_fisher_composite.png)

Individual codon plots (6 files, not embedded): [`../plots/within_condition_timepoint_fisher/d10_vs_d5/codon/individual`](../plots/within_condition_timepoint_fisher/d10_vs_d5/codon/individual)

### Data

- Tests run: **366** · Significant (p_adj < 0.05): **1** (0.3%)
- Direction split (significant only): **1** favor **day_10**, **0** favor **day_5**

**Most significant (top 5 per site by p_adj)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_10 | day_5 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | BWM | GGA | 0.930 | 1.905 | 0.000774 | 0.0472 | 94/833 | 42/671 | low-count |
| A | BWM | AAG | -0.628 | 0.647 | 0.00466 | 0.142 | 92/833 | 108/671 |  |
| A | BWM | CAA | -1.921 | 0.264 | 0.0107 | 0.218 | 5/833 | 15/671 | low-count |
| A | control | GCC | 1.444 | 2.721 | 0.0144 | 0.877 | 21/732 | 8/745 | low-count |
| A | control | AAG | -0.460 | 0.727 | 0.0484 | 0.997 | 78/732 | 105/745 |  |
| E | BWM | ATC | -1.057 | 0.481 | 0.00618 | 0.377 | 24/833 | 39/671 | low-count |
| E | control | AGA | -1.101 | 0.466 | 0.017 | 1 | 15/732 | 32/745 | low-count |
| E | BWM | GGA | 0.658 | 1.578 | 0.0414 | 1 | 61/833 | 32/671 | low-count |
| E | control | AAG | -0.368 | 0.775 | 0.0619 | 1 | 123/732 | 154/745 |  |
| E | control | AGC | -inf | 0.000 | 0.0621 | 1 | 0/732 | 5/745 | low-count |
| P | control | GGA | 0.768 | 1.703 | 0.0029 | 0.169 | 89/732 | 56/745 |  |
| P | control | GTC | -1.811 | 0.285 | 0.00554 | 0.169 | 6/732 | 21/745 | low-count |
| P | control | GCA | inf | inf | 0.0601 | 1 | 4/732 | 0/745 | low-count |
| P | BWM | AAA | -1.007 | 0.497 | 0.11 | 1 | 10/833 | 16/671 | low-count |
| P | BWM | AAG | -0.441 | 0.737 | 0.111 | 1 | 61/833 | 65/671 |  |

**Largest effect (top 5 per site by \|effect\|, all rows)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_10 | day_5 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | BWM | CTG | -2.319 | 0.200 | 0.179 | 1 | 1/833 | 4/671 | low-count |
| A | BWM | CAA | -1.921 | 0.264 | 0.0107 | 0.218 | 5/833 | 15/671 | low-count |
| A | BWM | GGT | 1.698 | 3.244 | 0.2 | 1 | 8/833 | 2/671 | low-count |
| A | control | GCC | 1.444 | 2.721 | 0.0144 | 0.877 | 21/732 | 8/745 | low-count |
| A | control | CGC | 1.366 | 2.577 | 0.049 | 0.997 | 15/732 | 6/745 | low-count |
| E | BWM | CAT | 2.281 | 4.861 | 0.139 | 1 | 6/833 | 1/671 | low-count |
| E | BWM | GTG | 2.016 | 4.046 | 0.234 | 1 | 5/833 | 1/671 | low-count |
| E | BWM | TGC | 2.016 | 4.046 | 0.234 | 1 | 5/833 | 1/671 | low-count |
| E | BWM | GCA | 1.693 | 3.233 | 0.389 | 1 | 4/833 | 1/671 | low-count |
| E | BWM | GGC | 1.693 | 3.233 | 0.389 | 1 | 4/833 | 1/671 | low-count |
| P | BWM | CAG | 2.016 | 4.046 | 0.234 | 1 | 5/833 | 1/671 | low-count |
| P | BWM | TCG | -1.902 | 0.268 | 0.33 | 1 | 1/833 | 3/671 | low-count |
| P | control | GTC | -1.811 | 0.285 | 0.00554 | 0.169 | 6/732 | 21/745 | low-count |
| P | BWM | TCC | -1.317 | 0.401 | 0.416 | 1 | 2/833 | 4/671 | low-count |
| P | BWM | TCA | -1.315 | 0.402 | 0.589 | 1 | 1/833 | 2/671 | low-count |

_82 row(s) have a fully separated 2x2 table (one arm's count is 0), giving an undefined/infinite odds ratio; excluded from the table above and always low-count-flagged._

### Interpretation

<!-- INTERP_CODON_START -->
- **GGA is the single codon behind the significant A-site Gly enrichment** (log2(OR)=0.930, p_adj=0.0472) — and behind the broader Gly pattern at every other site checked too: it supplies 90%/93% of BWM's A-site Gly counts (day_10/day_5), 86%/84% of BWM's E-site counts, and 88%/81% of control's P-site counts. Glycine's other three codons (GGC, GGG, GGT) contribute almost nothing to this signal anywhere.

- **AAG is similarly the dominant codon behind Lysine's decline**: 80–93% of the amino-acid-level Lys counts at every site/condition checked (A/BWM, E/control, P/BWM), with AAA (the other Lys codon) moving the same direction at the P-site/BWM.

- **Aggregating across synonymous codons is again what creates amino-acid-level significance that no single codon quite reaches alone** — GGA's own p_adj (0.047) barely clears the bar for Gly, and AAG's p_adj (0.14–1.0 across sites) never does for Lys, even though the amino-acid-level Lys hit is significant at the A-site. Same pattern seen in `timepoint_fisher_within_condition_d10_vs_d0`.

- 82 of 366 rows (22%) have a fully separated 2×2 table; the largest-magnitude codon rows are uniformly tiny-count noise (1–8 per arm), including coincidental exact ties (e.g. two different E-site codons both landing on log2(OR)=2.016 from an identical 5-vs-1 count split).
<!-- INTERP_CODON_END -->

## Key Points

<!-- KEY_POINTS_START -->
- **The cleanest, most site-consistent finding in this pipeline so far**: Glycine (via codon GGA) rises and Lysine (via AAG, plus AAA) falls from day_5 to day_10, in the same direction at every one of the 6 site×condition combinations, each codon individually accounting for 80–93% of its amino acid's signal wherever checked. Only the A-site/BWM cases reach significance for either feature, but the consistency everywhere else is hard to attribute to chance.

- **This connects to `timepoint_fisher_within_condition_d10_vs_d0`**: Lysine also declined from day_0 to day_10 there (P-site significant, E-site non-significant) — combined with `within_condition_binomial`'s finding that Lys-vs-background enrichment is strongest at day_0 and weakest at day_10, there's now a three-family-consistent picture of Lysine gradually declining across the whole day_0→day_5→day_10 time course. Worth checking `timepoint_fisher_within_condition_d5_vs_d0` next for the expected partial (day_0-to-day_5) piece of that decline.

- **Cysteine keeps recurring as a non-significant, low-count "watch this" pattern**: enriched at day_10 relative to *both* day_0 (previous family) and day_5 (this family) at the E-site, in both conditions — never enough counts to be significant in any single comparison, but consistent enough across two independent pairwise tests to be worth a specific look if more replicates become available.

- Signal is still sparse but slightly stronger than `timepoint_fisher_within_condition_d10_vs_d0`: 2/120 amino-acid tests (1.7%, same rate) but 1/366 codon tests (0.3%, vs 0% there) are significant.
<!-- KEY_POINTS_END -->

## Caveats

- **FDR grouping:** p-values are Benjamini-Hochberg corrected per (condition, site) — a row's `p_adj` is only comparable to other rows sharing that grouping.
- **Low-count threshold:** rows flagged `low-count` have a raw feature count below 50; treat their effect sizes as less reliable.

---
_Plots, Data, and Caveats are auto-generated by `result_interpretation_scripts/extract_key_data.py`
from `analysis/*.csv` and will be overwritten on the next run. The Interpretation (per level) and
Key Points (overall, at the bottom) sections are hand-authored and preserved across regenerations._
