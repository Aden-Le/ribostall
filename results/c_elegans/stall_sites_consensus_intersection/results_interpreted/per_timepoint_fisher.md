# Per-Timepoint Fisher's Exact (A3)

**Pipeline:** stall_sites_consensus_intersection (C. elegans)
**Test:** Fisher's exact test (two-sided), BWM vs control, at each timepoint independently, per E/P/A site (`ribostall.stats_core.fisher_row`, wrapped by `ribostall.enrichment.per_timepoint_fisher`). Null hypothesis: feature frequency at the stall site is independent of condition. Positive log2(odds ratio) favors BWM. Fisher's exact is a fair between-condition comparison here because the *intersection* transcript filtering gives every condition the same transcript universe, so raw stall-site shares are directly comparable.
**Source data:** `analysis/per_timepoint_fisher_aa.csv`, `analysis/per_timepoint_fisher_codon.csv`

## Amino Acid level

### Plots

![aa_fisher_composite](../plots/per_timepoint_fisher/composite/aa_fisher_composite.png)

Individual amino acid plots (9 files, not embedded): [`../plots/per_timepoint_fisher/individual`](../plots/per_timepoint_fisher/individual)

### Data

- Tests run: **180** · Significant (p_adj < 0.05): **1** (0.6%)
- Direction split (significant only): **0** favor **BWM**, **1** favor **control**

**Most significant (top 5 per site by p_adj)**

| Site | Timepoint | Feature | log2(OR) | Odds ratio | p_value | p_adj | BWM | control | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | day_10 | A | -1.232 | 0.426 | 0.00217 | 0.0433 | 20/833 | 40/732 | low-count |
| A | day_5 | L | 0.633 | 1.551 | 0.0873 | 0.688 | 41/671 | 30/745 | low-count |
| A | day_5 | D | 0.428 | 1.346 | 0.127 | 0.688 | 65/671 | 55/745 |  |
| A | day_5 | R | 0.445 | 1.361 | 0.145 | 0.688 | 54/671 | 45/745 | low-count |
| A | day_5 | F | -0.672 | 0.628 | 0.22 | 0.688 | 12/671 | 21/745 | low-count |
| E | day_10 | I | -0.889 | 0.540 | 0.00626 | 0.125 | 35/833 | 55/732 | low-count |
| E | day_5 | P | 0.824 | 1.770 | 0.127 | 0.81 | 22/671 | 14/745 | low-count |
| E | day_5 | E | -0.359 | 0.779 | 0.183 | 0.81 | 59/671 | 82/745 |  |
| E | day_5 | H | -0.863 | 0.550 | 0.216 | 0.81 | 8/671 | 16/745 | low-count |
| E | day_5 | L | 0.460 | 1.375 | 0.237 | 0.81 | 33/671 | 27/745 | low-count |
| P | day_10 | V | 0.812 | 1.755 | 0.0128 | 0.255 | 60/833 | 31/732 | low-count |
| P | day_0 | E | -0.616 | 0.653 | 0.0404 | 0.808 | 49/785 | 56/605 | low-count |
| P | day_10 | Y | -0.822 | 0.566 | 0.0873 | 0.873 | 17/833 | 26/732 | low-count |
| P | day_5 | K | 0.405 | 1.324 | 0.12 | 0.909 | 81/671 | 70/745 |  |
| P | day_5 | L | 0.380 | 1.301 | 0.261 | 0.909 | 45/671 | 39/745 | low-count |

**Largest effect (top 5 per site by \|effect\|, all rows)**

| Site | Timepoint | Feature | log2(OR) | Odds ratio | p_value | p_adj | BWM | control | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | day_10 | W | 1.408 | 2.654 | 0.155 | 0.741 | 9/833 | 3/732 | low-count |
| A | day_0 | M | 1.308 | 2.476 | 0.0651 | 0.764 | 19/785 | 6/605 | low-count |
| A | day_10 | A | -1.232 | 0.426 | 0.00217 | 0.0433 | 20/833 | 40/732 | low-count |
| A | day_0 | T | -0.836 | 0.560 | 0.0764 | 0.764 | 17/785 | 23/605 | low-count |
| A | day_10 | Q | -0.727 | 0.604 | 0.285 | 0.903 | 9/833 | 13/732 | low-count |
| E | day_5 | W | -1.176 | 0.442 | 0.456 | 0.914 | 2/671 | 5/745 | low-count |
| E | day_10 | W | 1.042 | 2.059 | 0.352 | 0.955 | 7/833 | 3/732 | low-count |
| E | day_0 | H | -0.978 | 0.508 | 0.106 | 1 | 10/785 | 15/605 | low-count |
| E | day_10 | I | -0.889 | 0.540 | 0.00626 | 0.125 | 35/833 | 55/732 | low-count |
| E | day_5 | H | -0.863 | 0.550 | 0.216 | 0.81 | 8/671 | 16/745 | low-count |
| P | day_10 | W | 1.402 | 2.642 | 0.628 | 1 | 3/833 | 1/732 | low-count |
| P | day_5 | W | 1.156 | 2.228 | 0.431 | 0.909 | 4/671 | 2/745 | low-count |
| P | day_10 | M | 0.826 | 1.772 | 0.219 | 1 | 16/833 | 8/732 | low-count |
| P | day_10 | Y | -0.822 | 0.566 | 0.0873 | 0.873 | 17/833 | 26/732 | low-count |
| P | day_10 | V | 0.812 | 1.755 | 0.0128 | 0.255 | 60/833 | 31/732 | low-count |

### Interpretation

<!-- INTERP_AA_START -->
- **Only 1 of 180 tests is significant: Alanine is depleted in BWM relative to control at the A-site, day_10** (log2(OR)=−1.232, BWM 20/833=2.4% vs control 40/732=5.5%, p_adj=0.0433). This is the single trustworthy signal in the whole amino-acid table.

- **The single strongest hit at every site is a day_10 comparison** — Ala at the A-site (significant, p_adj=0.0433), Ile at the E-site (log2(OR)=−0.889, p_adj=0.125), and Val at the P-site (log2(OR)=+0.812, p_adj=0.255). No day_0 or day_5 row comes close to significance at any site. day_10 is consistently where BWM and control diverge most in this comparison, even where two of the three site-level differences haven't yet cleared FDR.

- Beyond those three, every remaining row sits on a wide `p_adj` plateau (0.688–0.955) — the A-site's other four rows (Leu, Asp, Arg, Phe, all day_5) tie at 0.688; the E-/P-site day_5/day_0 rows cluster near 0.8–0.9. These are BH-FDR tie bands, not near-misses worth reading into individually.

- **The "largest effect" table is dominated by Tryptophan**, which lands in the top 5 of *every* site (5 of 15 rows total): A/day_10 (+1.41), E/day_5 (−1.18) and E/day_10 (+1.04) with opposite signs, P/day_10 (+1.40) and P/day_5 (+1.16). Trp has only one codon (TGG) and counts as low as 2–9 per arm — none of its five appearances are significant (p_adj 0.74–1.0). Treat this cluster as noise, not signal.
<!-- INTERP_AA_END -->

## Codon level

### Plots

![codon_fisher_composite](../plots/per_timepoint_fisher/codon/composite/codon_fisher_composite.png)

Individual codon plots (9 files, not embedded): [`../plots/per_timepoint_fisher/codon/individual`](../plots/per_timepoint_fisher/codon/individual)

### Data

- Tests run: **549** · Significant (p_adj < 0.05): **0** (0.0%)
- Direction split (significant only): **0** favor **BWM**, **0** favor **control**

**Most significant (top 5 per site by p_adj)**

| Site | Timepoint | Feature | log2(OR) | Odds ratio | p_value | p_adj | BWM | control | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | day_10 | GCC | -1.435 | 0.370 | 0.0149 | 0.911 | 9/833 | 21/732 | low-count |
| A | day_0 | ACC | -1.304 | 0.405 | 0.0539 | 1 | 8/785 | 15/605 | low-count |
| A | day_10 | GCT | -1.133 | 0.456 | 0.0586 | 1 | 10/833 | 19/732 | low-count |
| A | day_0 | ATG | 1.308 | 2.476 | 0.0651 | 1 | 19/785 | 6/605 | low-count |
| A | day_10 | GAG | -0.420 | 0.747 | 0.12 | 1 | 61/833 | 70/732 |  |
| E | day_10 | ATC | -1.143 | 0.453 | 0.00191 | 0.117 | 24/833 | 45/732 | low-count |
| E | day_0 | ATC | -0.559 | 0.679 | 0.121 | 1 | 36/785 | 40/605 | low-count |
| E | day_10 | CAT | 2.407 | 5.304 | 0.13 | 1 | 6/833 | 1/732 | low-count |
| E | day_0 | GCT | 1.124 | 2.179 | 0.163 | 1 | 14/785 | 5/605 | low-count |
| E | day_5 | GAG | -0.440 | 0.737 | 0.175 | 1 | 40/671 | 59/745 | low-count |
| P | day_10 | GTC | 1.781 | 3.436 | 0.00443 | 0.27 | 23/833 | 6/732 | low-count |
| P | day_0 | GAA | -1.318 | 0.401 | 0.0277 | 1 | 9/785 | 17/605 | low-count |
| P | day_10 | TAC | -0.823 | 0.565 | 0.1 | 1 | 15/833 | 23/732 | low-count |
| P | day_5 | AAT | 1.283 | 2.433 | 0.103 | 1 | 13/671 | 6/745 | low-count |
| P | day_0 | TCC | inf | inf | 0.137 | 1 | 4/785 | 0/605 | low-count |

**Largest effect (top 5 per site by \|effect\|, all rows)**

| Site | Timepoint | Feature | log2(OR) | Odds ratio | p_value | p_adj | BWM | control | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | day_5 | CTG | 2.158 | 4.462 | 0.196 | 1 | 4/671 | 1/745 | low-count |
| A | day_0 | CAT | 1.953 | 3.872 | 0.241 | 1 | 5/785 | 1/605 | low-count |
| A | day_5 | CAT | -1.666 | 0.315 | 0.184 | 1 | 2/671 | 7/745 | low-count |
| A | day_10 | GCC | -1.435 | 0.370 | 0.0149 | 0.911 | 9/833 | 21/732 | low-count |
| A | day_10 | TGG | 1.408 | 2.654 | 0.155 | 1 | 9/833 | 3/732 | low-count |
| E | day_10 | CAT | 2.407 | 5.304 | 0.13 | 1 | 6/833 | 1/732 | low-count |
| E | day_10 | GTG | 2.142 | 4.414 | 0.224 | 1 | 5/833 | 1/732 | low-count |
| E | day_0 | CCC | 1.629 | 3.093 | 0.395 | 1 | 4/785 | 1/605 | low-count |
| E | day_5 | GGC | -1.438 | 0.369 | 0.627 | 1 | 1/671 | 3/745 | low-count |
| E | day_0 | GTG | -1.385 | 0.383 | 0.189 | 1 | 3/785 | 6/605 | low-count |
| P | day_0 | ACA | -1.966 | 0.256 | 0.323 | 1 | 1/785 | 3/605 | low-count |
| P | day_5 | CAG | -1.855 | 0.276 | 0.378 | 1 | 1/671 | 4/745 | low-count |
| P | day_10 | GTC | 1.781 | 3.436 | 0.00443 | 0.27 | 23/833 | 6/732 | low-count |
| P | day_10 | TCG | -1.776 | 0.292 | 0.345 | 1 | 1/833 | 3/732 | low-count |
| P | day_0 | GTG | 1.629 | 3.093 | 0.395 | 1 | 4/785 | 1/605 | low-count |

_122 row(s) have a fully separated 2x2 table (one arm's count is 0), giving an undefined/infinite odds ratio; excluded from the table above and always low-count-flagged._

### Interpretation

<!-- INTERP_CODON_START -->
- **Not a single one of the 549 codon-level tests clears p_adj < 0.05.** The two closest are both at day_10, mirroring the amino-acid-level pattern: **ATC** (E-site, log2(OR)=−1.143, p_adj=0.117) and **GTC** (P-site, log2(OR)=+1.781, p_adj=0.27). No A-site codon comes close (best is GCC at p_adj=0.911).

- **The codon table explains where the one significant amino-acid hit comes from.** Alanine's depletion in BWM at the A-site/day_10 is carried almost entirely by two of its four codons: GCC (BWM 9/833 vs control 21/732) and GCT (BWM 10/833 vs control 19/732). Together they account for 19 of BWM's 20 Ala counts (95%) and *all* 40 of control's 40 Ala counts (100%) at that site/timepoint — the other two Ala codons (GCA, GCG) are essentially unused here.

- **ATC and GTC are also the specific codons anchoring the near-miss amino-acid signals**: ATC alone carries 24 of BWM's 35 E-site/day_10 Ile counts and 45 of control's 55 (vs. the aggregated Ile log2(OR)=−0.889); GTC shows a much larger swing (+1.78) than the aggregated P-site/day_10 Val signal (+0.812), meaning the other Val codons partly dilute it at the amino-acid level. (Sanity anchor: ATG/Met at A-site/day_0 has identical counts at both resolutions, 19/785 vs 6/605 — Met has only one codon.)

- **A concrete example of the "fully separated 2×2 table" caveat sits right in the P-site table**: TCC at day_0 has BWM=4/785 vs control=**0**/605 — an infinite odds ratio, since control never uses this codon at that site/timepoint. It still has a well-defined p-value (0.137, not significant) and so appears in the "most significant" ranking, but is excluded from "largest effect" (an undefined magnitude can't be ranked). 122 of 549 rows (22%) share this zero-count pattern.

- **The "largest effect" rows are uniformly rare-codon noise** (1–9 counts per arm, p_adj near 1 throughout). His (CAT) is the clearest example: it swings from +1.95 (A-site/day_0) to −1.67 (A-site/day_5) to +2.41 (E-site/day_10) — three different signs/magnitudes for the same codon, depending on which handful of stall sites happened to be sampled.
<!-- INTERP_CODON_END -->

## Key Points

<!-- KEY_POINTS_START -->
- **The one real finding in this family, at either resolution, is Alanine depletion in BWM at the A-site by day_10** — significant at the amino-acid level (p_adj=0.0433) and cleanly attributable at the codon level to GCC+GCT specifically, which together supply virtually all of both conditions' Ala counts at that site/timepoint (95% of BWM's, 100% of control's).

- **day_10 is where BWM and control diverge most, consistently, across every site and both resolutions** — not just for Alanine. The single closest-to-significant row at each site is a day_10 comparison (Ala/A-site: significant; Ile/E-site and Val/P-site: the two runners-up), and their codon-level anchors (GCC+GCT, ATC, GTC respectively) show the same pattern. day_0 and day_5 rows are uniformly far from significance at every site. Worth revisiting first if more replicates become available.

- **This connects to `within_condition_binomial`'s Alanine-depletion finding** — that family flagged the exact same BWM_day_10 A-site datapoint (20/833) as depleted relative to *background*. What this family adds: control's own measured rate at the same site/timepoint (40/732, roughly double BWM's) shows the depletion isn't purely a background-normalization artifact — BWM is specifically lower than control's actual observed rate, which strengthens the case for a real between-condition difference.

- **Signal is far sparser here than in `within_condition_binomial`**: 0.6% of amino-acid tests and 0% of codon tests are significant, versus 33.9%/14.8% there. Expected — comparing two conditions head-to-head (Fisher) is a harder bar than comparing each condition to its own background (binomial), and with only ~600–830 stall sites per condition/timepoint, this test has limited power to detect anything but the largest between-condition differences.

- **Two features (Trp at the amino-acid level, His at the codon level) repeatedly produce the largest \|effect\| values in this family purely because they're rare** — Trp lands in the top 5 largest-effect rows of every site, with counts as low as 2–9 per arm; His likewise swings sign three times at codon resolution (across the A- and E-sites) with counts as low as 1–7. Both flip direction across different site/timepoint slices with no consistent pattern — the signature of small-count noise, not biology. Treat any `low-count`-flagged "largest effect" row as a hypothesis to revisit with more data, not a finding.
<!-- KEY_POINTS_END -->

## Caveats

- **FDR grouping:** p-values are Benjamini-Hochberg corrected per (timepoint, site) — a row's `p_adj` is only comparable to other rows sharing that grouping.
- **Low-count threshold:** rows flagged `low-count` have a raw feature count below 50; treat their effect sizes as less reliable.

---
_Plots, Data, and Caveats are auto-generated by `result_interpretation_scripts/extract_key_data.py`
from `analysis/*.csv` and will be overwritten on the next run. The Interpretation (per level) and
Key Points (overall, at the bottom) sections are hand-authored and preserved across regenerations._
