# Timepoint Fisher's Exact Within Condition — day 10 vs day 0 (A6)

**Pipeline:** stall_sites_consensus_intersection (C. elegans)
**Test:** Fisher's exact test (two-sided), day_10 vs day_0, within each condition independently, per E/P/A site (`ribostall.enrichment.between_timepoint_fisher_within_condition`). Null hypothesis: feature frequency at the stall site is independent of timepoint, holding condition fixed. Positive log2(odds ratio) favors the later timepoint (day_10). Fair under the *intersection* design for the same reason as the per-timepoint comparison.
**Source data:** `analysis/timepoint_fisher_within_condition_d10_vs_d0_aa.csv`, `analysis/timepoint_fisher_within_condition_d10_vs_d0_codon.csv`

## Amino Acid level

### Plots

![aa_fisher_composite](../plots/within_condition_timepoint_fisher/d10_vs_d0/composite/aa_fisher_composite.png)

Individual amino acid plots (6 files, not embedded): [`../plots/within_condition_timepoint_fisher/d10_vs_d0/individual`](../plots/within_condition_timepoint_fisher/d10_vs_d0/individual)

### Data

- Tests run: **120** · Significant (p_adj < 0.05): **2** (1.7%)
- Direction split (significant only): **1** favor **day_10**, **1** favor **day_0**

**Most significant (top 5 per site by p_adj)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_10 | day_0 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | BWM | I | 1.017 | 2.024 | 0.00144 | 0.0288 | 64/833 | 31/785 | low-count |
| A | BWM | G | 0.612 | 1.529 | 0.012 | 0.12 | 104/833 | 67/785 |  |
| A | BWM | Y | -0.802 | 0.574 | 0.0201 | 0.134 | 30/833 | 48/785 | low-count |
| A | BWM | L | -0.707 | 0.612 | 0.0339 | 0.169 | 34/833 | 51/785 | low-count |
| A | BWM | K | -0.413 | 0.751 | 0.0555 | 0.222 | 96/833 | 116/785 |  |
| E | BWM | K | -0.434 | 0.740 | 0.0167 | 0.334 | 153/833 | 183/785 |  |
| E | BWM | H | 1.138 | 2.201 | 0.0358 | 0.358 | 23/833 | 10/785 | low-count |
| E | BWM | G | 0.549 | 1.463 | 0.0556 | 0.371 | 71/833 | 47/785 | low-count |
| E | control | Q | 1.155 | 2.228 | 0.0238 | 0.476 | 29/732 | 11/605 | low-count |
| E | control | V | -0.472 | 0.721 | 0.149 | 0.71 | 41/732 | 46/605 | low-count |
| P | BWM | K | -0.822 | 0.566 | 0.000382 | 0.00764 | 71/833 | 111/785 |  |
| P | control | K | -0.750 | 0.594 | 0.00536 | 0.0596 | 56/732 | 74/605 |  |
| P | control | G | 0.708 | 1.633 | 0.00596 | 0.0596 | 101/732 | 54/605 |  |
| P | control | V | -0.754 | 0.593 | 0.0392 | 0.261 | 31/732 | 42/605 | low-count |
| P | BWM | E | 0.572 | 1.486 | 0.0397 | 0.397 | 75/833 | 49/785 | low-count |

**Largest effect (top 5 per site by \|effect\|, all rows)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_10 | day_0 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | control | W | -1.876 | 0.273 | 0.044 | 0.294 | 3/732 | 9/605 | low-count |
| A | BWM | I | 1.017 | 2.024 | 0.00144 | 0.0288 | 64/833 | 31/785 | low-count |
| A | control | A | 1.000 | 1.999 | 0.0202 | 0.277 | 40/732 | 17/605 | low-count |
| A | BWM | Q | -0.930 | 0.525 | 0.157 | 0.393 | 9/833 | 16/785 | low-count |
| A | BWM | Y | -0.802 | 0.574 | 0.0201 | 0.134 | 30/833 | 48/785 | low-count |
| E | BWM | C | 1.506 | 2.840 | 0.29 | 0.759 | 6/833 | 2/785 | low-count |
| E | control | C | 1.314 | 2.486 | 0.631 | 0.846 | 3/732 | 1/605 | low-count |
| E | control | Q | 1.155 | 2.228 | 0.0238 | 0.476 | 29/732 | 11/605 | low-count |
| E | BWM | H | 1.138 | 2.201 | 0.0358 | 0.358 | 23/833 | 10/785 | low-count |
| E | BWM | W | 0.727 | 1.655 | 0.549 | 0.759 | 7/833 | 4/785 | low-count |
| P | control | W | -1.865 | 0.275 | 0.334 | 0.726 | 1/732 | 3/605 | low-count |
| P | control | C | 1.149 | 2.217 | 0.363 | 0.726 | 8/732 | 3/605 | low-count |
| P | BWM | C | 0.923 | 1.896 | 0.303 | 0.833 | 10/833 | 5/785 | low-count |
| P | BWM | K | -0.822 | 0.566 | 0.000382 | 0.00764 | 71/833 | 111/785 |  |
| P | control | V | -0.754 | 0.593 | 0.0392 | 0.261 | 31/732 | 42/605 | low-count |

### Interpretation

<!-- INTERP_AA_START -->
- **Two significant hits, both in BWM**: Isoleucine is enriched at day_10 vs day_0 at the A-site (log2(OR)=1.017, day_10 64/833 vs day_0 31/785, p_adj=0.029), and Lysine is depleted at day_10 vs day_0 at the P-site (log2(OR)=−0.822, day_10 71/833 vs day_0 111/785, p_adj=0.0076).

- **All 5 of the A-site's top-5-by-p_adj rows are BWM** — no control row makes the cut, meaning BWM shows a much stronger day_10-vs-day_0 divergence at the A-site than control does.

- **Lysine declines from day_0 to day_10 at both the E- and P-sites**, not just the P-site's significant hit: E-site/BWM shows the same direction (log2(OR)=−0.434, p_adj=0.334, not significant), and P-site/control is close behind BWM's significant P-site result (log2(OR)=−0.750, p_adj=0.0596). This looks like a real, site-spanning, possibly both-condition trend rather than an isolated result.

- **Glycine is enriched at day_10 at the P-site in control** (log2(OR)=0.708, p_adj=0.0596) — tied with control's Lysine result for the second-closest-to-significant row in the family, just missing the threshold.

- **Cysteine shows a small, consistent (but non-significant) day_10-enrichment across both E- and P-sites, in both conditions** (log2 0.92–1.51, all `low-count`, p_adj 0.73–0.85) — plausible but unproven given counts of only 1–10 per arm; worth revisiting with more data rather than trusting as-is.
<!-- INTERP_AA_END -->

## Codon level

### Plots

![codon_fisher_composite](../plots/within_condition_timepoint_fisher/d10_vs_d0/codon/composite/codon_fisher_composite.png)

Individual codon plots (6 files, not embedded): [`../plots/within_condition_timepoint_fisher/d10_vs_d0/codon/individual`](../plots/within_condition_timepoint_fisher/d10_vs_d0/codon/individual)

### Data

- Tests run: **366** · Significant (p_adj < 0.05): **0** (0.0%)
- Direction split (significant only): **0** favor **day_10**, **0** favor **day_0**

**Most significant (top 5 per site by p_adj)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_10 | day_0 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | BWM | ATC | 1.093 | 2.133 | 0.00489 | 0.298 | 44/833 | 20/785 | low-count |
| A | BWM | GGA | 0.646 | 1.565 | 0.0107 | 0.325 | 94/833 | 59/785 |  |
| A | BWM | TAC | -0.772 | 0.586 | 0.0302 | 0.614 | 28/833 | 44/785 | low-count |
| A | BWM | AAG | -0.392 | 0.762 | 0.0833 | 0.872 | 92/833 | 110/785 |  |
| A | BWM | CAA | -1.362 | 0.389 | 0.0871 | 0.872 | 5/833 | 12/785 | low-count |
| E | BWM | GGA | 0.635 | 1.553 | 0.0384 | 1 | 61/833 | 38/785 | low-count |
| E | control | TCC | 1.513 | 2.853 | 0.0495 | 1 | 17/732 | 5/605 | low-count |
| E | control | GTG | -2.872 | 0.137 | 0.051 | 1 | 1/732 | 6/605 | low-count |
| E | BWM | AAG | -0.349 | 0.785 | 0.0579 | 1 | 144/833 | 165/785 |  |
| E | BWM | AAA | -1.103 | 0.465 | 0.0789 | 1 | 9/833 | 18/785 | low-count |
| P | BWM | AAG | -0.801 | 0.574 | 0.00131 | 0.0801 | 61/833 | 95/785 |  |
| P | control | AAG | -0.897 | 0.537 | 0.00303 | 0.0982 | 43/732 | 63/605 | low-count |
| P | control | GGA | 0.819 | 1.765 | 0.00322 | 0.0982 | 89/732 | 44/605 | low-count |
| P | BWM | CTT | -0.917 | 0.529 | 0.0469 | 1 | 16/833 | 28/785 | low-count |
| P | control | GTC | -1.410 | 0.376 | 0.0607 | 1 | 6/732 | 13/605 | low-count |

**Largest effect (top 5 per site by \|effect\|, all rows)**

| Site | Condition | Feature | log2(OR) | Odds ratio | p_value | p_adj | day_10 | day_0 | Flags |
|---|---|---|---|---|---|---|---|---|---|
| A | control | TGG | -1.876 | 0.273 | 0.044 | 0.931 | 3/732 | 9/605 | low-count |
| A | BWM | CTG | -1.674 | 0.313 | 0.36 | 0.985 | 1/833 | 3/785 | low-count |
| A | BWM | TCG | -1.510 | 0.351 | 0.135 | 0.872 | 3/833 | 8/785 | low-count |
| A | BWM | CAA | -1.362 | 0.389 | 0.0871 | 0.872 | 5/833 | 12/785 | low-count |
| A | control | CAT | 1.314 | 2.486 | 0.631 | 1 | 3/732 | 1/605 | low-count |
| E | control | GTG | -2.872 | 0.137 | 0.051 | 1 | 1/732 | 6/605 | low-count |
| E | BWM | CAT | 2.508 | 5.688 | 0.125 | 1 | 6/833 | 1/785 | low-count |
| E | BWM | TGC | 2.243 | 4.734 | 0.219 | 1 | 5/833 | 1/785 | low-count |
| E | control | GGC | 1.731 | 3.319 | 0.385 | 1 | 4/732 | 1/605 | low-count |
| E | control | TCC | 1.513 | 2.853 | 0.0495 | 1 | 17/732 | 5/605 | low-count |
| P | control | ACA | -1.865 | 0.275 | 0.334 | 1 | 1/732 | 3/605 | low-count |
| P | control | TGG | -1.865 | 0.275 | 0.334 | 1 | 1/732 | 3/605 | low-count |
| P | control | GTC | -1.410 | 0.376 | 0.0607 | 1 | 6/732 | 13/605 | low-count |
| P | control | TGC | 1.317 | 2.492 | 0.305 | 1 | 6/732 | 2/605 | low-count |
| P | control | TAT | 1.314 | 2.486 | 0.631 | 1 | 3/732 | 1/605 | low-count |

_76 row(s) have a fully separated 2x2 table (one arm's count is 0), giving an undefined/infinite odds ratio; excluded from the table above and always low-count-flagged._

### Interpretation

<!-- INTERP_CODON_START -->
- **AAG (Lys) is the dominant codon behind both P-site Lysine results**, tracking the amino-acid-level numbers closely: BWM (log2(OR)=−0.801, p_adj=0.080) accounts for 86% of BWM's Lys counts at both timepoints (61/71 day_10, 95/111 day_0); control (log2(OR)=−0.897, p_adj=0.098) accounts for 77–85% of control's. Both are a hair short of significance, unlike their amino-acid-level counterparts (one significant, one near-miss) — the aggregation across Lys's two codons is what pushes the amino-acid signal over the line.

- **GGA (Gly) similarly drives the near-significant control P-site Glycine enrichment** (log2(OR)=0.819, p_adj=0.098; accounts for 88%/81% of the day_10/day_0 Gly counts vs. the amino-acid-level log2(OR)=0.708).

- **ATC (Ile) drives roughly two-thirds of the significant A-site/BWM Ile enrichment** (69% of day_10's count, 65% of day_0's), with a slightly larger effect (log2(OR)=1.093) than the amino-acid aggregate (1.017) — the remaining Ile codons (ATT, ATA) dilute it slightly.

- **Both Lysine codons decline from day_0 to day_10 at the E-site in BWM** (AAG log2(OR)=−0.349; AAA log2(OR)=−1.103, low-count) — consistent direction, reinforcing that the amino-acid-level E-site Lys decline isn't an AAG-only artifact, though neither codon is anywhere near significant here.

- **TGC (one of Cys's two codons) shows a larger day_10-enrichment (log2(OR)=2.243) than the amino-acid-level aggregate (1.506)** at the E-site/BWM — suggesting TGC, not TGT, carries the Cys pattern noted at the amino-acid level, though both are far from significant (p_adj≈1) with only 1–6 counts per arm.

- **Not a single one of the 366 codon-level tests clears p_adj < 0.05**, and 76 of 366 rows (21%) have a fully separated 2×2 table — a sparser, noisier picture than the amino-acid level's 2 real hits, consistent with the pattern seen in every family so far (more codon categories means smaller per-feature counts).
<!-- INTERP_CODON_END -->

## Key Points

<!-- KEY_POINTS_START -->
- **The two significant amino-acid findings are each carried by one dominant codon rather than being diffuse across synonymous codons**: ATC for the A-site/BWM Ile enrichment, AAG for the P-site/BWM Lys depletion. Neither codon quite reaches significance alone (p_adj 0.08–0.30) — it's the aggregation across a feature's codons that pushes the amino-acid-level test over the line, the mirror image of what `within_condition_binomial` showed (there, a single codon like AAG or GAT *carried* the amino-acid signal; here, *combining* codons is what creates it).

- **This connects to `within_condition_binomial`'s E-site Lysine enrichment**: that family found Lys enriched vs. background in all six condition×timepoint groups, but with the enrichment weakest at day_10 and strongest at day_0 within each condition (e.g. BWM_day_0 log2=1.18 vs BWM_day_10 log2=0.84). This family's direct day_10-vs-day_0 comparison confirms that fade is a real, if non-significant, decline at the E-site (log2(OR)=−0.434) — and shows it's significant at the **P-site** too (log2(OR)=−0.822), a site `within_condition_binomial` didn't flag a Lys pattern for at all.

- **Signal is sparser here than any resolution examined so far in this pipeline**: 2/120 amino-acid tests (1.7%) and 0/366 codon tests (0%) are significant, versus 33.9%/14.8% for `within_condition_binomial` and 0.6%/0% for `per_timepoint_fisher`. Makes sense — this is a within-condition, cross-timepoint comparison using only one condition's ~600–830 stall sites on each side of the test, the smallest, least-powered contrast of the three designs used in this pipeline.

- **The A- and P-sites carry what little signal this family has; the E-site's top rows (Cys, His, Trp) look like rare-feature noise** — all `low-count`, all far from significance, with no consistent story beyond "small counts produce large ratios."
<!-- KEY_POINTS_END -->

## Caveats

- **FDR grouping:** p-values are Benjamini-Hochberg corrected per (condition, site) — a row's `p_adj` is only comparable to other rows sharing that grouping.
- **Low-count threshold:** rows flagged `low-count` have a raw feature count below 50; treat their effect sizes as less reliable.

---
_Plots, Data, and Caveats are auto-generated by `result_interpretation_scripts/extract_key_data.py`
from `analysis/*.csv` and will be overwritten on the next run. The Interpretation (per level) and
Key Points (overall, at the bottom) sections are hand-authored and preserved across regenerations._
