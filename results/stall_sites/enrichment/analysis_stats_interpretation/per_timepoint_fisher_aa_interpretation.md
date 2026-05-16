---
input_csv: results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_aa.csv
family: per_timepoint_fisher
test_type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, aa); BH-FDR within each (timepoint, site) family of 20 AAs
test_type_source: user-confirmed
n_tests: 180
n_significant_fdr05: 27
n_significant_fdr10: 35
min_p_adj: 2.8687614397053295e-15
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "imbalanced-N", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "bh-per-(timepoint,site)", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "small-bh-family-discreteness", proposed_by: dylan, status: confirmed, why: "BH-FDR is computed within per-(timepoint, site) families of 20 AAs each. At family size 20 with discrete Fisher p-values the BH adjacency steps are coarse, and `p_adj` values cluster at a small number of distinct levels within a single cell (visible in this file: e.g. day_5,A has 5 rows tied at p_adj=0.0517 and below, day_10,P top of family at p_adj=0.148)."}
  - {label: "rare-aa-low-count", proposed_by: dylan, status: confirmed, why: "Rare amino acids (W, M, C) yield BWM_count cells in the single/double digits in some (timepoint, site) cells: e.g. day_0,E,W BWM_k=50 ctrl_k=263; day_5,A,C BWM_k=128 ctrl_k=132; day_10,A,W BWM_k=78 ctrl_k=80. OR estimates and their CIs are unstable for rows with k<100 in either arm; flagged `rare-aa` in Top hits."}
  - {label: "OR-direction-anchor", proposed_by: dylan, status: confirmed, why: "Each row is itself a BWM-vs-control 2x2: OR > 1 means BWM enriched relative to control at that (timepoint, site, aa); OR < 1 means BWM depleted. Direction interpretation is fixed by the contingency layout and is stated in Methods so downstream consumers do not invert."}
  - {label: "n-asymmetry-affects-power-not-direction", proposed_by: dylan, status: confirmed, why: "BWM_total varies across timepoints (day_0=6091, day_5=11935, day_10=6945); control_total varies more (day_0=27732, day_5=11177, day_10=8788). At day_0 the control:BWM ratio is 4.55:1 — the largest single-cell magnitude differences inflate to extreme p magnitudes there at moderate effect sizes (file-min p_adj 2.87e-15 occurs at day_0 E K with log2OR=+0.480, BWM_k=983, ctrl_k=3362). At day_10 the ratio is 1.27:1 and similar effect sizes do not reach FDR. Treat per-cell hit counts as power-modulated by N, not as evidence for or against direction-of-effect biology — that is a question for cross-test reconciliation."}
caveats_considered: []
headline: "Per-timepoint BWM-vs-control Fisher at AA resolution: 27/180 hits at FDR<0.05 (35/180 at FDR<0.10); per-(timepoint, site) cell n_sig at FDR<0.05 ranges from 0 (day_10 P) to 7 (day_0 A and day_0 E and day_0 P). 21 of 60 (site, aa) cells are concordant in OR direction across all 3 timepoints; 39 of 60 cells show at least one sign change across the 3 timepoints, of which only 1 reaches FDR<0.05 in two opposite-sign cells (E:K, day_0 BWM-enriched at log2OR=+0.480 / day_5 BWM-depleted at log2OR=-0.195). Largest-magnitude cells (by single-cell |log2OR| at p_adj<0.05) are: day_0 A:N -0.755 (rare-aa, BWM_k=134), day_0 P:R +0.519, day_0 E:R +0.482, day_0 E:K +0.480 (file-min p_adj=2.87e-15), day_0 E:Q -0.467, day_0 P:P -0.460, day_0 A:K +0.439, day_10 A:A -0.415, day_0 A:R +0.381. Largest-magnitude cells in the cross-timepoint concordant set (by mean log2OR across the 3 timepoints, single sign throughout) are P:K +0.253, P:R +0.239, A:R +0.218, A:T -0.178."
user_directives:
  - "(invocation context) `flat-prior` token — apply A.2.1 through A.2.9 strictly; do not import findings from prior CSV interpretations as priors; read this CSV cold without consulting `_INDEX.md` cross-family hooks; rank features by (a) effect size in high-count rows, (b) cross-synonym coherence (n/a at AA resolution; treated as cross-cell coherence), (c) reproducibility within this CSV's per-cell neighbours; report shared-direction features at equal billing with divergent features; for any p_adj < 1e-10 row name at least one alternative explanation."
  - "(per-CSV triage, carried over from original triage; not re-litigated this run) Test type confirmation → `Fisher's exact (two-sided), BWM vs control 2x2 per (timepoint, site, aa); BH-FDR within each (timepoint, site) family of 20 AAs`."
  - "(per-CSV triage, this run) CSV-specific caveats beyond family-wide → confirmed `small-bh-family-discreteness`, `rare-aa-low-count`, `OR-direction-anchor`, and added `n-asymmetry-affects-power-not-direction` as the magnitude-vs-power discipline analogous to the binomial files' p-magnitude-anchored caveat. The prior run's `cross-timepoint-direction-axis` caveat is dropped on this run because it phrased a temporal-trajectory axis that A.2.5 reserves for synthesis."
  - "(per-CSV triage, this run) Framing firmness → `Mixed` (re-derived under flat-prior from the prior run's `Firm`). Cells with the very smallest p_adj (single-row p_adj<1e-10) are firm by magnitude. Cells with p_adj ~1e-3 in BH families of 20 are near the discreteness floor of the small-bh-family-discreteness caveat and are exploratory until cross-test corroboration is in hand."
  - "(per-CSV triage, this run) Spotlight → `No spotlight`. Headline ranks features by data alone per A.2.3."
---

# Interpretation — per_timepoint_fisher_aa

> Source: `results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_aa.csv`
> Family: `per_timepoint_fisher` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, aa); BH-FDR within each (timepoint, site) family of 20 AAs (source: user-confirmed)

## User directives
- (invocation context) `flat-prior` — apply A.2.1 through A.2.9 strictly; read CSV cold; rank by effect size + per-cell reproducibility; equal billing for shared-direction vs divergent; alternative-explanation flags on every `p_adj < 1e-10` row.
- (per-CSV triage, carried over from original triage; not re-litigated) "Test type confirmation" → "Fisher's exact (two-sided), BWM vs control 2x2 per (timepoint, site, aa); BH-FDR within each (timepoint, site) family of 20 AAs."
- (per-CSV triage, this run) "CSV-specific caveats beyond family-wide" → confirmed `small-bh-family-discreteness`, `rare-aa-low-count`, `OR-direction-anchor`, `n-asymmetry-affects-power-not-direction`. The prior run's `cross-timepoint-direction-axis` caveat is dropped because it framed a temporal-trajectory axis that A.2.5 reserves for synthesis.
- (per-CSV triage, this run) "Framing firmness" → Mixed (re-derived under flat-prior from prior `Firm`). Single-row p_adj<1e-10 = firm by magnitude. p_adj ~1e-3 in BH-20 cells = exploratory pending cross-test corroboration.
- (per-CSV triage, this run) "Spotlight" → none. Data-ranked only per A.2.3.

## Headline
180 tests (3 timepoints x 3 sites x 20 AAs), 27 hits at p_adj<0.05 (15.0%), 35 at p_adj<0.10 (19.4%). Per-(timepoint, site) cell n_sig at p_adj<0.05: day_0 A=7, day_0 E=6, day_0 P=5; day_5 A=1, E=5, P=1; day_10 A=1, E=1, P=0. Of 60 (site, aa) cells, 21 are concordant in OR direction across all 3 timepoints (9 enriched in BWM, 12 depleted in BWM); 39 show at least one sign change across the 3 timepoint values. Only 1 of those 39 has the sign change at p_adj<0.05 on both opposite-sign rows: E:K (day_0 log2OR=+0.480, p_adj=2.87e-15; day_5 log2OR=-0.195, p_adj=4.93e-3; day_10 log2OR=-0.035, ns). Largest-magnitude single-cell signals at p_adj<0.05 (mixing directions): day_0 A:N -0.755 (rare-aa: BWM_k=134), day_0 P:R +0.519, day_0 E:R +0.482, day_0 E:K +0.480 (file-min p_adj), day_0 E:Q -0.467, day_0 P:P -0.460, day_0 A:K +0.439, day_10 A:A -0.415, day_0 A:R +0.381. Largest-magnitude cells in the cross-timepoint concordant set (mean log2OR across the 3 timepoints, single sign throughout): P:K +0.253 (BWM-enriched, 2/3 cell-sig), P:R +0.239, A:R +0.218, A:T -0.178. Treat per-cell n_sig differences across timepoints as power-modulated by the BWM/control N imbalance (control:BWM ratio 4.55:1 at day_0, 0.94:1 at day_5, 1.27:1 at day_10), not as direction-of-effect biology.

## Top hits

### Cross-timepoint direction concordance (primary table)

21 of 60 (site, aa) cells have the same OR direction in all 3 timepoints. Sorted by |mean log2OR| across the 3 timepoint values. `#sig` is the count (out of 3) of timepoints where this row reaches p_adj<0.05; `min_p_adj` is the smallest of the 3 per-timepoint values. Flag column: `iid-amp` for any cell whose smallest per-timepoint `p_adj` falls below 1e-10; `rare-aa` if any of the 3 timepoint rows has BWM_k<100; `single-tp-driven` if `#sig=1/3` (i.e. one timepoint carries all the cell's significance); `multi-tp-coherent` if `#sig>=2/3`.

| direction | cell (site:aa) | mean log2OR | per-tp log2OR (d0, d5, d10) | #sig (FDR<0.05) | min p_adj | flag |
| --- | --- | --- | --- | --- | --- | --- |
| enriched | P:K | +0.253 | +0.36, +0.21, +0.19 | 2/3 | 1.05e-06 | multi-tp-coherent |
| enriched | P:R | +0.239 | +0.52, +0.09, +0.10 | 1/3 | 4.65e-09 | single-tp-driven |
| enriched | A:R | +0.218 | +0.38, +0.13, +0.15 | 1/3 | 2.34e-05 | single-tp-driven |
| depleted | A:T | -0.178 | -0.34, -0.05, -0.14 | 1/3 | 5.98e-03 | single-tp-driven |
| enriched | A:W | +0.172 | +0.09, +0.12, +0.31 | 0/3 | 3.96e-01 | rare-aa |
| depleted | P:T | -0.166 | -0.22, -0.04, -0.23 | 0/3 | 7.07e-02 |  |
| enriched | A:H | +0.153 | +0.23, +0.11, +0.12 | 0/3 | 1.90e-01 |  |
| depleted | A:C | -0.146 | -0.08, -0.14, -0.22 | 0/3 | 4.95e-01 | rare-aa |
| depleted | P:A | -0.137 | -0.09, -0.13, -0.19 | 0/3 | 1.64e-01 |  |
| depleted | P:S | -0.136 | -0.28, -0.11, -0.02 | 1/3 | 2.96e-02 | single-tp-driven |
| depleted | P:C | -0.118 | -0.16, -0.12, -0.07 | 0/3 | 6.49e-01 | rare-aa |
| enriched | P:L | +0.117 | +0.04, +0.11, +0.20 | 0/3 | 1.48e-01 |  |
| depleted | P:E | -0.108 | -0.19, -0.12, -0.01 | 0/3 | 5.59e-02 |  |
| depleted | P:D | -0.091 | -0.10, -0.02, -0.15 | 0/3 | 1.64e-01 |  |
| depleted | E:D | -0.084 | -0.12, -0.05, -0.08 | 0/3 | 3.47e-01 |  |
| depleted | E:E | -0.080 | -0.05, -0.17, -0.02 | 1/3 | 3.32e-02 | single-tp-driven |
| enriched | E:L | +0.065 | +0.02, +0.15, +0.02 | 0/3 | 1.19e-01 |  |
| depleted | P:N | -0.049 | -0.09, -0.01, -0.05 | 0/3 | 5.72e-01 |  |
| depleted | P:F | -0.046 | -0.01, -0.06, -0.07 | 0/3 | 7.34e-01 |  |
| depleted | E:H | -0.045 | -0.04, -0.05, -0.04 | 0/3 | 7.43e-01 |  |
| depleted | E:N | -0.043 | -0.04, -0.05, -0.04 | 0/3 | 5.96e-01 |  |

Counts in the concordant set: 9 enriched, 12 depleted. Of the 21, only 4 have any per-timepoint cell at p_adj<0.05; the remaining 17 are direction-consistent at sub-significance and are reported here for symmetric coverage with the direction-flip table below.

### Direction-flip cells across timepoints (equal-billing companion table)

39 of 60 (site, aa) cells show at least one sign change across the 3 timepoint values. Of these, only 1 has the flip register at p_adj<0.05 on both opposite-sign rows; the rest have at least one sub-significant row, so the "flip" magnitude is uncertain. Sorted by max single-cell |log2OR|.

| cell (site:aa) | log2OR range | per-tp log2OR (sig at p_adj<0.05 marked `*`) | sig-flip | flag |
| --- | --- | --- | --- | --- |
| A:N | [-0.755, +0.302] | -0.75*, -0.25, +0.30 | no | rare-aa (d10 BWM_k=267) |
| E:W | [-0.211, +0.621] | -0.21, -0.16, +0.62 | no | rare-aa |
| E:C | [-0.310, +0.520] | -0.31, -0.21, +0.52 | no | rare-aa |
| E:R | [-0.095, +0.482] | +0.48*, -0.10, +0.39* | no |  |
| E:K | [-0.195, +0.480] | +0.48*, -0.19*, -0.04 | yes (d0 vs d5 both p_adj<0.05) | iid-amp (d0 cell only) |
| E:Q | [-0.467, +0.121] | -0.47*, -0.06, +0.12 | no |  |
| P:P | [-0.460, +0.025] | -0.46*, +0.03, -0.22 | no |  |
| A:K | [-0.062, +0.439] | +0.44*, -0.06, +0.13 | no |  |
| A:A | [-0.415, +0.031] | -0.08, +0.03, -0.42* | no |  |
| P:Q | [-0.255, +0.378] | -0.26, +0.20, +0.38 | no |  |
| E:S | [-0.375, +0.009] | -0.37*, +0.01, -0.02 | no |  |
| E:Y | [-0.371, +0.243] | -0.37*, +0.24, +0.21 | no |  |
| E:P | [-0.133, +0.358] | -0.15, +0.36*, -0.13 | no |  |
| A:Q | [-0.201, +0.336] | -0.20, +0.11, +0.34 | no |  |
| A:F | [-0.321, +0.116] | -0.32*, +0.12, -0.01 | no |  |

(15 highest-magnitude flip cells shown; 24 more lower-magnitude flip cells are in the data but all have max |log2OR| < 0.30.)

### Per-(timepoint, site) tables

Each block is one BH-FDR family of 20 AAs at one (timepoint, site) cell. Top hits at p_adj<0.10 are listed (top-5 enriched + top-5 depleted by |log2OR|; fewer if fewer rows clear p_adj<0.10).

<details>
<summary>day_0 (A / E / P)</summary>

#### day_0, A site (n_sig p_adj<0.05: 7/20)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | K | +0.439 (OR=1.356) | 2.31e-09 |  |
| enriched | R | +0.381 (OR=1.302) | 2.34e-05 |  |
| enriched | G | +0.283 (OR=1.217) | 6.72e-04 |  |
| depleted | N | -0.755 (OR=0.593) | 3.17e-08 |  |
| depleted | T | -0.339 (OR=0.791) | 5.98e-03 |  |
| depleted | F | -0.321 (OR=0.800) | 1.98e-02 |  |
| depleted | D | -0.203 (OR=0.869) | 4.21e-02 |  |

#### day_0, E site (n_sig p_adj<0.05: 6/20)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | R | +0.482 (OR=1.396) | 1.09e-07 |  |
| enriched | K | +0.480 (OR=1.395) | 2.87e-15 | iid-amp |
| depleted | Q | -0.467 (OR=0.723) | 1.66e-04 |  |
| depleted | S | -0.375 (OR=0.771) | 2.26e-03 |  |
| depleted | Y | -0.371 (OR=0.773) | 3.00e-02 |  |
| depleted | T | -0.248 (OR=0.842) | 4.78e-02 |  |
| depleted | F | -0.222 (OR=0.857) | 1.88e-01 | nominal-only |

#### day_0, P site (n_sig p_adj<0.05: 5/20)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | R | +0.519 (OR=1.433) | 4.65e-09 |  |
| enriched | K | +0.360 (OR=1.283) | 1.05e-06 |  |
| enriched | G | +0.195 (OR=1.145) | 2.96e-02 |  |
| depleted | P | -0.460 (OR=0.727) | 1.30e-04 |  |
| depleted | S | -0.281 (OR=0.823) | 2.96e-02 |  |
| depleted | T | -0.224 (OR=0.856) | 7.07e-02 |  |
| depleted | E | -0.192 (OR=0.875) | 5.59e-02 |  |

</details>

<details>
<summary>day_5 (A / E / P)</summary>

#### day_5, A site (n_sig p_adj<0.05: 1/20)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | L | +0.280 (OR=1.215) | 4.98e-03 |  |
| enriched | P | +0.236 (OR=1.177) | 8.62e-02 |  |
| depleted | N | -0.255 (OR=0.838) | 5.17e-02 |  |
| depleted | E | -0.180 (OR=0.883) | 5.17e-02 |  |
| depleted | I | -0.177 (OR=0.884) | 8.62e-02 |  |

#### day_5, E site (n_sig p_adj<0.05: 5/20)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | P | +0.358 (OR=1.282) | 4.93e-03 |  |
| enriched | A | +0.299 (OR=1.230) | 9.35e-03 |  |
| enriched | T | +0.219 (OR=1.164) | 6.05e-02 |  |
| depleted | G | -0.202 (OR=0.869) | 3.32e-02 |  |
| depleted | K | -0.195 (OR=0.874) | 4.93e-03 |  |
| depleted | E | -0.174 (OR=0.886) | 3.32e-02 |  |

#### day_5, P site (n_sig p_adj<0.05: 1/20)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | K | +0.211 (OR=1.157) | 4.43e-02 |  |
| depleted | V | -0.196 (OR=0.873) | 8.38e-02 |  |

</details>

<details>
<summary>day_10 (A / E / P)</summary>

#### day_10, A site (n_sig p_adj<0.05: 1/20)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| depleted | A | -0.415 (OR=0.750) | 2.91e-03 |  |

#### day_10, E site (n_sig p_adj<0.05: 1/20)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | R | +0.388 (OR=1.308) | 1.86e-03 |  |

#### day_10, P site (n_sig p_adj<0.05: 0/20)
*(no rows clear p_adj<0.10; closest are P:I, P:K, P:L all at p_adj=0.148, log2OR ~+0.16 to +0.21.)*

</details>

## Numbers at a glance
- `n_tests`: 180 (3 timepoints x 3 sites x 20 AAs)
- `n_significant` (adjusted-p < 0.05): 27 (15.0%)
- `n_significant` (adjusted-p < 0.10): 35 (19.4%)
- `min adjusted-p`: 2.8687614397053295e-15 (day_0 E K, log2OR=+0.480, BWM_k=983/n=6091, ctrl_k=3362/n=27732)
- `p_floor`: n/a — Fisher's exact has no analytic floor; effective discreteness floor at small (k_bwm + k_ctrl) is captured by `small-bh-family-discreteness`
- Cells with p_adj < 1e-10: 1 of 180 (only day_0 E K); flagged `iid-amp`
- Cross-(timepoint, site) BH families and their n_sig at p_adj<0.05:

| timepoint | A | E | P | BWM_total | control_total | ctrl:BWM ratio |
| --- | --- | --- | --- | --- | --- | --- |
| day_0 | 7/20 (min p 2.31e-09) | 6/20 (min p 2.87e-15) | 5/20 (min p 4.65e-09) | 6091 | 27732 | 4.55 |
| day_5 | 1/20 (min p 4.98e-03) | 5/20 (min p 4.93e-03) | 1/20 (min p 4.43e-02) | 11935 | 11177 | 0.94 |
| day_10 | 1/20 (min p 2.91e-03) | 1/20 (min p 1.86e-03) | 0/20 (min p 1.48e-01) | 6945 | 8788 | 1.27 |

- Cross-timepoint (site, aa) cells, all 3 timepoints same OR direction: 21/60 (9 BWM-enriched, 12 BWM-depleted)
- Cross-timepoint (site, aa) cells with at least one sign change: 39/60
- Cells with sign change at p_adj<0.05 on both opposite-sign rows: 1 (E:K)

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 contingency BWM_count vs control_count for each (timepoint, site, aa), with BH-FDR correction within each (timepoint, site) family of 20 AAs; user confirmed in the original triage and the answer is carried over to this re-run without re-litigation. Effect column is `odds_ratio` (re-expressed as log2OR in this report's tables for symmetry with sister files); test statistic and exact p are returned by `scipy.stats.fisher_exact`. The p-correction column is `p_adj` (BH per (timepoint, site) family). The test answers "is the BWM-vs-control AA composition at this site different at this timepoint?". It does *not* answer "is each condition's per-codon enrichment vs background informative" (that is the `within_condition_binomial` family) and *not* "is the BWM-vs-control difference itself shifting across timepoints" (that requires a 3-way model or an explicit between-timepoint contrast — see the `between_timepoint_wilcoxon` family for the within-condition variant of that question).

## Caveats
### Confirmed
- **pseudorep** (family-wide) — 2x2 contingencies pool replicates by design; per-replicate variation is not represented in the test statistic. Inherited from family `per_timepoint_fisher`.
- **large-N-Fisher-anticonservative** (family-wide) — Fisher's exact on contingencies built from many thousands of pooled stall events behaves anti-conservatively when stall positions are correlated (cluster on motifs, share transcripts). Effective N is smaller than k_total. Inherited from family `per_timepoint_fisher`.
- **imbalanced-N** (family-wide) — control_total is much larger than BWM_total at day_0 (27732 vs 6091, ratio 4.55:1). The asymmetry concentrates Fisher's information in the larger arm and produces extreme p_adj at moderate effect sizes there. Inherited from family `per_timepoint_fisher`.
- **bh-per-(timepoint,site)** (family-wide) — BH correction is applied independently within each of the 9 (timepoint, site) families of 20 AAs, not across the full 180-test grid. Two cells with the same raw p in different (timepoint, site) families can therefore receive different `p_adj` values; cross-(timepoint, site) p_adj rankings are not directly commensurable. Inherited from family `per_timepoint_fisher`.
- **small-bh-family-discreteness** (per-CSV) — at family size 20 the BH adjacency steps are coarse, and `p_adj` values cluster at a small number of distinct levels within a single cell. Visible in this file: day_5,A has 5 rows at p_adj=0.0517 and below, day_10,P top-of-family at p_adj=0.148. Use the `p_adj < 0.05` threshold as a binary flag, not a precise rank.
- **rare-aa-low-count** (per-CSV) — rare AAs (W, M, C and several others depending on (timepoint, site)) have BWM_k or ctrl_k under 100 in some cells (e.g. day_0,E,W BWM_k=50 ctrl_k=263; day_5,A,C BWM_k=128 ctrl_k=132; day_10,A,W BWM_k=78 ctrl_k=80; day_0,A,C BWM_k=75 ctrl_k=360). OR estimates and their CIs are unstable at small k; rows are flagged `rare-aa` in Top hits and the cross-timepoint concordance table. The day_0,A,N depleted cell (log2OR=-0.755, the file's largest single-cell magnitude at p_adj<0.05) sits at BWM_k=134 — the magnitude is real but interpretive weight should account for the count.
- **OR-direction-anchor** (per-CSV) — OR > 1 = BWM enriched relative to control at that (timepoint, site, aa); OR < 1 = BWM depleted. Direction is fixed by the contingency layout. Stated here so downstream consumers do not invert.
- **n-asymmetry-affects-power-not-direction** (per-CSV) — BWM_total varies across timepoints (6091 / 11935 / 6945) and control_total varies more (27732 / 11177 / 8788). The day_0 control:BWM ratio of 4.55:1 means a moderate effect size at day_0 inflates to a smaller p_adj than the same effect size would at day_5 or day_10. The 12-OOM gap between min p_adj at day_0 (2.87e-15) and day_5 (4.93e-3) or day_10 (1.86e-3) is consistent with this N-modulated power asymmetry. Per-cell n_sig differences are therefore power-modulated, not necessarily direction-of-effect biology; cross-test reconciliation is the place to disentangle the two.

### Considered but not applicable
*(none denied this run; no per-CSV proposals were rejected.)*

## For Chumeng (joint-reading hooks)
- Family: `per_timepoint_fisher` — sister CSV in this family that should be reconciled: `per_timepoint_fisher_codon.csv` (codon resolution; same design, same family-wide caveats).
- Open questions Chumeng should resolve at synthesis time:
  - The 21 cross-timepoint concordant cells (top by mean |log2OR|: P:K +0.253, P:R +0.239, A:R +0.218, A:T -0.178, A:W +0.172, P:T -0.166, A:H +0.153, A:C -0.146, P:A -0.137, P:S -0.136 — 9 enriched, 12 depleted overall). Do these reappear with the same direction in the codon sister file aggregated to AA, and in the within-condition Fisher and within-condition binomial AA files where applicable? If a cell is concordant here but reverses sign in a sister test, that flags either a design-axis difference (BWM-vs-control vs within-condition vs binomial-vs-bg) or an aggregation artifact.
  - The single sig-flip cell (E:K: day_0 BWM-enriched +0.480 at p_adj=2.87e-15, day_5 BWM-depleted -0.195 at p_adj=4.93e-3, day_10 BWM-depleted -0.035 ns). Does the codon sister file show the same flip pattern at any one of E:K's two codons (AAA, AAG), or does the AA-level flip dissolve when split by codon? Independently: does the within-condition Fisher show consistent E:K direction within each condition's BWM cells across d0/d5/d10? If the AA-level flip is real and not aggregation-driven, both routes should support it.
  - The day_0 A:N cell (log2OR=-0.755, p_adj=3.17e-08, BWM_k=134 ctrl_k=1014). It is the largest single-cell magnitude in the file but has BWM_k=134 (not rare by the k<100 cutoff but at the lower-end of the high-count range); does this magnitude reappear in the codon sister file at any of N's two codons (AAT, AAC), and does the within-condition AA Wilcoxon flag it as a closest-to-significant feature on either condition? If it shows up in only one of those, the magnitude may be an A-site rare-N edge effect at the high-N day_0 control arm rather than a stable feature.
  - The N-imbalance asymmetry: per-(timepoint, site) n_sig is 18 at day_0, 7 at day_5, 2 at day_10, while min p_adj follows the same gradient over 12 orders of magnitude. Does the codon sister file show the same gradient (consistent with N-driven power) or a different one (consistent with biology being timepoint-specific)? If both files show the same gradient with the same N-imbalance, the gradient is likely power-modulated; if the codon file shows a different gradient, that flags a resolution-dependent biological signal Chumeng should investigate.
  - The 17 sub-significant cross-timepoint concordant cells in the primary table (#sig=0/3 or 0/3 with consistent direction). These are direction-coherent but not individually significant under any (timepoint, site) BH family. Does any of them reach significance at codon resolution (where some synonyms may carry the signal that AA aggregation dilutes), or under the within-condition binomial (where the direction-vs-background test has different power profile)? Useful to triangulate before discarding them as noise.
