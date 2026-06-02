---
input_csv: results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_aa.csv
family: per_timepoint_fisher
test_type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, aa); BH-FDR within each (timepoint, site) family of 20 AAs
test_type_source: user-confirmed
synced_from_olive_qmd: 2026-06-01
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
  - {label: "small-bh-family-discreteness", proposed_by: dylan, status: confirmed, why: "BH-FDR is computed within per-(timepoint, site) families of 20 AAs each. At family size 20 with discrete Fisher p-values the BH adjacency steps are coarse, and `p_adj` values cluster at a small number of distinct levels within a single cell (visible in this file: e.g. day_5,A has 5 rows at p_adj=0.0517 and below, day_10,P top of family at p_adj=0.148)."}
  - {label: "rare-aa-low-count", proposed_by: dylan, status: confirmed, why: "Rare amino acids (W, M, C) yield BWM_count cells in the single/double digits in some (timepoint, site) cells: e.g. day_0,E,W BWM_k=50 ctrl_k=263; day_5,A,C BWM_k=128 ctrl_k=132; day_10,A,W BWM_k=78 ctrl_k=80. OR estimates and their CIs are unstable for rows with k<100 in either arm; flagged `rare-aa` in Top hits."}
  - {label: "OR-direction-anchor", proposed_by: dylan, status: confirmed, why: "Each row is itself a BWM-vs-control 2x2: OR > 1 means BWM enriched relative to control at that (timepoint, site, aa); OR < 1 means BWM depleted. Direction interpretation is fixed by the contingency layout and is stated in Methods so downstream consumers do not invert."}
  - {label: "n-asymmetry-affects-power-not-direction", proposed_by: dylan, status: confirmed, why: "BWM_total varies across timepoints (day_0=6091, day_5=11935, day_10=6945); control_total varies more (day_0=27732, day_5=11177, day_10=8788). At day_0 the control:BWM ratio is 4.55:1 — the largest single-cell magnitude differences inflate to extreme p magnitudes there at moderate effect sizes (file-min p_adj 2.87e-15 occurs at day_0 E K with log2OR=+0.480, BWM_k=983, ctrl_k=3362). At day_10 the ratio is 1.27:1 and similar effect sizes do not reach FDR. Treat per-cell hit counts as power-modulated by N, not as evidence for or against direction-of-effect biology — that is a question for cross-test reconciliation."}
caveats_considered: []
headline: "Per-timepoint BWM-vs-control Fisher at AA resolution: 27/180 hits at FDR<0.05 (35/180 at FDR<0.10); per-(timepoint, site) cell n_sig at FDR<0.05 ranges from 0 (day_10 P) to 7 (day_0 A). 21 of 60 (site, aa) cells are concordant in OR direction across all 3 timepoints; 39 of 60 cells show at least one sign change across the 3 timepoints, of which only 1 reaches FDR<0.05 in two opposite-sign cells (E:K, day_0 BWM-enriched at log2OR=+0.480 / day_5 BWM-depleted at log2OR=-0.195). Largest-magnitude cells (by single-cell |log2OR| at p_adj<0.05) are: day_0 A:N -0.755 (BWM_k=134; magnitude real, not count-floor), day_0 P:R +0.519, day_0 E:R +0.482, day_0 E:K +0.480 (file-min p_adj=2.87e-15), day_0 E:Q -0.467, day_0 P:P -0.460, day_0 A:K +0.439, day_0 A:R +0.381, day_10 A:A -0.415. Largest-magnitude cells in the cross-timepoint concordant set (by mean log2OR across the 3 timepoints, single sign throughout) are P:K +0.253, P:R +0.239, A:R +0.218, A:T -0.178."
user_directives:
  - "(invocation context) `flat-prior` token — apply A.2.1 through A.2.9 strictly; do not import findings from prior CSV interpretations as priors; read this CSV cold without consulting `_INDEX.md` cross-family hooks; rank features by (a) effect size in high-count rows, (b) cross-synonym coherence (n/a at AA resolution; treated as cross-cell coherence), (c) reproducibility within this CSV's per-cell neighbours; report shared-direction features at equal billing with divergent features; for any p_adj < 1e-10 row name at least one alternative explanation."
  - "(per-CSV triage, carried over from original triage; not re-litigated this run) Test type confirmation → `Fisher's exact (two-sided), BWM vs control 2x2 per (timepoint, site, aa); BH-FDR within each (timepoint, site) family of 20 AAs`."
  - "(per-CSV triage, this run) CSV-specific caveats beyond family-wide → confirmed `small-bh-family-discreteness`, `rare-aa-low-count`, `OR-direction-anchor`, and added `n-asymmetry-affects-power-not-direction` as the magnitude-vs-power discipline analogous to the binomial files' p-magnitude-anchored caveat. The prior run's `cross-timepoint-direction-axis` caveat is dropped on this run because it phrased a temporal-trajectory axis that A.2.5 reserves for synthesis."
  - "(per-CSV triage, this run) Framing firmness → `Mixed` (re-derived under flat-prior from the prior run's `Firm`). Cells with the very smallest p_adj (single-row p_adj<1e-10) are firm by magnitude. Cells with p_adj ~1e-3 in BH families of 20 are near the discreteness floor of the small-bh-family-discreteness caveat and are exploratory until cross-test corroboration is in hand."
  - "(per-CSV triage, this run) Spotlight → `No spotlight`. Headline ranks features by data alone per A.2.3."
  - "(readback) Reconciled shared content from the corrected .qmd on 2026-06-01 → adopted Olive's section order under Option A: the six per-(timepoint, direction) merged-site Top-hits tables (with the raw `p_value` column) now lead, and the cross-timepoint direction-concordance + direction-flip tables move to a `Cross-timepoint summary` at the end; the old per-(timepoint, site) `<details>` blocks are dropped and a Flag glossary added. Bare AA codes kept (no `K (lysine)` expansion). Reconciled Methods (per-codon → per-aa typo). Corrected the concordant-set significance-count sentence (4 → 6 cells with ≥1 sig timepoint; 17 → 15 sub-significant)."
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
- (readback) Section order reconciled to the corrected `.qmd`: per-(timepoint, direction) merged-site tables lead, cross-timepoint concordance/flip moved to the end; raw `p_value` column adopted; Methods per-codon→per-aa typo fixed; concordant-set sig-count sentence corrected (6 with ≥1 sig timepoint / 15 sub-significant).

## Headline
180 tests (3 timepoints x 3 sites x 20 AAs), 27 hits at p_adj<0.05 (15.0%), 35 at p_adj<0.10 (19.4%). Per-(timepoint, site) cell n_sig at p_adj<0.05: day_0 A=7, P=5, E=6; day_5 A=1, P=1, E=5; day_10 A=1, P=0, E=1. Of 60 (site, aa) cells, 21 are concordant in OR direction across all 3 timepoints (7 enriched in BWM, 14 depleted in BWM); 39 show at least one sign change across the 3 timepoint values. Only 1 of those 39 has the sign change at p_adj<0.05 on both opposite-sign rows: E:K (day_0 log2OR=+0.480, p_adj=2.87e-15; day_5 log2OR=-0.195, p_adj=4.93e-3; day_10 log2OR=-0.035, ns). Largest-magnitude single-cell signals at p_adj<0.05 (mixing directions): day_0 A:N -0.755 (BWM_k=134; magnitude real, not count-floor), day_0 P:R +0.519, day_0 E:R +0.482, day_0 E:K +0.480 (file-min p_adj), day_0 E:Q -0.467, day_0 P:P -0.460, day_0 A:K +0.439, day_0 A:R +0.381, day_10 A:A -0.415. Largest-magnitude cells in the cross-timepoint concordant set (mean log2OR across the 3 timepoints, single sign throughout): P:K +0.253 (BWM-enriched, 2/3 cell-sig), P:R +0.239, A:R +0.218, A:T -0.178. Treat per-cell n_sig differences across timepoints as power-modulated by the BWM/control N imbalance (control:BWM ratio 4.55:1 at day_0, 0.94:1 at day_5, 1.27:1 at day_10), not as direction-of-effect biology.

## Top hits

The effect column is `log2_OR` (the log2 of the `odds_ratio` column). Direction is fixed by the BWM-vs-control contingency layout: positive log2(OR) = BWM-enriched, negative = BWM-depleted. `p_value` is the raw two-sided Fisher's exact p; `p_adj` is BH-corrected within each (timepoint, site) family of 20 AAs. Tables below merge all three ribosome sites and are split per-direction within each timepoint.

**Inclusion criterion.** An amino acid appears in a table only if its `p_adj` is below 0.05 within its own (timepoint, site) family of 20 — i.e. it clears BH-FDR correction inside that family. Every row clearing `p_adj` < 0.05 is shown (no row cap). Each table merges all three ribosome sites for a (timepoint, direction) pair; rows are grouped by `site` in A -> P -> E order and within each site sorted by `|log2_OR|` descending. If no rows clear the threshold for a given (timepoint, direction) combination, that table is omitted entirely.

Two cross-timepoint summary tables (direction concordance and direction-flip cells) appear under "Cross-timepoint summary" at the end of this section, after the per-(timepoint, direction) blocks.

### day_0 — Enriched (BWM > control)

| site | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| A | K | +0.439 | 1.16e-10 | 2.31e-09 |  |
| A | R | +0.381 | 3.50e-06 | 2.34e-05 |  |
| A | G | +0.283 | 1.35e-04 | 6.72e-04 |  |
| P | R | +0.519 | 2.32e-10 | 4.65e-09 |  |
| P | K | +0.360 | 1.05e-07 | 1.05e-06 |  |
| P | G | +0.195 | 7.10e-03 | 2.96e-02 |  |
| E | R | +0.482 | 1.09e-08 | 1.09e-07 |  |
| E | K | +0.480 | 1.43e-16 | 2.87e-15 | iid-amp |

### day_0 — Depleted (BWM < control)

| site | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| A | N | -0.755 | 3.17e-09 | 3.17e-08 |  |
| A | T | -0.339 | 1.50e-03 | 5.98e-03 |  |
| A | F | -0.321 | 5.90e-03 | 1.98e-02 |  |
| A | D | -0.203 | 1.47e-02 | 4.21e-02 |  |
| P | P | -0.460 | 1.95e-05 | 1.30e-04 |  |
| P | S | -0.281 | 7.40e-03 | 2.96e-02 |  |
| E | Q | -0.467 | 2.49e-05 | 1.66e-04 |  |
| E | S | -0.375 | 4.53e-04 | 2.26e-03 |  |
| E | Y | -0.371 | 7.50e-03 | 3.00e-02 |  |
| E | T | -0.248 | 1.43e-02 | 4.78e-02 |  |

### day_5 — Enriched (BWM > control)

| site | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| A | L | +0.280 | 2.49e-04 | 4.98e-03 |  |
| P | K | +0.211 | 2.20e-03 | 4.43e-02 |  |
| E | P | +0.358 | 4.89e-04 | 4.93e-03 |  |
| E | A | +0.299 | 1.40e-03 | 9.35e-03 |  |

### day_5 — Depleted (BWM < control)

| site | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| E | G | -0.202 | 8.30e-03 | 3.32e-02 |  |
| E | K | -0.195 | 4.93e-04 | 4.93e-03 |  |
| E | E | -0.174 | 7.20e-03 | 3.32e-02 |  |

### day_10 — Enriched (BWM > control)

| site | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| E | R | +0.388 | 9.29e-05 | 1.86e-03 |  |

### day_10 — Depleted (BWM < control)

| site | amino acid | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| A | A | -0.415 | 1.45e-04 | 2.91e-03 |  |

### Cross-timepoint summary

#### Cross-timepoint direction concordance

21 of 60 (site, aa) cells hold the same OR direction across all 3 timepoints. Below the cells are split by displayed mean-log2OR direction into BWM-enriched and BWM-depleted sub-tables; each is sorted by `#sig` descending, then by `|mean log2OR|` descending. `#sig` is the count (out of 3) of timepoints where the row reaches `p_adj` < 0.05 (the FDR-corrected threshold); `min p_adj` is the smallest of the 3 per-timepoint values. The `per-tp log2OR` column lists the three per-timepoint values in chronological order (day 0, day 5, day 10); `*` marks each per-timepoint value at `p_adj` < 0.05.

##### Concordance — BWM-enriched (mean log2OR > 0)

| Amino Acid | mean log2OR | per-tp log2OR | #sig | min p_adj | flag |
| --- | --- | --- | --- | --- | --- |
| P:K | +0.253 | +0.36\* +0.21\* +0.19 | 2/3 | 1.05e-06 |  |
| P:R | +0.239 | +0.52\* +0.09 +0.10 | 1/3 | 4.65e-09 |  |
| A:R | +0.218 | +0.38\* +0.13 +0.15 | 1/3 | 2.34e-05 |  |
| A:W | +0.172 | +0.09 +0.12 +0.31 | 0/3 | 3.96e-01 | rare-aa (0, 5, 10) |
| A:H | +0.153 | +0.23 +0.11 +0.12 | 0/3 | 1.90e-01 |  |
| P:L | +0.117 | +0.04 +0.11 +0.20 | 0/3 | 1.48e-01 |  |
| E:L | +0.065 | +0.02 +0.15 +0.02 | 0/3 | 1.19e-01 |  |

##### Concordance — BWM-depleted (mean log2OR < 0)

| Amino Acid | mean log2OR | per-tp log2OR | #sig | min p_adj | flag |
| --- | --- | --- | --- | --- | --- |
| A:T | -0.178 | -0.34\* -0.05 -0.14 | 1/3 | 5.98e-03 |  |
| P:S | -0.136 | -0.28\* -0.11 -0.02 | 1/3 | 2.96e-02 |  |
| E:E | -0.080 | -0.05 -0.17\* -0.02 | 1/3 | 3.32e-02 |  |
| P:T | -0.166 | -0.22 -0.04 -0.23 | 0/3 | 7.07e-02 |  |
| A:C | -0.146 | -0.08 -0.14 -0.22 | 0/3 | 4.95e-01 | rare-aa (0, 10) |
| P:A | -0.137 | -0.09 -0.13 -0.19 | 0/3 | 1.64e-01 |  |
| P:C | -0.118 | -0.16 -0.12 -0.07 | 0/3 | 6.49e-01 | rare-aa (0, 10) |
| P:E | -0.108 | -0.19 -0.12 -0.01 | 0/3 | 5.59e-02 |  |
| P:D | -0.091 | -0.10 -0.02 -0.15 | 0/3 | 1.64e-01 |  |
| E:D | -0.084 | -0.12 -0.05 -0.08 | 0/3 | 3.47e-01 |  |
| P:N | -0.049 | -0.09 -0.01 -0.05 | 0/3 | 5.72e-01 |  |
| P:F | -0.046 | -0.01 -0.06 -0.07 | 0/3 | 7.34e-01 |  |
| E:H | -0.045 | -0.04 -0.05 -0.04 | 0/3 | 7.43e-01 |  |
| E:N | -0.043 | -0.04 -0.05 -0.04 | 0/3 | 5.96e-01 |  |

Counts in the concordant set: 7 enriched, 14 depleted. Of the 21, only 6 have any per-timepoint cell at p_adj<0.05 (P:K, P:R, A:R enriched; A:T, P:S, E:E depleted); the remaining 15 are direction-consistent at sub-significance and are reported here for symmetric coverage with the direction-flip table below.

#### Direction-flip cells across timepoints

39 of 60 (site, aa) cells show at least one sign change across the 3 timepoint values. Of these, only 1 (E:K) has the flip register at p_adj<0.05 on both opposite-sign rows — flagged `both-flip-sig`. `#sig` is the count (out of 3 timepoints) of rows reaching p_adj<0.05; the `per-tp log2OR` column lists the three per-timepoint values in chronological order (day 0, day 5, day 10); `*` marks each per-timepoint value that reaches the threshold. Sorted by `#sig` desc, then by max single-cell |log2OR| desc.

| Amino Acid | log2OR range | per-tp log2OR | #sig | flag |
| --- | --- | --- | --- | --- |
| E:R | [-0.096, +0.482] | +0.48\* -0.10 +0.39\* | 2/3 |  |
| E:K | [-0.195, +0.480] | +0.48\* -0.19\* -0.04 | 2/3 | both-flip-sig; iid-amp (0) |
| A:N | [-0.755, +0.297] | -0.75\* -0.25 +0.30 | 1/3 |  |
| E:Q | [-0.467, +0.120] | -0.47\* -0.06 +0.12 | 1/3 |  |
| P:P | [-0.460, +0.025] | -0.46\* +0.03 -0.22 | 1/3 |  |
| A:K | [-0.062, +0.439] | +0.44\* -0.06 +0.13 | 1/3 |  |
| A:A | [-0.415, +0.031] | -0.08 +0.03 -0.42\* | 1/3 |  |
| E:S | [-0.375, +0.009] | -0.37\* +0.01 -0.02 | 1/3 |  |
| E:Y | [-0.371, +0.242] | -0.37\* +0.24 +0.21 | 1/3 |  |
| E:P | [-0.150, +0.358] | -0.15 +0.36\* -0.13 | 1/3 |  |
| A:F | [-0.321, +0.117] | -0.32\* +0.12 -0.01 | 1/3 |  |
| E:A | [-0.099, +0.299] | -0.08 +0.30\* -0.10 | 1/3 |  |
| A:G | [-0.056, +0.283] | +0.28\* +0.02 -0.06 | 1/3 |  |
| A:L | [-0.147, +0.280] | -0.02 +0.28\* -0.15 | 1/3 |  |
| E:T | [-0.248, +0.219] | -0.25\* +0.22 -0.12 | 1/3 |  |

(15 highest-magnitude flip cells shown; 24 more flip cells below the cutoff (#sig, |log2OR|) = (1/3, 0.248) are in the data but not listed.)

### Flag glossary

- `iid-amp` — single per-timepoint `p_adj` falls below 1e-10. The magnitude of `-log10(p_adj)` is inflated by Fisher's anti-conservative behaviour at large pooled N + correlated stall positions; rank by effect size, not by p magnitude.
- `rare-aa` — at least one of the 3 per-timepoint rows for this cell has `BWM_k` < 100 (or similarly low `ctrl_k`). Odds-ratio estimate is unstable; large |log2OR| values may reflect sampling noise rather than biology.
- `both-flip-sig` — direction-flip cell where both opposite-sign timepoint rows reach `p_adj` < 0.05; the cell records a genuine FDR-significant sign change rather than a sub-threshold near-zero crossing. Only E:K qualifies in this file.

## Numbers at a glance
- `n_tests`: 180 (3 timepoints x 3 sites x 20 AAs)
- `n_significant` (adjusted-p < 0.05): 27 (15.0%)
- `n_significant` (adjusted-p < 0.10): 35 (19.4%)
- `min adjusted-p`: 2.8687614397053295e-15 (day_0 E K, log2OR=+0.480, BWM_k=983/n=6091, ctrl_k=3362/n=27732)
- `p_floor`: n/a — Fisher's exact has no analytic floor; effective discreteness floor at small (k_bwm + k_ctrl) is captured by `small-bh-family-discreteness`
- Cells with p_adj < 1e-10: 1 of 180 (only day_0 E K); flagged `iid-amp`
- Cross-(timepoint, site) BH families and their n_sig at p_adj<0.05:

| timepoint | A | P | E | BWM_total | control_total | ctrl:BWM ratio |
| --- | --- | --- | --- | --- | --- | --- |
| day_0 | 7/20 | 5/20 | 6/20 | 6091 | 27732 | 4.55 |
| day_5 | 1/20 | 1/20 | 5/20 | 11935 | 11177 | 0.94 |
| day_10 | 1/20 | 0/20 | 1/20 | 6945 | 8788 | 1.27 |

- Cross-timepoint (site, aa) cells, all 3 timepoints same OR direction: 21/60 (7 BWM-enriched, 14 BWM-depleted)
- Cross-timepoint (site, aa) cells with at least one sign change: 39/60
- Cells with sign change at p_adj<0.05 on both opposite-sign rows: 1 (E:K)

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 contingency BWM_count vs control_count for each (timepoint, site, aa), with BH-FDR correction within each (timepoint, site) family of 20 AAs; user confirmed. Effect column is `odds_ratio` (re-expressed as log2OR in this report's tables for symmetry with sister files); test statistic and exact p are returned by `scipy.stats.fisher_exact`. The p-correction column is `p_adj` (BH per (timepoint, site) family). The test answers "is the BWM-vs-control AA composition at this site different at this timepoint?". It does *not* answer "is each condition's per-aa enrichment vs background informative" (that is the `within_condition_binomial` family) and *not* "is the BWM-vs-control difference itself shifting across timepoints" (that requires a 3-way model or an explicit between-timepoint contrast — see the `between_timepoint_wilcoxon` family for the within-condition variant of that question).

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
  - The 21 cross-timepoint concordant cells (top by mean |log2OR|: P:K +0.253, P:R +0.239, A:R +0.218, A:T -0.178, A:W +0.172, P:T -0.166, A:H +0.153, A:C -0.146, P:A -0.137, P:S -0.136 — 7 enriched, 14 depleted overall). Do these reappear with the same direction in the codon sister file aggregated to AA, and in the within-condition Fisher and within-condition binomial AA files where applicable? If a cell is concordant here but reverses sign in a sister test, that flags either a design-axis difference (BWM-vs-control vs within-condition vs binomial-vs-bg) or an aggregation artifact.
  - The single sig-flip cell (E:K: day_0 BWM-enriched +0.480 at p_adj=2.87e-15, day_5 BWM-depleted -0.195 at p_adj=4.93e-3, day_10 BWM-depleted -0.035 ns). Does the codon sister file show the same flip pattern at any one of E:K's two codons (AAA, AAG), or does the AA-level flip dissolve when split by codon? Independently: does the within-condition Fisher show consistent E:K direction within each condition's BWM cells across d0/d5/d10? If the AA-level flip is real and not aggregation-driven, both routes should support it.
  - The day_0 A:N cell (log2OR=-0.755, p_adj=3.17e-08, BWM_k=134 ctrl_k=1014). It is the largest single-cell magnitude in the file but has BWM_k=134 (not rare by the k<100 cutoff but at the lower-end of the high-count range); does this magnitude reappear in the codon sister file at any of N's two codons (AAT, AAC), and does the within-condition AA Wilcoxon flag it as a closest-to-significant feature on either condition? If it shows up in only one of those, the magnitude may be an A-site rare-N edge effect at the high-N day_0 control arm rather than a stable feature.
  - The N-imbalance asymmetry: per-(timepoint, site) n_sig is 18 at day_0, 7 at day_5, 2 at day_10, while min p_adj follows the same gradient over 12 orders of magnitude. Does the codon sister file show the same gradient (consistent with N-driven power) or a different one (consistent with biology being timepoint-specific)? If both files show the same gradient with the same N-imbalance, the gradient is likely power-modulated; if the codon file shows a different gradient, that flags a resolution-dependent biological signal Chumeng should investigate.
  - The 15 sub-significant cross-timepoint concordant cells in the summary table (#sig=0/3, consistent direction). These are direction-coherent but not individually significant under any (timepoint, site) BH family. Does any of them reach significance at codon resolution (where some synonyms may carry the signal that AA aggregation dilutes), or under the within-condition binomial (where the direction-vs-background test has different power profile)? Useful to triangulate before discarding them as noise.
