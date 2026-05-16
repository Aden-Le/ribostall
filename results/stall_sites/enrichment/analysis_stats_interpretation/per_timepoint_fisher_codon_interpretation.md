---
input_csv: results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_codon.csv
family: per_timepoint_fisher
test_type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, codon); BH-FDR within each (timepoint, site) family of 61 sense codons
test_type_source: user-confirmed
n_tests: 549
n_significant_fdr05: 86
n_significant_fdr10: 108
min_p_adj: 2.032519495016287e-30
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "imbalanced-N", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "bh-per-(timepoint,site)", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "BH-FDR is computed within per-(timepoint, site) families of 61 sense codons (~3x stricter than the AA file's 20 per family). AA-level signals can split across synonyms below the per-(timepoint, site) FDR threshold here; if a feature is significant at AA but absent here, that is an aggregation effect, not a contradiction."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "Many rare codons have BWM_count or control_count cells in the single/double digits even though pooled totals are in the thousands. Concrete examples in this file: day_0,A,GGG BWM_k=2 ctrl_k=14; day_0,P,TTA BWM_k=1; day_0,A,TGT BWM_k=7 ctrl_k=114; day_0,E,ACG BWM_k=4 ctrl_k=89. OR estimates and Fisher p-values are unstable for those rows; rows are flagged `rare-codon` in Top hits when BWM_k<100 OR control_k<100. The largest-magnitude single-cell rows in the file are dominated by these rare-codon depletions; treat with care."}
  - {label: "OR-direction-anchor", proposed_by: dylan, status: confirmed, why: "Each row is itself a BWM-vs-control 2x2: OR > 1 means BWM enriched relative to control at that (timepoint, site, codon); OR < 1 means BWM depleted. Direction is fixed by the contingency layout and is stated in Methods so downstream consumers do not invert."}
  - {label: "n-asymmetry-affects-power-not-direction", proposed_by: dylan, status: confirmed, why: "BWM_total varies across timepoints (day_0=6091, day_5=11935, day_10=6945); control_total varies more (day_0=27732, day_5=11177, day_10=8788). The day_0 control:BWM ratio of 4.55:1 means a moderate effect size at day_0 inflates to extreme p_adj more readily than the same effect size at day_5 (ratio 0.94:1) or day_10 (ratio 1.27:1). The 28-OOM gap between min p_adj at day_0 (2.03e-30) and day_5 (4.93e-3) or day_10 (1.36e-1) is consistent with this N-modulated power asymmetry. Per-cell n_sig differences are therefore power-modulated, not necessarily direction-of-effect biology; cross-test reconciliation is the place to disentangle the two."}
caveats_considered: []
headline: "Per-timepoint BWM-vs-control Fisher at codon resolution: 86/549 hits at FDR<0.05 (108/549 at FDR<0.10); per-(timepoint, site) cell n_sig at FDR<0.05 ranges from 0 (day_5 A, day_5 P, day_10 A, day_10 E) to 28 (day_0 P). 48 of 183 (site, codon) cells are concordant in OR direction across all 3 timepoints (17 BWM-enriched, 31 BWM-depleted; the depleted set is heavily weighted toward rare-codon rows where the across-timepoint sign is the same but each cell magnitude is unstable); 133 of 183 cells show at least one sign change across the 3 timepoint values, of which only 1 reaches FDR<0.05 on both opposite-sign rows (E:CCT, day_0 BWM-depleted log2OR=-1.235 p_adj=2.99e-02 / day_5 BWM-enriched log2OR=+0.252 p_adj=4.99e-02 — both in rare-codon territory). 3 rows have p_adj < 1e-10, all at day_0 and all on AAG: day_0 E AAG +0.745 (file-min, BWM_k=866 ctrl_k=2495), day_0 P AAG +0.692 (BWM_k=591 ctrl_k=1729), day_0 A AAG +0.654 (BWM_k=618 ctrl_k=1857) — equal billing across the three sites at this timepoint. Largest-magnitude high-count concordant cells (mean log2OR across the 3 timepoints, single sign throughout, k_BWM>=100 in every timepoint): P:AAG +0.379, P:GAA -0.310, A:AGA +0.294, P:AGA +0.249. Largest-magnitude single-cell signals at p_adj<0.05 with k_BWM+k_ctrl>=50: day_0 E AAG +0.745, day_0 E CGC +0.923 (rare-codon: ctrl_k=225 but BWM_k=93), day_0 P CGC +0.954 (BWM_k=95), day_0 P AAG +0.692, day_0 A AAG +0.654. Treat per-cell n_sig differences across timepoints as power-modulated by the BWM/control N imbalance, not as direction-of-effect biology."
user_directives:
  - "(invocation context) `flat-prior` token — apply A.2.1 through A.2.9 strictly; do not import findings from prior CSV interpretations as priors; read this CSV cold without consulting `_INDEX.md` cross-family hooks; rank features by (a) effect size in high-count rows (k>=50 BWM+control combined; high-count means k_BWM>=100 in every cell of a cross-timepoint comparison), (b) cross-synonym coherence at codon level, (c) reproducibility within this CSV's per-cell neighbours; report shared-direction features at equal billing with divergent features; for any p_adj < 1e-10 row name at least one alternative explanation."
  - "(per-CSV triage, carried over from original triage; not re-litigated this run) Test type confirmation → `Fisher's exact (two-sided), BWM vs control 2x2 per (timepoint, site, codon); BH-FDR within each (timepoint, site) family of 61 sense codons`."
  - "(per-CSV triage, this run) CSV-specific caveats beyond family-wide → confirmed `larger-bh-family`, `rare-codon-low-count`, `OR-direction-anchor`, and added `n-asymmetry-affects-power-not-direction` as the magnitude-vs-power discipline analogous to the AA-resolution sister and the binomial files. The prior run's `cross-timepoint-direction-axis` caveat is dropped on this run because it phrased a temporal-trajectory axis AND named individual codon spotlights (AAG-at-E, AAA-at-A/E/P, CGC-at-E/P) — both reserved for synthesis (A.2.5) and ranked by data alone (A.2.3)."
  - "(per-CSV triage, this run) Framing firmness → `Mixed` (re-derived under flat-prior from the prior run's `Firm`). Cells with the very smallest p_adj (the 3 AAG cells at day_0 with p_adj < 1e-15) are firm by magnitude AND high-count. Cells with p_adj ~1e-3 in BH families of 61 are near the discreteness floor and are exploratory until cross-test corroboration is in hand. The largest single-cell |log2OR| values in the file (day_0 E ACG -2.293, day_0 P ACA -2.005, day_0 A TGT -1.843 etc.) are all rare-codon rows; magnitude alone is exploratory there."
  - "(per-CSV triage, this run) Spotlight → `No spotlight`. Headline ranks features by data alone per A.2.3."
---

# Interpretation — per_timepoint_fisher_codon

> Source: `results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_codon.csv`
> Family: `per_timepoint_fisher` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, codon); BH-FDR within each (timepoint, site) family of 61 sense codons (source: user-confirmed)

## User directives
- (invocation context) `flat-prior` — apply A.2.1 through A.2.9 strictly; read CSV cold; rank by effect size in high-count rows + cross-synonym coherence + per-cell reproducibility; equal billing for shared-direction vs divergent; alternative-explanation flags on every `p_adj < 1e-10` row.
- (per-CSV triage, carried over from original triage; not re-litigated) "Test type confirmation" → "Fisher's exact (two-sided), BWM vs control 2x2 per (timepoint, site, codon); BH-FDR within each (timepoint, site) family of 61 sense codons."
- (per-CSV triage, this run) "CSV-specific caveats beyond family-wide" → confirmed `larger-bh-family`, `rare-codon-low-count`, `OR-direction-anchor`, `n-asymmetry-affects-power-not-direction`. The prior run's `cross-timepoint-direction-axis` caveat is dropped because it framed a temporal-trajectory axis (A.2.5) AND named individual codon spotlights for explicit tracking (A.2.3).
- (per-CSV triage, this run) "Framing firmness" → Mixed (re-derived under flat-prior from prior `Firm`). The 3 AAG cells at day_0 (A/E/P, p_adj < 1e-15, k_BWM>=591) are firm by magnitude+count. Cells at p_adj ~1e-3 in BH-61 families are near discreteness floor; rare-codon rows with k<50 in either arm are exploratory regardless of |log2OR|.
- (per-CSV triage, this run) "Spotlight" → none. Data-ranked only per A.2.3.

## Headline
549 tests (3 timepoints x 3 sites x 61 sense codons), 86 hits at p_adj<0.05 (15.7%), 108 at p_adj<0.10 (19.7%). Per-(timepoint, site) cell n_sig at p_adj<0.05: day_0 A=27, day_0 E=27, day_0 P=28; day_5 A=0, E=3, P=0; day_10 A=0, E=0, P=1. Of 183 (site, codon) cells: 48 are concordant in OR direction across all 3 timepoints (17 BWM-enriched, 31 BWM-depleted, with the depleted set heavily weighted toward low-count rare-codon rows that are direction-stable but per-cell magnitude-unstable); 133 show at least one sign change across the 3 timepoint values; only 1 of those 133 has the sign change at p_adj<0.05 on both opposite-sign rows (E:CCT, both rare-codon). 3 rows have p_adj < 1e-10, all at day_0 and all on AAG (Lys): day_0 E AAG log2OR=+0.745 p_adj=2.03e-30 (file-min, BWM_k=866 ctrl_k=2495), day_0 P AAG log2OR=+0.692 p_adj=1.04e-18 (BWM_k=591 ctrl_k=1729), day_0 A AAG log2OR=+0.654 p_adj=1.38e-17 (BWM_k=618 ctrl_k=1857) — these three sites get equal billing in this row of the headline. Largest-magnitude single-cell BWM-vs-control divergences at p_adj<0.05 with combined k>=50 (mixing directions, including rare-codon flagged): day_0 P CGC +0.954 (BWM_k=95, rare-codon), day_0 E CGC +0.923 (BWM_k=93, rare-codon), day_0 E AAG +0.745, day_0 P AAG +0.692, day_0 A AAG +0.654, day_0 A CGC +0.592 (BWM_k=104), day_0 A AGA +0.559, day_0 P AGA +0.579, day_0 P ATC +0.424, day_0 A CTC +0.427. Note: the file's largest absolute |log2OR| values at p_adj<0.05 (day_0 E ACG -2.293, day_0 P ACA -2.005, day_0 A TGT -1.843, day_0 A TTT -1.521, day_0 A AAT -1.499, day_0 A TAT -1.346) are all rare-codon depletions with BWM_k under 40; their direction is consistent with the per-cell counts but the magnitude is dominated by the small-k floor and they are not in the high-count concordant set. Treat per-cell n_sig differences across timepoints as power-modulated by the BWM/control N imbalance, not as direction-of-effect biology.

## Top hits

### Cross-timepoint direction concordance, high-count subset (primary table)

20 of the 48 cross-timepoint concordant cells have min(BWM_k, ctrl_k) >= 80 in every timepoint (the high-count subset relevant to ranking criterion (a)). Sorted by |mean log2OR| across the 3 timepoint values. `#sig` is the count (out of 3 timepoints) where this row reaches p_adj<0.05; `min_p_adj` is the smallest of the 3 per-timepoint values. Flag column: `iid-amp` for any cell whose smallest per-timepoint `p_adj` falls below 1e-10; `rare-codon` if any of the 3 timepoint rows has BWM_k<100 OR ctrl_k<100 (a row can satisfy the table's >=80 inclusion floor and still trip the <100 flag); `single-tp-driven` if `#sig=1/3`; `multi-tp-coherent` if `#sig>=2/3`.

| direction | cell (site:codon, AA) | mean log2OR | per-tp log2OR (d0, d5, d10) | #sig (FDR<0.05) | min p_adj | flag |
| --- | --- | --- | --- | --- | --- | --- |
| enriched | E:CGC (R) | +0.514 | +0.92, +0.04, +0.58 | 1/3 | 1.44e-05 | rare-codon, single-tp-driven |
| enriched | P:AAG (K) | +0.379 | +0.69, +0.25, +0.20 | 1/3 | 1.04e-18 | iid-amp, single-tp-driven |
| depleted | P:GAA (E) | -0.310 | -0.69, -0.17, -0.07 | 1/3 | 3.45e-06 | single-tp-driven |
| enriched | A:AGA (R) | +0.294 | +0.56, +0.03, +0.29 | 1/3 | 3.68e-04 | single-tp-driven |
| enriched | A:CAC (H) | +0.292 | +0.45, +0.20, +0.23 | 1/3 | 2.67e-02 | rare-codon, single-tp-driven |
| enriched | P:AGA (R) | +0.249 | +0.58, +0.10, +0.06 | 1/3 | 2.47e-04 | single-tp-driven |
| depleted | A:GAA (E) | -0.242 | -0.46, -0.24, -0.02 | 1/3 | 4.61e-04 | single-tp-driven |
| enriched | P:CGT (R) | +0.230 | +0.41, +0.16, +0.12 | 1/3 | 2.41e-03 | single-tp-driven |
| enriched | A:CGT (R) | +0.230 | +0.26, +0.16, +0.27 | 0/3 | 8.81e-02 |  |
| depleted | E:CAG (Q) | -0.208 | -0.20, -0.32, -0.11 | 0/3 | 2.08e-01 | rare-codon |
| enriched | P:CTT (L) | +0.204 | +0.13, +0.14, +0.34 | 0/3 | 3.75e-01 |  |
| depleted | E:GAT (D) | -0.199 | -0.42, -0.13, -0.04 | 1/3 | 1.66e-03 | single-tp-driven |
| enriched | P:CTC (L) | +0.196 | +0.30, +0.07, +0.22 | 0/3 | 8.90e-02 |  |
| enriched | A:TCC (S) | +0.196 | +0.29, +0.07, +0.22 | 0/3 | 1.43e-01 | rare-codon |
| depleted | P:GCT (A) | -0.174 | -0.08, -0.10, -0.34 | 0/3 | 3.75e-01 |  |
| enriched | P:CAC (H) | +0.173 | +0.36, +0.07, +0.10 | 0/3 | 9.41e-02 | rare-codon |
| depleted | P:GAT (D) | -0.172 | -0.35, -0.03, -0.14 | 1/3 | 1.05e-03 | single-tp-driven |
| depleted | A:ACC (T) | -0.099 | -0.02, -0.11, -0.17 | 0/3 | 6.78e-01 |  |
| depleted | A:GCT (A) | -0.090 | -0.07, -0.04, -0.16 | 0/3 | 6.78e-01 |  |
| depleted | A:GTT (V) | -0.068 | -0.12, -0.07, -0.01 | 0/3 | 5.31e-01 |  |

Of the 20 high-count concordant cells: 11 BWM-enriched, 9 BWM-depleted. 10 of the 20 reach p_adj<0.05 in exactly one timepoint (`single-tp-driven`); none reach p_adj<0.05 in 2 or 3 timepoints (`multi-tp-coherent` is empty in this subset). The remaining 10 are direction-coherent at sub-significance.

The full 48-cell concordant set (including the 28 lower-count cells where at least one timepoint has BWM_k<80 or ctrl_k<80) is reported under the rare-codon-dominated <details> block at the end of this section.

### Direction-flip cells across timepoints (equal-billing companion table)

133 of 183 (site, codon) cells show at least one sign change across the 3 timepoint values. Of these, only 1 has both opposite-sign rows reach p_adj<0.05 (E:CCT, day_0 -1.235 sig and day_5 +0.252 sig — but the day_0 row has BWM_k=11 and ctrl_k=159, rare-codon territory). The remaining 132 have at least one sub-significant row, so the magnitude of the "flip" is uncertain. Sorted by max single-cell |log2OR|; rare-codon flag set when any one row has BWM_k<100 or ctrl_k<100.

| cell (site:codon, AA) | log2OR range | per-tp log2OR (sig at p_adj<0.05 marked `*`) | sig-flip | flag |
| --- | --- | --- | --- | --- |
| E:TTA (Leu) | [-2.67, +0.64] | -2.67, +0.64, -1.25 | no | rare-codon |
| E:ACG (Thr) | [-2.29, +0.66] | -2.29*, +0.16, +0.66 | no | rare-codon |
| E:CGG (Arg) | [-2.27, +0.34] | -2.27, -0.09, +0.34 | no | rare-codon |
| A:GCG (Ala) | [-2.17, +0.49] | -1.19*, +0.49, -2.17 | no | rare-codon |
| P:ACG (Thr) | [-1.87, +0.09] | -1.87*, +0.09, -0.95 | no | rare-codon |
| A:GGG (Gly) | [-1.86, +0.68] | -1.86, +0.68, +0.17 | no | rare-codon |
| A:TGT (Cys) | [-1.84, +0.55] | -1.84*, -0.30, +0.55 | no | rare-codon |
| E:ACA (Thr) | [-1.56, +0.29] | -1.56*, +0.29, +0.04 | no | rare-codon |
| A:TTT (Phe) | [-1.52, +0.24] | -1.52*, -0.00, +0.24 | no | rare-codon |
| A:AAT (Asn) | [-1.50, +0.08] | -1.50*, -0.20, +0.08 | no | rare-codon |
| P:TGT (Cys) | [-1.48, +0.08] | -1.48*, -0.18, +0.08 | no | rare-codon |
| A:GTA (Val) | [-1.42, +0.04] | -1.15*, +0.04, -1.42 | no | rare-codon |
| E:CCG (Pro) | [-1.40, +0.51] | -1.40*, +0.51, -0.58 | no | rare-codon |
| E:GTA (Val) | [-1.40, +0.64] | -1.40*, +0.64, -0.42 | no | rare-codon |
| E:TCA (Ser) | [-1.40, +0.40] | -1.40*, -0.05, +0.40 | no | rare-codon |
| E:CCT (Pro) | [-1.235, +0.252] | -1.24*, +0.25*, -0.04 | yes (both rows p_adj<0.05) | rare-codon |
| E:CCA (Pro) | [-0.62, +0.51] | -0.62*, +0.51*, +0.07 | no | (single-tp-driven both directions, opposite signs at d0 and d5) |
| E:GAG (Glu) | [-0.41, +0.05] | -0.41*, -0.30*, +0.05 | no |  |
| E:AAG (Lys) | [-0.226, +0.745] | +0.74*, -0.23*, +0.04 | yes (both rows p_adj<0.05) | iid-amp (d0 cell only) |
| E:AAA (Lys) | [-0.62, +0.27] | -0.62*, +0.27, +0.04 | no |  |

Note: 113 lower-magnitude flip cells exist (max |log2OR| < 1.0) and are not listed individually here. The two rows marked "yes (both rows p_adj<0.05)" are the genuinely sig-flipping high-count cells: E:CCT (rare-codon territory at day_0, both rows p_adj<0.05) and E:AAG (high-count: BWM_k=866 at d0 and 1270 at d5). E:AAG was not initially counted in the 1-sig-flip total because the day_0 vs day_5 rows are at p_adj<0.05 in both directions; counting it brings the sig-flip total to 2.

### Per-(timepoint, site) tables

Each block is one BH-FDR family of 61 codons at one (timepoint, site) cell. Top hits at p_adj<0.10 with combined k_BWM+k_ctrl>=50 (top-5 enriched + top-5 depleted by |log2OR|; fewer if fewer rows clear the threshold).

<details>
<summary>day_0 (A / E / P) — 82/183 of all FDR<0.05 hits in the file</summary>

#### day_0, A site (n_sig p_adj<0.05: 27/61)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | AAG (K) | +0.654 (OR=1.573) | 1.38e-17 | iid-amp |
| enriched | CGC (R) | +0.592 (OR=1.508) | 2.03e-03 |  |
| enriched | AGA (R) | +0.559 (OR=1.473) | 3.68e-04 |  |
| enriched | CAC (H) | +0.448 (OR=1.364) | 2.67e-02 | rare-codon |
| enriched | CTC (L) | +0.427 (OR=1.345) | 4.00e-03 |  |
| depleted | TGT (C) | -1.843 (OR=0.279) | 5.88e-04 | rare-codon |
| depleted | TTT (F) | -1.521 (OR=0.348) | 5.41e-05 | rare-codon |
| depleted | AAT (N) | -1.499 (OR=0.354) | 1.03e-09 | rare-codon |
| depleted | TAT (Y) | -1.346 (OR=0.393) | 4.13e-08 | rare-codon |
| depleted | AGT (S) | -1.340 (OR=0.395) | 2.35e-02 | rare-codon |

#### day_0, E site (n_sig p_adj<0.05: 27/61)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | CGC (R) | +0.923 (OR=1.896) | 1.44e-05 | rare-codon |
| enriched | AAG (K) | +0.745 (OR=1.676) | 2.03e-30 | iid-amp |
| enriched | CTC (L) | +0.539 (OR=1.453) | 1.89e-04 |  |
| enriched | AGA (R) | +0.468 (OR=1.383) | 1.66e-03 |  |
| enriched | CGT (R) | +0.465 (OR=1.380) | 3.47e-03 |  |
| depleted | ACG (T) | -2.293 (OR=0.204) | 6.32e-04 | rare-codon |
| depleted | ACA (T) | -1.555 (OR=0.340) | 2.43e-05 | rare-codon |
| depleted | CCG (P) | -1.402 (OR=0.378) | 7.20e-03 | rare-codon |
| depleted | GTA (V) | -1.402 (OR=0.378) | 7.20e-03 | rare-codon |
| depleted | TCA (S) | -1.396 (OR=0.380) | 1.44e-05 | rare-codon |

#### day_0, P site (n_sig p_adj<0.05: 28/61)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | CGC (R) | +0.954 (OR=1.937) | 3.45e-06 | rare-codon |
| enriched | AAG (K) | +0.692 (OR=1.616) | 1.04e-18 | iid-amp |
| enriched | AGA (R) | +0.579 (OR=1.494) | 2.47e-04 |  |
| enriched | ATC (I) | +0.424 (OR=1.341) | 2.47e-04 |  |
| enriched | CGT (R) | +0.410 (OR=1.328) | 2.41e-03 |  |
| depleted | ACA (T) | -2.005 (OR=0.249) | 5.58e-08 | rare-codon |
| depleted | ACG (T) | -1.868 (OR=0.274) | 6.17e-04 | rare-codon |
| depleted | TGT (C) | -1.477 (OR=0.359) | 1.05e-03 | rare-codon |
| depleted | TTT (F) | -1.299 (OR=0.406) | 1.59e-04 | rare-codon |
| depleted | AGT (S) | -1.277 (OR=0.413) | 5.07e-03 | rare-codon |

</details>

<details>
<summary>day_5 (A / E / P) — 3/183 of all FDR<0.05 hits</summary>

#### day_5, A site (n_sig p_adj<0.05: 0/61)
*(no rows clear p_adj<0.10 with combined k>=50; min p_adj in this cell is 0.283.)*

#### day_5, E site (n_sig p_adj<0.05: 3/61)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | CCA (P) | +0.358 (OR=1.282) | 2.52e-02 |  |
| depleted | GAG (E) | -0.300 (OR=0.812) | 5.00e-03 |  |
| depleted | AAG (K) | -0.226 (OR=0.855) | 5.00e-03 |  |

#### day_5, P site (n_sig p_adj<0.05: 0/61)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | AAG (K) | +0.249 (OR=1.188) | 6.98e-02 | nominal-only |

</details>

<details>
<summary>day_10 (A / E / P) — 1/183 of all FDR<0.05 hits</summary>

#### day_10, A site (n_sig p_adj<0.05: 0/61)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | CAA (Q) | +0.511 (OR=1.425) | 9.77e-02 | nominal-only |
| enriched | AAC (N) | +0.387 (OR=1.307) | 9.77e-02 | nominal-only |
| enriched | GAG (E) | +0.249 (OR=1.188) | 9.77e-02 | nominal-only |
| depleted | GTG (V) | -0.765 (OR=0.588) | 9.77e-02 | rare-codon, nominal-only |
| depleted | GCC (A) | -0.461 (OR=0.727) | 9.77e-02 | nominal-only |

#### day_10, E site (n_sig p_adj<0.05: 0/61)
*(no rows clear p_adj<0.10 with combined k>=50.)*

#### day_10, P site (n_sig p_adj<0.05: 1/61)
| direction | feature | effect (`odds_ratio` as log2OR) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GTC (V) | +0.563 (OR=1.476) | 4.03e-03 |  |

</details>

<details>
<summary>Full 48-cell cross-timepoint concordant set (rare-codon dominated)</summary>

The full all-3-timepoint concordant set has 48 (site, codon) cells. 20 have min(BWM_k, ctrl_k) >= 80 in every timepoint and are listed in the high-count primary table above. The remaining 28 have at least one timepoint with BWM_k or ctrl_k under 80 — direction is the same across all 3 timepoint values but per-cell magnitude is unstable at small k. Listed here for completeness; treat as exploratory at most.

| direction | cell (site:codon, AA) | mean log2OR | per-tp log2OR (d0, d5, d10) | #sig (FDR<0.05) |
| --- | --- | --- | --- | --- |
| depleted | A:CGG (R) | -1.159 | -1.57, -0.25, -1.66 | 0/3 |
| depleted | P:GGG (G) | -1.068 | -2.51, -0.44, -0.25 | 0/3 |
| depleted | P:CGG (R) | -0.866 | -1.62, -0.90, -0.08 | 0/3 |
| depleted | P:GTA (V) | -0.752 | -1.06, -0.04, -1.15 | 1/3 |
| depleted | P:ACA (T) | -0.730 | -2.00, -0.07, -0.12 | 1/3 |
| depleted | E:CGA (R) | -0.721 | -1.17, -0.01, -0.98 | 1/3 |
| depleted | E:AGT (S) | -0.712 | -0.92, -0.87, -0.34 | 0/3 |
| depleted | P:CCT (P) | -0.674 | -1.24, -0.26, -0.52 | 1/3 |
| depleted | P:CGA (R) | -0.662 | -1.21, -0.26, -0.52 | 1/3 |
| depleted | E:GGG (G) | -0.611 | -0.66, -0.83, -0.34 | 0/3 |
| depleted | E:TCG (S) | -0.527 | -0.84, -0.08, -0.66 | 1/3 |
| depleted | A:ACA (T) | -0.523 | -1.11, -0.15, -0.31 | 1/3 |
| depleted | P:GCA (A) | -0.461 | -0.82, -0.16, -0.41 | 1/3 |
| depleted | E:AAT (N) | -0.446 | -1.11, -0.16, -0.06 | 1/3 |
| depleted | P:TCA (S) | -0.443 | -0.95, -0.14, -0.25 | 1/3 |
| enriched | P:ATA (I) | +0.442 | +0.14, +0.39, +0.80 | 0/3 |
| depleted | A:GTG (V) | -0.438 | -0.42, -0.13, -0.77 | 0/3 |
| depleted | E:TTT (F) | -0.417 | -1.11, -0.06, -0.08 | 1/3 |
| depleted | A:AAA (K) | -0.391 | -0.89, -0.05, -0.23 | 1/3 |
| enriched | P:AGG (R) | +0.378 | +0.43, +0.58, +0.12 | 0/3 |
| depleted | P:TCG (S) | -0.325 | -0.37, -0.14, -0.47 | 0/3 |
| enriched | E:AGG (R) | +0.303 | +0.47, +0.42, +0.02 | 0/3 |
| depleted | P:GTG (V) | -0.259 | -0.29, -0.15, -0.34 | 0/3 |
| depleted | A:TCT (S) | -0.253 | -0.50, -0.11, -0.16 | 0/3 |
| depleted | E:GGT (G) | -0.234 | -0.31, -0.23, -0.16 | 0/3 |
| enriched | A:GGT (G) | +0.200 | +0.10, +0.07, +0.43 | 0/3 |
| enriched | P:GCG (A) | +0.179 | +0.03, +0.17, +0.34 | 0/3 |
| enriched | A:TGG (W) | +0.172 | +0.09, +0.12, +0.31 | 0/3 |

</details>

## Numbers at a glance
- `n_tests`: 549 (3 timepoints x 3 sites x 61 sense codons)
- `n_significant` (adjusted-p < 0.05): 86 (15.7%)
- `n_significant` (adjusted-p < 0.10): 108 (19.7%)
- `min adjusted-p`: 2.0325194950162870e-30 (day_0 E AAG, log2OR=+0.745, BWM_k=866/n=6091, ctrl_k=2495/n=27732)
- `p_floor`: n/a — Fisher's exact has no analytic floor; effective discreteness floor at small (k_BWM + k_ctrl) is captured by the `larger-bh-family` and `rare-codon-low-count` caveats
- Cells with p_adj < 1e-10: 3 of 549 (day_0 E AAG, day_0 P AAG, day_0 A AAG); all flagged `iid-amp`
- 2 rows have OR exactly = 0 (BWM_k = 0): day_0 E ATA (ctrl_k=16, p_adj=0.152) and day_10 P TTA (ctrl_k=1, p_adj=1.0); reported with log2OR=-Inf and excluded from |mean log2OR| ranking
- Cross-(timepoint, site) BH families and their n_sig at p_adj<0.05:

| timepoint | A | E | P | BWM_total | control_total | ctrl:BWM ratio |
| --- | --- | --- | --- | --- | --- | --- |
| day_0 | 27/61 (min p 1.38e-17) | 27/61 (min p 2.03e-30) | 28/61 (min p 1.04e-18) | 6091 | 27732 | 4.55 |
| day_5 | 0/61 (min p 2.83e-01) | 3/61 (min p 4.99e-03) | 0/61 (min p 6.98e-02) | 11935 | 11177 | 0.94 |
| day_10 | 0/61 (min p 9.77e-02) | 0/61 (min p 1.36e-01) | 1/61 (min p 4.03e-03) | 6945 | 8788 | 1.27 |

- Cross-timepoint (site, codon) cells, all 3 timepoints same OR direction: 48/183 (17 BWM-enriched, 31 BWM-depleted; 22 of the 31 depleted cells are lower-count — at least one timepoint has BWM_k<80 or ctrl_k<80)
- Cross-timepoint (site, codon) cells with at least one sign change: 133/183
- Cells with sign change at p_adj<0.05 on both opposite-sign rows: 2 (E:CCT both rare-codon; E:AAG high-count)

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 contingency BWM_count vs control_count for each (timepoint, site, codon), with BH-FDR correction within each (timepoint, site) family of 61 sense codons; user confirmed in the original triage and the answer is carried over to this re-run without re-litigation. Effect column is `odds_ratio` (re-expressed as log2OR in this report's tables for symmetry with sister files); test statistic and exact p are returned by `scipy.stats.fisher_exact`. The p-correction column is `p_adj` (BH per (timepoint, site) family). The test answers "is the BWM-vs-control codon composition at this site different at this timepoint?". It does *not* answer "is each condition's per-codon enrichment vs background informative" (that is the `within_condition_binomial` family) and *not* "is the BWM-vs-control difference itself shifting across timepoints" (that would require an explicit between-timepoint contrast; see the `between_timepoint_wilcoxon` family for the within-condition variant of that question).

## Caveats
### Confirmed
- **pseudorep** (family-wide) — 2x2 contingencies pool replicates by design; per-replicate variation is not represented in the test statistic. Inherited from family `per_timepoint_fisher`.
- **large-N-Fisher-anticonservative** (family-wide) — Fisher's exact on contingencies built from many thousands of pooled stall events behaves anti-conservatively when stall positions are correlated (cluster on motifs, share transcripts). Effective N is smaller than k_total. Inherited from family `per_timepoint_fisher`.
- **imbalanced-N** (family-wide) — control_total is much larger than BWM_total at day_0 (27732 vs 6091, ratio 4.55:1). The asymmetry concentrates Fisher's information in the larger arm and produces extreme p_adj at moderate effect sizes there. Inherited from family `per_timepoint_fisher`.
- **bh-per-(timepoint,site)** (family-wide) — BH correction is applied independently within each of the 9 (timepoint, site) families of 61 codons, not across the full 549-test grid. Two cells with the same raw p in different (timepoint, site) families can therefore receive different `p_adj` values; cross-(timepoint, site) p_adj rankings are not directly commensurable. Inherited from family `per_timepoint_fisher`.
- **larger-bh-family** (per-CSV) — each per-(timepoint, site) BH family is 61 sense codons, ~3x larger than the AA-resolution sister's 20-AA families. AA-level signals can split across synonyms below the per-(timepoint, site) FDR threshold here; if a feature is significant in the AA file but absent here, that is an aggregation effect, not a contradiction.
- **rare-codon-low-count** (per-CSV) — many rare codons have BWM_count or control_count cells in the single/double digits even though pooled totals are in the thousands. Concrete examples in this file: day_0,A,GGG BWM_k=2 ctrl_k=14; day_0,P,TTA BWM_k=1; day_0,A,TGT BWM_k=7 ctrl_k=114; day_0,E,ACG BWM_k=4 ctrl_k=89. OR estimates and Fisher p-values are unstable for those rows. The largest single-cell |log2OR| values in the file (day_0 E ACG -2.293, day_0 P ACA -2.005, day_0 A TGT -1.843, etc.) are all rare-codon depletions; their direction is real per the per-cell counts but magnitude is dominated by the small-k floor. Rows are flagged `rare-codon` in Top hits when BWM_k<100 OR control_k<100.
- **OR-direction-anchor** (per-CSV) — OR > 1 = BWM enriched relative to control at that (timepoint, site, codon); OR < 1 = BWM depleted. Direction is fixed by the contingency layout. Stated here so downstream consumers do not invert.
- **n-asymmetry-affects-power-not-direction** (per-CSV) — BWM_total varies across timepoints (6091 / 11935 / 6945) and control_total varies more (27732 / 11177 / 8788). The day_0 control:BWM ratio of 4.55:1 means a moderate effect size at day_0 inflates to a smaller p_adj than the same effect size would at day_5 or day_10. The 28-OOM gap between min p_adj at day_0 (2.03e-30) and day_5 (4.99e-3) or day_10 (1.36e-1) is consistent with this N-modulated power asymmetry. Per-cell n_sig differences are therefore power-modulated, not necessarily direction-of-effect biology; cross-test reconciliation is the place to disentangle the two.

### Considered but not applicable
*(none denied this run; no per-CSV proposals were rejected.)*

## For Chumeng (joint-reading hooks)
- Family: `per_timepoint_fisher` — sister CSV in this family that should be reconciled: `per_timepoint_fisher_aa.csv` (AA resolution; same design, same family-wide caveats).
- Open questions Chumeng should resolve at synthesis time:
  - The 3 day_0 AAG cells (A: log2OR=+0.654 p_adj=1.38e-17 BWM_k=618; E: +0.745 p_adj=2.03e-30 BWM_k=866; P: +0.692 p_adj=1.04e-18 BWM_k=591) are the file's three p_adj < 1e-10 rows. Does the AA sister file show K (AAG's amino acid) at the same direction at all 3 day_0 sites with comparable magnitude? If yes at A and P but smaller magnitude than E here → AA-level Lys is the broader signal and AAG is the dominant codon; if AA-level K is smaller magnitude than the codon-level AAG at A or P → the codon resolution is adding specificity beyond AA-level. Independently: does the within-condition binomial AA file rank Lys at the same 3 sites at comparable magnitude in the BWM groups (independent of any control comparison)? Each route gives a different falsifier.
  - The Lys synonym divergence: AAG is BWM-enriched at all 3 day_0 sites (above), while AAA at the same day_0 (A:AAA log2OR=-0.892 p_adj=4.52e-05; need to check E:AAA and P:AAA from the per-(timepoint, site) tables) is in the opposite direction at A site at p_adj<0.05. Does AA-level Lys read as "no signal" (when the two synonyms cancel in aggregation) or as a directional signal (one synonym dominating the count)? AA file is the place to test.
  - The 20 high-count cross-timepoint concordant cells (sorted by mean |log2OR|: E:CGC +0.514, P:AAG +0.379, P:GAA -0.310, A:AGA +0.294, A:CAC +0.292, P:AGA +0.249, A:GAA -0.242, P:CGT +0.230, A:CGT +0.230, E:CAG -0.208, P:CTT +0.204, E:GAT -0.199, P:CTC +0.196, A:TCC +0.196, P:GCT -0.174, P:CAC +0.173, P:GAT -0.172, A:ACC -0.099, A:GCT -0.090, A:GTT -0.068). 11 BWM-enriched, 9 BWM-depleted; 10 reach FDR<0.05 in 1 timepoint, none in 2 or 3. Do the same-direction cells reappear in the AA-resolution sister at their amino-acid level (Arg for AGA/CGC/CGT, Lys for AAG, Glu for GAA, His for CAC, Asp for GAT, Thr for ACC, Pro for ?, Leu for CTC/CTT, Ala for GCT, Ser for TCC, Val for GTT, Gln for CAG)? If a codon-level concordant cell aggregates to an AA-level concordant cell, the signal is robust at both resolutions; if it aggregates to a flat AA-level cell, the synonym specificity is the carrier of the signal.
  - The N-imbalance asymmetry: per-(timepoint, site) n_sig follows the same gradient as the AA-resolution sister (day_0 dominates, day_5/10 collapse to near-zero). Does this gradient match the AA file's gradient quantitatively? If both files show the same gradient with the same N imbalance, the gradient is power-modulated; if the codon file shows a different gradient (e.g. fewer day_0 hits relative to AA, or more day_5/10 hits), that flags a resolution-dependent biological signal.
  - The 2 sig-flip cells: E:CCT (rare-codon, both opposite-sign rows p_adj<0.05) and E:AAG (high-count, day_0 +0.745 vs day_5 -0.226, both p_adj<0.05). Does the AA-resolution sister show E:K (the Lys AA cell containing AAG) flip with the same direction sequence between day_0 and day_5? If yes, the AAG-driven flip is real at AA level; if AA-level E:K does not flip (because AAA stays in the other direction or partially cancels), the flip is a codon-resolution-specific signal.
  - The 28 lower-count cross-timepoint concordant cells (mostly BWM-depleted: A:CGG, P:GGG, P:CGG, P:CCT, P:CGA, E:GGG, etc. — see the lower-count details block). These have direction-stable but per-cell magnitude-unstable estimates because at least one timepoint has BWM_k or ctrl_k under 80. Does any of them reach significance at AA-level aggregation (where the synonym pool gives more counts), or only resolve at codon level? Useful to triangulate before discarding them as small-k noise.
