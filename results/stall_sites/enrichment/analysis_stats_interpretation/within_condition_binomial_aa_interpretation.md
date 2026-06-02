---
input_csv: results/stall_sites/enrichment/analysis_stats/within_condition_binomial_aa.csv
family: within_condition_binomial
test_type: One-sample binomial test (k=stall_count out of n=total_n vs H0: p=bg_freq, two-sided), BH-FDR within each (group, site) family of ~20 aa
test_type_source: user-confirmed
n_tests: 360
n_significant_fdr05: 239
n_significant_fdr10: 264
min_p_adj: 3.4349119166303614e-132
p_floor: null
pseudoreplicated: true
synced_from_olive_qmd: 2026-06-02
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "iid-violation-binomial", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "bg-pseudocount-1e-6", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "bh-per-(group,site)", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "rare-aa-low-count", proposed_by: dylan, status: confirmed, why: "Several aa cells have observed_count below 100 in some groups (lowest: BWM_d0 E:C k=34; BWM_d10 P:W k=36). Their log2_enrichment estimates are noisier than common-aa cells. Under the #sig>=2 reproducibility floor both clear: E:C (min count 34, #sig 6/6) and P:W (min count 36, #sig 5/6) appear in the concordant-depletion table, and the discordant table carries rare-aa A:W, A:C, P:C. Rows with k<100 in any group carry a `rare-aa` flag in Top hits; direction is reliable for these rows, magnitude is not."}
  - {label: "small-bh-family-discreteness", proposed_by: dylan, status: confirmed, why: "Per (group, site) BH-FDR family is only ~20 aa tests; with the iid-violation-driven extreme p magnitudes, discreteness rarely binds (239/360 hits at FDR<0.05), but flag for parity with the sister codon file's larger BH families."}
  - {label: "p-magnitude-anchored-ranking", proposed_by: dylan, status: confirmed, why: "p_adj at 1e-15 to 1e-132 in this file reflects the iid-violation-binomial caveat amplifying real but moderate effects (typically log2≈0.6-0.8 at k>1000, n>20000) more than it reflects biological strength. Per A.2.4, every p_adj<1e-10 row in the Top hits tables carries an `iid-amp` flag (and `bg-tight` for rows with bg_freq>0.05). Rank by log2_enrichment + stall_count + cross-group reproducibility, not by p magnitude."}
caveats_considered: []
headline: "Within-group one-sample binomial vs bg_freq across 6 groups (BWM and control × d0/d5/d10), 360 tests, BH-FDR per (group, site) family of 20: 239/360 hits at FDR<0.05 (264/360 at FDR<0.10); file min p_adj = 3.43e-132 at control_d0 E:K (log2_enrichment=+0.625, k=3362/27732). The dominant structure is cross-group concordance: of the 60 (site, aa) cells, 13 are concordantly enriched and 19 concordantly depleted across all 6 groups, leaving 28 with at least one cross-group sign disagreement; the largest-magnitude cells (E:K, P:D enriched; A:A, P:A, E:A depleted) hold one direction across all 6 groups and no comparable-magnitude direction-flip cell exists in the data. Top hits are three sign-partitioned cross-group tables (concordant enrichment / concordant depletion / discordant), each showing every cell significant in >=2 of the 6 groups (a #sig>=2 reproducibility floor, not a row cap; 13/18/21 cells displayed), ranked by #sig desc then min count desc, displayed A/P/E. p_adj down to 1e-132 is co-amplified by iid-violation-binomial and common-aa bg_freq tightness; magnitude-plus-reproducibility is the anchor."
user_directives:
  - "(invocation context) `flat-prior` → A.2.9 strict: rank cold from this CSV alone; no priors imported from prior interpretation files; no _INDEX.md cross-family lookup."
  - "(invocation context) `Rank features by (a) effect size in high-count rows (k≥50), (b) cross-synonym coherence at codon level, (c) reproducibility within this CSV's per-cell neighbours` → applied; (b) is N/A at AA resolution and is referred to the sister codon CSV in `For Chumeng`."
  - "(invocation context) `Report shared-direction features at equal billing with divergent features` → applied per A.2.2; headline reports both axes and notes the empirical asymmetry (no comparable-magnitude divergent cells exist in this CSV)."
  - "(invocation context) `For any p_adj < 1e-10 row, name at least one alternative explanation` → applied per A.2.6; every Top hits row with p_adj<1e-10 carries `iid-amp` (and `bg-tight` for bg_freq>0.05) in the flag column."
  - "(invocation context) Banned-terminology rule (A.2.1) → confirmed; A.2.8 self-check on the rendered draft returned zero matches across headline, Top hits, caveats, and joint-reading hooks."
  - "(readback) Reconciled shared content from the corrected .qmd on 2026-06-02 → replaced the old per-group concordance + 6 per-group sub-tables with the .qmd's three sign-partitioned cross-group tables (concordant enrichment / depletion / discordant) under the #sig>=2 floor (13/18/21 displayed); dropped the stall_count>=200 gate from Methods + the rare-aa caveat; corrected the per-(group,site) Numbers-at-a-glance counts (sum 239) and control_d10 E:K min p_adj 1.09e-35 -> 1.30e-41."

# Interpretation — within_condition_binomial_aa

> Source: `results/stall_sites/enrichment/analysis_stats/within_condition_binomial_aa.csv`
> Family: `within_condition_binomial` (see [`_INDEX.md`](_INDEX.md))
> Test type: One-sample binomial vs bg_freq, BH-FDR per (group, site) family of ~20 aa (source: user-confirmed)

## User directives
- (invocation context) `flat-prior` → A.2.9 applied strictly: ranked cold from this CSV alone; did not import any feature from prior interpretation files; did not consult `_INDEX.md`'s cross-family hooks.
- (invocation context) Ranking criteria `(a) effect size with k≥50, (b) cross-synonym coherence at codon level, (c) per-cell reproducibility within the CSV` → applied; (b) is N/A at AA resolution and is referred to the sister codon CSV under `For Chumeng`.
- (invocation context) Symmetric reporting of shared-direction vs divergent features → applied per A.2.2.
- (invocation context) Alternative-explanation flagging for every p_adj<1e-10 row → applied per A.2.6 in the flag column of Top hits tables.
- (invocation context) Banned-words list (A.2.1) → A.2.8 self-check on this file returned zero matches.
- (readback) "Reconciled shared content from the corrected .qmd on 2026-06-02" → adopted the .qmd's three sign-partitioned cross-group Top-hits tables (concordant enrichment / depletion / discordant) under the `#sig` >= 2 reproducibility floor (13/18/21 displayed), replacing the old per-group concordance table + 6 per-group sub-tables; dropped the `stall_count` >= 200 gate from Methods and the rare-aa caveat; corrected the per-(group, site) Numbers-at-a-glance counts (now sum to 239) and control_day_10 E:K min p_adj 1.09e-35 → 1.30e-41. Olive-only sections (Biological interpretation, composite + individual plots, Overview) intentionally not imported.

## Headline
Within-group binomial against bg_freq, 6 groups (BWM and control × d0/d5/d10) × 3 sites (A/P/E) × 20 aa = 360 tests; BH-FDR per (group, site) family of 20. 239/360 hits at FDR<0.05 (264/360 at FDR<0.10). File min `p_adj` = 3.43e-132 at control_day_0 E:K (`log2_enrichment` = +0.625, `stall_count` = 3362, `total_n` = 27732). The dominant structure of this file is **cross-group concordance**: the largest-magnitude cells move in the same direction across all 6 groups, and cells with cross-group direction flips of comparable magnitude do not appear in the data — see the three "Cross-group concordance" sub-tables in Top hits. p_adj magnitudes down to 1e-132 are co-amplified by `iid-violation-binomial` (within-transcript stall correlation breaking the binomial independence assumption) and by common-aa `bg_freq` tightness (an aa with bg_freq ≈ 0.10 produces extreme p at modest log2 once n exceeds ~20000); magnitude-plus-reproducibility is the anchor for reading.

## Top hits

The effect column is `log2_enrichment` = log2(observed `stall_freq` / `bg_freq`); positive means the amino acid is over-represented at stall sites in that group, negative under-represented. Top hits are three **cross-group** sub-tables that partition the 60 (site, aa) cells by the sign of `log2_enrichment` across the 6 groups (BWM and control × d0/d5/d10): concordant enrichment (positive in all 6), concordant depletion (negative in all 6), and discordant (>= 1 sign disagreement). A cell enters the partition only if it has a value in all 6 groups. Each sub-table shows every cell that reached `p_adj` < 0.05 in at least 2 of the 6 groups — a reproducibility floor on the `#sig` axis (`#sig` = groups with FDR<0.05, of 6), **not** a fixed row cap — ranked by `#sig` descending then `min count` descending, displayed by site in A / P / E order. Ranking is deliberately not by p magnitude: `iid-violation-binomial` and common-aa `bg_freq` tightness inflate p for modest effects at large n (see `p-magnitude-anchored-ranking`).

In the `log2_enrichment` column the six values are listed as `BWM d0, d5, d10` then `ctrl d0, d5, d10`. `min count` is the smallest `observed_count` across the 6 groups. In the data there are 13 concordant-enriched, 19 concordant-depleted, and 28 discordant cells; after the `#sig` >= 2 floor the tables show 13, 18, and 21 cells respectively (all 13 concordant-enriched clear the floor; 1 concordant-depleted and 7 discordant cells significant in fewer than 2 of the 6 groups are omitted).

### Concordant enrichment: significant in >= 2 of 6 groups

| site | aa | log2_enrichment (BWM d0/d5/d10; ctrl d0/d5/d10) | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- |
| A | E | BWM +0.42, +0.32, +0.54; ctrl +0.35, +0.46, +0.40 | 554 | 6/6 | iid-amp, bg-tight |
| A | D | BWM +0.27, +0.55, +0.54; ctrl +0.31, +0.45, +0.46 | 383 | 6/6 | iid-amp, bg-tight |
| A | Y | BWM +0.79, +0.22, +0.27; ctrl +0.72, +0.25, +0.60 | 251 | 6/6 | iid-amp |
| A | K | BWM +0.18, +0.22, +0.17; ctrl +0.10, +0.31, +0.08 | 675 | 5/6 | iid-amp, bg-tight |
| A | M | BWM +0.19, +0.12, +0.36; ctrl +0.20, +0.30, +0.14 | 122 | 3/6 |  |
| P | D | BWM +0.60, +0.57, +0.76; ctrl +0.54, +0.60, +0.73 | 482 | 6/6 | iid-amp, bg-tight |
| P | N | BWM +0.44, +0.48, +0.40; ctrl +0.44, +0.46, +0.53 | 331 | 6/6 | iid-amp |
| P | F | BWM +0.21, +0.27, +0.22; ctrl +0.18, +0.34, +0.32 | 254 | 6/6 |  |
| E | K | BWM +0.73, +0.55, +0.52; ctrl +0.63, +0.76, +0.58 | 917 | 6/6 | iid-amp, bg-tight |
| E | E | BWM +0.32, +0.32, +0.38; ctrl +0.26, +0.46, +0.41 | 516 | 6/6 | iid-amp, bg-tight |
| E | D | BWM +0.23, +0.18, +0.24; ctrl +0.19, +0.24, +0.16 | 374 | 6/6 | bg-tight |
| E | N | BWM +0.46, +0.41, +0.38; ctrl +0.41, +0.44, +0.49 | 335 | 6/6 | iid-amp |
| E | M | BWM +0.25, +0.08, +0.16; ctrl +0.26, +0.27, +0.02 | 127 | 2/6 |  |

### Concordant depletion: significant in >= 2 of 6 groups

| site | aa | log2_enrichment (BWM d0/d5/d10; ctrl d0/d5/d10) | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- |
| A | A | BWM -0.69, -0.80, -0.94; ctrl -0.64, -0.77, -0.65 | 289 | 6/6 | iid-amp, bg-tight |
| A | S | BWM -0.24, -0.32, -0.37; ctrl -0.18, -0.32, -0.37 | 271 | 6/6 | bg-tight |
| A | T | BWM -0.57, -0.38, -0.51; ctrl -0.24, -0.37, -0.38 | 217 | 6/6 | bg-tight |
| A | V | BWM -0.19, -0.14, -0.22; ctrl -0.10, -0.08, -0.18 | 391 | 5/6 | bg-tight |
| A | Q | BWM -0.05, -0.11, -0.37; ctrl -0.11, -0.21, -0.52 | 203 | 4/6 |  |
| P | A | BWM -0.81, -0.80, -0.75; ctrl -0.76, -0.62, -0.68 | 294 | 6/6 | iid-amp, bg-tight |
| P | T | BWM -0.31, -0.24, -0.38; ctrl -0.09, -0.25, -0.17 | 260 | 6/6 | bg-tight |
| P | S | BWM -0.47, -0.32, -0.38; ctrl -0.29, -0.28, -0.32 | 232 | 6/6 | iid-amp, bg-tight |
| P | Q | BWM -0.40, -0.33, -0.46; ctrl -0.40, -0.51, -0.65 | 160 | 6/6 | iid-amp |
| P | L | BWM -0.18, -0.09, -0.22; ctrl -0.18, -0.18, -0.41 | 411 | 5/6 | iid-amp, bg-tight |
| P | W | BWM -0.46, -0.46, -0.78; ctrl -0.42, -0.72, -0.96 | 36 | 5/6 | rare-aa (all but ctrl d0) |
| E | A | BWM -0.85, -0.77, -0.74; ctrl -0.81, -1.00, -0.76 | 286 | 6/6 | iid-amp, bg-tight |
| E | T | BWM -0.38, -0.23, -0.33; ctrl -0.13, -0.48, -0.22 | 248 | 6/6 | iid-amp, bg-tight |
| E | S | BWM -0.58, -0.18, -0.31; ctrl -0.31, -0.25, -0.26 | 215 | 6/6 | iid-amp, bg-tight |
| E | C | BWM -1.13, -0.93, -0.50; ctrl -0.96, -0.88, -0.81 | 34 | 6/6 | iid-amp, rare-aa (all but ctrl d0) |
| E | L | BWM -0.24, -0.12, -0.11; ctrl -0.23, -0.25, -0.13 | 392 | 5/6 | iid-amp, bg-tight |
| E | Y | BWM -0.42, -0.19, -0.38; ctrl -0.10, -0.43, -0.46 | 128 | 5/6 |  |
| E | F | BWM -0.33, -0.09, -0.13; ctrl -0.16, -0.27, -0.19 | 174 | 4/6 |  |

### Discordant: significant in >= 2 of 6 groups

| site | aa | log2_enrichment (BWM d0/d5/d10; ctrl d0/d5/d10) | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- |
| A | N | BWM -0.86, -0.16, -0.19; ctrl -0.22, +0.06, -0.40 | 134 | 5/6 | iid-amp |
| A | G | BWM +0.18, +0.04, +0.40; ctrl -0.17, +0.03, +0.25 | 536 | 4/6 | iid-amp, bg-tight |
| A | I | BWM -0.08, +0.15, +0.18; ctrl +0.06, +0.30, +0.17 | 326 | 4/6 | bg-tight |
| A | R | BWM +0.10, +0.14, +0.18; ctrl +0.05, -0.02, +0.18 | 442 | 3/6 | bg-tight |
| A | L | BWM -0.02, -0.07, -0.42; ctrl +0.04, -0.32, -0.28 | 390 | 3/6 | bg-tight |
| A | P | BWM +0.15, -0.01, -0.11; ctrl -0.09, -0.21, +0.14 | 263 | 2/6 |  |
| A | W | BWM +0.77, +0.02, +0.33; ctrl +0.60, -0.04, +0.16 | 78 | 2/6 | iid-amp, rare-aa (BWM d0, BWM d10, ctrl d5, ctrl d10) |
| A | C | BWM +0.01, -0.29, -0.23; ctrl -0.05, -0.31, +0.19 | 75 | 2/6 | rare-aa (BWM d0, BWM d10) |
| P | E | BWM -0.01, +0.10, +0.36; ctrl +0.08, +0.20, +0.38 | 412 | 5/6 | iid-amp, bg-tight |
| P | G | BWM +0.24, +0.10, +0.44; ctrl -0.03, +0.13, +0.39 | 559 | 4/6 | iid-amp, bg-tight |
| P | I | BWM +0.17, +0.21, +0.08; ctrl +0.20, +0.27, -0.10 | 387 | 4/6 | bg-tight |
| P | K | BWM +0.17, +0.06, -0.10; ctrl +0.16, -0.10, -0.25 | 595 | 3/6 | bg-tight |
| P | Y | BWM +0.16, +0.17, -0.00; ctrl +0.33, +0.05, +0.29 | 192 | 3/6 | iid-amp |
| P | V | BWM -0.13, -0.10, -0.02; ctrl -0.09, +0.08, -0.20 | 409 | 2/6 | bg-tight |
| P | M | BWM +0.23, -0.05, -0.08; ctrl +0.14, -0.17, -0.38 | 114 | 2/6 |  |
| P | C | BWM -0.22, -0.05, +0.01; ctrl -0.20, -0.09, +0.28 | 64 | 2/6 | rare-aa (BWM d0, BWM d10) |
| E | R | BWM +0.07, -0.16, -0.06; ctrl -0.08, -0.10, -0.29 | 431 | 3/6 | bg-tight |
| E | V | BWM -0.14, -0.10, -0.14; ctrl +0.02, -0.19, -0.15 | 407 | 3/6 | bg-tight |
| E | Q | BWM -0.13, +0.14, +0.11; ctrl +0.06, +0.21, +0.17 | 192 | 3/6 |  |
| E | G | BWM -0.07, -0.20, +0.14; ctrl -0.26, +0.01, -0.03 | 450 | 2/6 | iid-amp, bg-tight |
| E | I | BWM +0.07, +0.18, -0.14; ctrl +0.17, +0.01, +0.11 | 361 | 2/6 | bg-tight |

## Numbers at a glance
- `n_tests`: 360 (6 groups × 3 sites × 20 aa)
- `n_significant` (adjusted-p < 0.05): 239
- `n_significant` (adjusted-p < 0.10): 264
- `min adjusted-p`: 3.43e-132 (control_day_0, E site, K; log2=+0.625, k=3362/27732) — under `iid-violation-binomial` and `bg-tight`, the alternative explanation column for this row carries both flags; this is **not** a magnitude claim.
- `p_floor`: n/a — no exact-test floor for the binomial at these n.
- Per-(group, site) BH families (each is 20 aa tests; sites in A/P/E order; counts sum to 239):
  - BWM_day_0: A 12/20, P 12/20, E 12/20 hits at p_adj<0.05; min p_adj per site 1.54e-20 (A:A), 7.63e-27 (P:A), 8.70e-53 (E:K).
  - BWM_day_5: A 12/20, P 11/20, E 15/20; min 1.43e-48 (A:A), 2.42e-49 (P:A), 7.69e-48 (E:K).
  - BWM_day_10: A 15/20, P 12/20, E 9/20; min 3.26e-36 (A:A), 2.65e-32 (P:D), 1.09e-26 (E:K).
  - control_day_0: A 14/20, P 17/20, E 14/20; min 3.98e-84 (A:A), 2.50e-111 (P:A), 3.43e-132 (E:K).
  - control_day_5: A 13/20, P 12/20, E 15/20; min 1.69e-41 (A:A), 2.69e-33 (P:D), 4.82e-90 (E:K).
  - control_day_10: A 14/20, P 16/20, E 14/20; min 1.36e-27 (A:A), 6.53e-43 (P:D), 1.30e-41 (E:K).

## Methods
Dylan parsed the test from filename + columns + the C.4.1 invocation context as a one-sample binomial vs bg_freq (`stall_count` ~ Binomial(`total_n`, `bg_freq`), two-sided), with BH-FDR computed within each (group, site) family of 20 aa tests; the user previously confirmed this test type in the original triage (carried into this redo as `test_type_source: user-confirmed`). Effect column is `log2_enrichment` = log2(`stall_freq` / `bg_freq`); a count-weighted variant `weighted_log2_enrichment` is also present in the CSV but is not used for ranking here (the unweighted log2 is what the binomial p reflects). The test answers "is amino acid X observed at stall sites at a different frequency than its `bg_freq` in the same group's transcriptome window?", *not* "is amino acid X enriched at BWM stall sites relative to control" (that is the per-timepoint Fisher) and *not* "does the within-condition replicate-level frequency change between timepoints" (that is between-timepoint Wilcoxon). Top hits are presented as three cross-group tables that partition the 60 (site, aa) cells by the sign of `log2_enrichment` across the 6 groups: concordant enrichment (positive in all 6; 13 cells), concordant depletion (negative in all 6; 19 cells), and discordant (>= 1 sign disagreement; 28 cells). A cell enters the partition only if it has a value in all 6 groups. Each table shows every cell that reached `p_adj` < 0.05 in at least 2 of the 6 groups (`#sig` >= 2, a reproducibility floor on the `#sig` axis rather than a fixed row cap), ranked by `#sig` descending then `min count` (smallest `observed_count` across the 6 groups) descending, displayed A / P / E; the three tables show 13, 18, and 21 cells respectively. The cross-group view is the lead Top-hits tabulation because the file's dominant structure is reproducibility across groups, and ranking by p magnitude is deliberately avoided since `iid-violation-binomial` and common-aa `bg_freq` tightness inflate p for modest effects at large n.

## Caveats
### Confirmed
- **pseudorep** (family-wide) — replicates pooled before the binomial; `total_n` is a pooled stall-site count rather than a sum-of-independent-replicates, so the binomial null treats correlated draws as independent.
- **iid-violation-binomial** (family-wide) — stall events within a transcript are not independent (one stall site biases neighbouring positions and may bias whole-transcript codon usage); the binomial null assumes iid draws from `bg_freq`. The dominant practical effect is to compress p-values toward zero for every cell with a non-trivial `log2_enrichment`, including the many extreme rows reported above.
- **bg-pseudocount-1e-6** (family-wide) — the upstream pipeline floors `bg_freq` at 1e-6 to avoid log2 division-by-zero. Does not affect any of the cells in Top hits (all have bg_freq > 0.008).
- **bh-per-(group, site)** (family-wide) — multiple-testing correction is per (group, site), so site-level FDR control is per 20-aa family, not pooled across the 60 aa per group.
- **rare-aa-low-count** (per-CSV) — cells with `observed_count` below ~100 in some groups (lowest: BWM_d10 P:W k=36; BWM_d0 E:C k=34) have noisier log2 estimates and carry the `rare-aa` flag where displayed. Under the `#sig` >= 2 floor both clear the threshold and appear in the concordant-depletion table: E:C (min count 34, #sig 6/6, |log2_enrichment| up to 1.13 in BWM_day_0) and P:W (min count 36, #sig 5/6, |log2_enrichment| up to ~0.96 in control_day_10); the discordant table likewise carries rare-aa A:W, A:C, and P:C. The tables sort by `#sig` then `min count`, so these low-count cells sit toward the bottom within their site block. Read direction, not magnitude, for these rows; do not rank them alongside high-count cells.
- **small-bh-family-discreteness** (per-CSV) — per (group, site) BH families are only 20 tests each. With the iid-violation-driven extreme p magnitudes here, the discreteness almost never binds (FDR<0.05 hit fractions per family run 9/20 to 17/20), but flag for parity with the sister codon CSV's larger 61-codon BH families and so Chumeng can compare the two file's hit-rate inflation symmetrically.
- **p-magnitude-anchored-ranking** (per-CSV) — p_adj down to 3.4e-132 in this file is best read as a joint readout of (`log2_enrichment` magnitude) × (`total_n`) × (`iid-violation-binomial` amplification factor), not as a pure biological strength signal. The Top hits flag column applies `iid-amp` to every p_adj<1e-10 row in this file (A.2.4 symmetry: no selective dampening — every extreme-p row is flagged regardless of direction), and adds `bg-tight` for rows where `bg_freq` > 0.05 (which makes the binomial null especially tight at high N). Reading priority: log2_enrichment magnitude + stall_count + cross-group reproducibility, in that order.

### Considered but not applicable
*(none for this redo — the original triage's caveats were preserved or rewritten as above; no new caveats were proposed and denied during the C.4.1 re-run.)*

## For Chumeng (joint-reading hooks)
- Family: `within_condition_binomial` — sister CSV in this family that should be reconciled: `within_condition_binomial_codon.csv` (codon resolution; same design, same family-wide caveats; will be redone independently in C.4.2 under flat-prior).
- Open questions Chumeng should resolve at synthesis time, framed as falsifiers per A.2.7:
  - The 13 concordantly enriched and 19 concordantly depleted (site, aa) cells (of the 60 that have a value in all 6 groups; the strongest examples are P:D and E:K enriched, A:A / P:A / E:A depleted) are this CSV's largest-magnitude reproducible signals. **Does each of them re-appear at the codon level in `within_condition_binomial_codon.csv` with consistent direction across all 6 groups, or does the aa-level signal split unevenly across synonyms (suggesting a single codon, not the aa, drives the effect)?** A clean cross-synonym split at one or two cells would change the biological reading from "aa property" to "codon-level decoding effect"; spread across many synonyms would point toward an aa-level effect.
  - **Do the same enriched and depleted cells reappear with consistent direction in `per_timepoint_fisher_aa.csv` (BWM-vs-control contrasts at each timepoint)?** The within-condition binomial is by construction blind to BWM-vs-control divergence (each group is tested against its own bg_freq independently); per-timepoint Fisher is the natural design for catching that. If a binomial concordance cell is also a Fisher-significant BWM-vs-control hit, the aa-property reading needs a perturbation overlay; if the binomial cell is concordant *and* the Fisher cell is null at that (timepoint, site, aa), the aa property is the simpler reading.
  - The cross-group concordance picture in this CSV (no comparable-magnitude divergent cells) is **a constraint** on what BWM-vs-control contrast tests should show. **Does the per-timepoint Fisher file flag any high-magnitude (OR≫1 or ≪1) features that this binomial file shows as concordant across all 6 groups?** A high-OR Fisher hit at a binomial-concordant cell is a red flag for at least one of: imbalanced-N artefact in Fisher, a count-imbalance in one group not picked up by the within-group binomial, or a real perturbation effect on top of stable absolute enrichment.
  - File-min p_adj cell (control_d0 E:K, p_adj=3.43e-132). **Does this cell's magnitude re-appear at comparable magnitude in `per_timepoint_fisher_aa.csv` at day_0?** If the Fisher OR at (day_0, E, K) is near 1.0 (BWM and control both enrich K at E to similar degrees), the binomial p magnitude is a stable-frequency-with-tight-bg readout, not a perturbation signal. If the Fisher OR diverges from 1, the binomial p is masking a perturbation effect that the within-group design cannot see.
  - **Does `between_condition_wilcoxon_aa.csv` show any of the binomial-concordance cells as differentially distributed at the per-replicate frequency level?** That file's already-noted firm null at FDR<0.05 (0/60) means: every concordance cell in this file is consistent with stable per-replicate frequencies in both BWM and control — i.e., the binomial p magnitudes here are not driven by per-replicate frequency divergence. Confirm or contradict by direction at the closest-to-significant Wilcoxon cells.
