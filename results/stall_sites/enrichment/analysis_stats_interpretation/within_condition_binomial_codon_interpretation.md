---
input_csv: results/stall_sites/enrichment/analysis_stats/within_condition_binomial_codon.csv
family: within_condition_binomial
test_type: One-sample binomial test (k=stall_count out of n=total_n vs H0: p=bg_freq, two-sided), BH-FDR within each (group, site) family of 61 sense codons
test_type_source: user-confirmed
synced_from_olive_qmd: 2026-06-02
n_tests: 1098
n_significant_fdr05: 491
n_significant_fdr10: 552
min_p_adj: 3.5335639633970407e-97
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "iid-violation-binomial", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "bg-pseudocount-1e-6", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "bh-per-(group,site)", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "Each per-(group,site) BH family is 61 codons (~3x larger than the AA file's 20 per family). AA-level signals can distribute across synonyms and fall below the per-(group,site) FDR threshold here; the codon-vs-AA power tradeoff is specificity gained, per-BH-family power lost."}
  - {label: "rare-codon-low-count-instability", proposed_by: dylan, status: confirmed, why: "Rare codons (ATA, TTA, AGG, GGG, CGG, ACG, TCG and similar) have unstable log2_enrichment because the binomial variance at small k is dominated by counting noise. The Top-hits tables carry a `rare-codon` flag on any cell where at least one of the 6 groups has observed_count < 100; within each #sig tier the cells sort by min count descending so the weakest-supported fall to the bottom of their tier. Read flagged rows by direction, not magnitude."}
  - {label: "p-magnitude-anchored-ranking", proposed_by: dylan, status: confirmed, why: "min p_adj = 3.53e-97 (control_day_0, P, GAT) reflects iid violation amplifying a moderate per-site enrichment (log2 +0.828) at large stall count (k=1558 / n=27732). 110 of 1098 rows sit at p_adj < 1e-10. Rank by log2_enrichment magnitude + min count + cross-group reproducibility, never by p magnitude alone; the cross-group tables rank by #sig then min count for exactly this reason."}
caveats_considered: []
headline: "Codon-level within-group binomial vs bg_freq returns 491/1098 hits at FDR<0.05 (44.7%; 552/1098 at FDR<0.10), file min p_adj 3.53e-97 at control_day_0 P:GAT (log2_enrichment +0.828, k=1558/27732); of the 183 (site, codon) cells 50 are concordantly enriched and 45 concordantly depleted in a single sign across all 6 groups with 88 discordant, and the largest-magnitude cells (by mean log2 across the 6 groups) hold one sign across all 6: P:TCC -1.04, P:GCC -1.02, P:GAT +0.98, E:GCC -0.94, A:GCC -0.90, P:AAT +0.88, E:GCT -0.86, A:GCT -0.82, E:AAA +0.78, P:GGT +0.74."
user_directives:
  - "(invocation context) `flat-prior` token — apply A.2.1 through A.2.9 strictly; do not import findings from prior CSV interpretations as priors; read this CSV cold without consulting `_INDEX.md` cross-family hooks; rank features by (a) effect size in high-count rows, (b) cross-synonym coherence at codon level, (c) reproducibility within this CSV's per-cell neighbours; report shared-direction features at equal billing with divergent features; for any p_adj < 1e-10 row name at least one alternative explanation."
  - "(per-CSV triage, carried over from original triage; not re-litigated) Test type confirmation → `Confirm: one-sample binomial vs bg_freq` (k=stall_count out of n=total_n, H0: p=bg_freq, two-sided, BH-FDR within each (group, site) family of 61 sense codons)."
  - "(per-CSV triage, carried over) Codon-specific caveats beyond family-wide → confirmed `larger-bh-family`, `rare-codon-low-count-instability`, `p-magnitude-anchored-ranking`."
  - "(per-CSV triage, carried over) Framing firmness → `Mixed`. Hits with consistent effect across groups + corroboration from the per-timepoint Fisher and between-condition Wilcoxon files are firm; binomial-only hits with extreme p but no cross-test corroboration are exploratory because of the iid violation."
  - "(per-CSV triage, carried over) Spotlight → `No spotlight`. The prior invocation's AAG-at-E spotlight directive is revoked; rank by data alone per A.2.3."
  - "(readback) \"Reconciled shared content from the corrected .qmd on 2026-06-02\" → \"Adopted the .qmd's three sign-partitioned cross-group Top-hits tables (concordant enrichment / concordant depletion / discordant; #sig >= 2 reproducibility floor; columns Site | codon | aa | log2_enrichment | min count | #sig | flag); replaced the prior k>=50 concordance/direction-flip/per-group layout; reconciled Methods + Numbers + rare-codon caveat (threshold 100, no k>=50 gate); refreshed two stale For-Chumeng numbers (discordant partition replaces '7 sign-flip cells'; A:GGA positive in 5 of 6 groups, not 4). Every number enumerated and verified; 0 shared-value corrections.\""
---

# Interpretation — within_condition_binomial_codon

> Source: `results/stall_sites/enrichment/analysis_stats/within_condition_binomial_codon.csv`
> Family: `within_condition_binomial` (see [`_INDEX.md`](_INDEX.md))
> Test type: One-sample binomial test, BH-FDR within each (group, site) family of 61 sense codons (source: user-confirmed)

## User directives
- (invocation context) `flat-prior` — apply A.2.1 through A.2.9 strictly; read CSV cold; rank by effect size in high-count rows, cross-synonym coherence, and per-cell reproducibility; equal billing for shared-direction vs divergent; alternative-explanation flags on every `p_adj < 1e-10` row.
- (per-CSV triage, carried over from original triage; not re-litigated) "Test type confirmation" → "Confirm: one-sample binomial vs bg_freq" with H0 `stall_freq = bg_freq`, two-sided, BH-FDR per (group, site).
- (per-CSV triage, carried over) "Codon-specific caveats beyond family-wide" → confirmed `larger-bh-family`, `rare-codon-low-count-instability`, `p-magnitude-anchored-ranking`.
- (per-CSV triage, carried over) "Framing firmness" → Mixed. Firm = consistent across groups + cross-test corroboration; binomial-only extreme-p hits = exploratory because of the iid violation.
- (per-CSV triage, carried over) "Spotlight" → none. Prior invocation's AAG-at-E spotlight directive is revoked; data-ranked only per A.2.3.

## Headline
Across 1098 tests (6 groups x 3 sites x 61 sense codons) the file returns 491 hits at `p_adj`<0.05 (44.7%) and 552 at `p_adj`<0.10 (50.3%). File-min `p_adj` is at control_day_0 P:GAT (`log2_enrichment` +0.828, k=1558 / n=27732, `p_adj` = 3.53e-97). Of the 183 (site, codon) cells, 95 hold a single sign across all six groups — 50 concordantly enriched and 45 concordantly depleted — and 88 show at least one cross-group sign disagreement; the discordant cells sit at low magnitude or rest on rare-codon counts. The largest-magnitude shared-direction cells (by mean `log2_enrichment` across the 6 groups) are: P:TCC -1.04 (Ser), P:GCC -1.02 (Ala), P:GAT +0.98 (Asp), E:GCC -0.94 (Ala), A:GCC -0.90 (Ala), P:AAT +0.88 (Asn), E:GCT -0.86 (Ala), A:GCT -0.82 (Ala), E:AAA +0.78 (Lys), P:GGT +0.74 (Gly). Read magnitude + cross-group reproducibility as the primary signal; the `p_adj` distribution from 3.53e-97 to ~1.0 is shape-driven by iid-violation amplification at high k and is not a faithful priority signal across cells.

## Top hits

The effect column is `log2_enrichment` = log2(observed `stall_freq` / `bg_freq`); positive means the codon is over-represented at stall sites in that group, negative means under-represented. The three cross-group tables below partition the 183 (site, codon) cells by the sign of `log2_enrichment` across the 6 groups: concordant enrichment (positive in all 6), concordant depletion (negative in all 6), and discordant (at least one sign disagreement). Each table shows every cell that reached `p_adj` < 0.05 in at least 2 of the 6 groups — a reproducibility floor on the `#sig` axis (groups with FDR<0.05, of 6), not a fixed row cap — ranked by `#sig` descending then `min count` (smallest `observed_count` across the 6 groups) descending, never by p magnitude (`iid-violation-binomial` and common-codon `bg_freq` tightness inflate p for modest effects at large n). Rows are displayed by site in A / P / E order. In the `log2_enrichment` column the six per-group values are listed on two lines: BWM day_0, day_5, day_10 first, then control day_0, day_5, day_10. The `aa` column is the single-letter amino-acid translation of each codon (no parenthetical expansion); the codon cell stays bare. Of the 183 cells the partition is 50 / 45 / 88; after the `#sig` >= 2 floor the tables show 44 / 37 / 25 cells respectively.

### Concordant enrichment — significant in >= 2 of 6 groups

| Site | codon | aa | log2_enrichment | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- | --- |
| A | GAG | E | +0.38, +0.31, +0.58 <br> +0.31, +0.45, +0.43 | 381 | 6/6 | iid-amp |
| A | GAT | D | +0.31, +0.86, +0.78 <br> +0.43, +0.71, +0.67 | 184 | 6/6 | iid-amp |
| A | GAA | E | +0.50, +0.33, +0.44 <br> +0.41, +0.49, +0.36 | 173 | 6/6 | iid-amp |
| A | TTG | L | +0.63, +0.56, +0.38 <br> +0.73, +0.40, +0.58 | 96 | 6/6 | iid-amp, rare-codon (BWM d10) |
| A | ATA | I | +1.44, +1.43, +1.28 <br> +1.22, +1.14, +1.20 | 7 | 6/6 | rare-codon (all) |
| A | AAG | K | +0.21, +0.22, +0.20 <br> +0.14, +0.31, +0.09 | 618 | 5/6 | iid-amp, bg-tight |
| A | AGA | R | +0.48, +0.41, +0.31 <br> +0.33, +0.32, +0.22 | 155 | 5/6 |  |
| A | ATT | I | +0.27, +0.48, +0.67 <br> +0.28, +0.49, +0.44 | 88 | 5/6 | rare-codon (BWM d0) |
| A | CAT | H | +0.74, +0.53, +0.46 <br> +0.43, +0.52, +0.31 | 59 | 5/6 | rare-codon (BWM d0, BWM d10, ctrl d10) |
| A | TAT | Y | +0.58, +0.85, +0.61 <br> +0.93, +0.76, +0.89 | 36 | 5/6 | iid-amp, rare-codon (BWM d0, BWM d10, ctrl d5, ctrl d10) |
| A | GAC | D | +0.23, +0.20, +0.29 <br> +0.15, +0.17, +0.25 | 199 | 4/6 |  |
| A | TAC | Y | +0.82, +0.06, +0.21 <br> +0.63, +0.13, +0.54 | 208 | 3/6 | iid-amp |
| A | ATG | M | +0.19, +0.12, +0.36 <br> +0.20, +0.30, +0.14 | 122 | 3/6 |  |
| A | AGG | R | +1.34, +0.93, +0.55 <br> +0.70, +0.61, +1.25 | 5 | 3/6 | rare-codon (all) |
| A | ACG | T | +1.08, +0.35, +0.59 <br> +0.35, +0.10, +0.20 | 14 | 2/6 | rare-codon (all but ctrl d0) |
| A | CCC | P | +1.14, +0.79, +0.94 <br> +0.34, +0.31, +0.88 | 12 | 2/6 | rare-codon (all) |
| A | TTA | L | +2.06, +0.89, +0.81 <br> +1.09, +0.32, +0.83 | 5 | 2/6 | rare-codon (all) |
| P | GAT | D | +0.88, +0.96, +1.13 <br> +0.83, +1.00, +1.11 | 272 | 6/6 | iid-amp |
| P | AAC | N | +0.30, +0.32, +0.27 <br> +0.21, +0.37, +0.43 | 257 | 6/6 |  |
| P | TTC | F | +0.22, +0.17, +0.20 <br> +0.12, +0.27, +0.26 | 234 | 6/6 |  |
| P | GGT | G | +0.83, +0.66, +1.02 <br> +0.48, +0.73, +0.72 | 81 | 6/6 | rare-codon (BWM d0, BWM d10) |
| P | AAA | K | +0.44, +0.56, +0.68 <br> +0.52, +0.55, +0.42 | 80 | 6/6 | iid-amp, rare-codon (BWM d0) |
| P | AAT | N | +1.04, +0.94, +0.87 <br> +0.84, +0.73, +0.83 | 74 | 6/6 | iid-amp, rare-codon (BWM d0) |
| P | CAT | H | +0.51, +0.80, +0.46 <br> +0.67, +0.48, +0.52 | 52 | 6/6 | iid-amp, rare-codon (BWM d0, BWM d10, ctrl d5, ctrl d10) |
| P | TAT | Y | +0.65, +0.85, +0.64 <br> +0.90, +0.62, +1.12 | 38 | 6/6 | iid-amp, rare-codon (BWM d0, BWM d10, ctrl d5, ctrl d10) |
| P | GTT | V | +0.42, +0.26, +0.21 <br> +0.18, +0.45, +0.22 | 203 | 5/6 |  |
| P | ATT | I | +0.55, +0.58, +0.45 <br> +0.51, +0.52, +0.10 | 107 | 5/6 | iid-amp |
| P | GAG | E | +0.00, +0.15, +0.45 <br> +0.06, +0.25, +0.50 | 293 | 4/6 | iid-amp |
| P | TTG | L | +0.40, +0.42, +0.23 <br> +0.24, +0.39, +0.22 | 86 | 4/6 | rare-codon (BWM d0, BWM d10) |
| P | TTT | F | +0.14, +0.93, +0.39 <br> +0.45, +0.78, +0.85 | 20 | 4/6 | rare-codon (all but ctrl d0) |
| P | GAC | D | +0.30, +0.11, +0.32 <br> +0.11, +0.12, +0.30 | 210 | 3/6 |  |
| P | ATA | I | +1.44, +1.43, +1.57 <br> +0.40, +0.76, +1.03 | 7 | 3/6 | rare-codon (all) |
| P | CGT | R | +0.28, +0.24, +0.20 <br> +0.11, +0.07, +0.17 | 200 | 2/6 |  |
| P | AGT | S | +0.48, +0.41, +0.26 <br> +0.46, +0.28, +0.92 | 12 | 2/6 | rare-codon (all but ctrl d0) |
| E | AAG | K | +0.69, +0.52, +0.52 <br> +0.57, +0.75, +0.55 | 812 | 6/6 | iid-amp, bg-tight |
| E | GAG | E | +0.30, +0.27, +0.42 <br> +0.29, +0.56, +0.40 | 360 | 6/6 | iid-amp |
| E | AAC | N | +0.45, +0.40, +0.35 <br> +0.41, +0.40, +0.49 | 285 | 6/6 | iid-amp |
| E | AGA | R | +0.62, +0.33, +0.34 <br> +0.55, +0.50, +0.31 | 164 | 6/6 | iid-amp |
| E | GAA | E | +0.35, +0.42, +0.29 <br> +0.23, +0.26, +0.43 | 156 | 6/6 |  |
| E | AAA | K | +0.99, +0.78, +0.57 <br> +0.81, +0.79, +0.77 | 105 | 6/6 | iid-amp |
| E | AAT | N | +0.48, +0.47, +0.48 <br> +0.40, +0.54, +0.50 | 50 | 5/6 | rare-codon (BWM d0, BWM d10) |
| E | GAT | D | +0.24, +0.24, +0.27 <br> +0.27, +0.38, +0.17 | 175 | 4/6 |  |
| E | ATT | I | +0.19, +0.36, +0.28 <br> +0.22, +0.30, +0.11 | 83 | 3/6 | rare-codon (BWM d0) |
| E | ATG | M | +0.25, +0.08, +0.16 <br> +0.26, +0.27, +0.02 | 127 | 2/6 |  |

### Concordant depletion — significant in >= 2 of 6 groups

| Site | codon | aa | log2_enrichment | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- | --- |
| A | GTC | V | -0.34, -0.25, -0.29 <br> -0.29, -0.18, -0.38 | 194 | 6/6 |  |
| A | CTC | L | -0.43, -0.48, -0.81 <br> -0.43, -0.78, -0.81 | 139 | 6/6 | iid-amp |
| A | GCC | A | -0.76, -1.05, -1.09 <br> -0.85, -0.95, -0.70 | 126 | 6/6 | iid-amp |
| A | GCT | A | -0.89, -0.79, -0.84 <br> -0.88, -0.73, -0.77 | 122 | 6/6 | iid-amp |
| A | ACC | T | -0.97, -0.77, -0.81 <br> -0.45, -0.69, -0.58 | 107 | 6/6 | iid-amp |
| A | TCC | S | -0.56, -0.55, -0.52 <br> -0.31, -0.66, -0.55 | 97 | 6/6 | rare-codon (BWM d0) |
| A | TCT | S | -0.57, -0.32, -0.59 <br> -0.28, -0.30, -0.51 | 50 | 6/6 | rare-codon (BWM d0, BWM d10, ctrl d10) |
| A | AAC | N | -1.06, -0.36, -0.36 <br> -0.48, -0.11, -0.63 | 100 | 5/6 | iid-amp |
| A | CAC | H | -0.26, -0.40, -0.45 <br> -0.26, -0.57, -0.62 | 83 | 5/6 | rare-codon (BWM d0, BWM d10, ctrl d10) |
| A | CAG | Q | -0.28, -0.36, -0.39 <br> -0.30, -0.48, -0.43 | 71 | 5/6 | rare-codon (BWM d0, BWM d10, ctrl d10) |
| A | CTT | L | -0.01, -0.02, -0.46 <br> -0.15, -0.28, -0.27 | 131 | 4/6 |  |
| A | ACT | T | -0.30, -0.13, -0.29 <br> -0.35, -0.20, -0.33 | 70 | 2/6 | rare-codon (BWM d0, BWM d10) |
| P | GCT | A | -0.72, -0.59, -0.72 <br> -0.70, -0.47, -0.47 | 137 | 6/6 | iid-amp |
| P | ACC | T | -0.66, -0.65, -0.62 <br> -0.31, -0.60, -0.38 | 132 | 6/6 | iid-amp |
| P | CTC | L | -0.82, -0.64, -0.63 <br> -0.70, -0.68, -0.89 | 128 | 6/6 | iid-amp |
| P | GCC | A | -1.16, -1.10, -0.92 <br> -1.06, -0.85, -1.00 | 113 | 6/6 | iid-amp |
| P | CGC | R | -0.36, -0.53, -0.46 <br> -0.69, -0.53, -0.46 | 95 | 6/6 | iid-amp, rare-codon (BWM d0) |
| P | TCC | S | -1.28, -1.02, -1.03 <br> -1.04, -0.89, -0.97 | 59 | 6/6 | iid-amp, rare-codon (BWM d0, BWM d10, ctrl d10) |
| P | CAG | Q | -0.78, -0.64, -0.51 <br> -0.65, -0.80, -0.73 | 50 | 6/6 | iid-amp, rare-codon (BWM d0, BWM d10, ctrl d5, ctrl d10) |
| P | CCA | P | -0.29, -0.19, -0.33 <br> -0.16, -0.19, -0.07 | 176 | 5/6 |  |
| P | GTC | V | -0.73, -0.45, -0.21 <br> -0.45, -0.25, -0.74 | 148 | 5/6 | iid-amp |
| P | CAC | H | -0.31, -0.50, -0.49 <br> -0.21, -0.53, -0.53 | 81 | 5/6 | rare-codon (BWM d0, BWM d10, ctrl d10) |
| P | TGG | W | -0.46, -0.46, -0.78 <br> -0.42, -0.72, -0.96 | 36 | 5/6 | rare-codon (all but ctrl d0) |
| P | CAA | Q | -0.18, -0.17, -0.43 <br> -0.28, -0.38, -0.60 | 110 | 4/6 |  |
| P | TCT | S | -0.31, -0.07, -0.24 <br> -0.22, -0.13, -0.33 | 60 | 2/6 | rare-codon (BWM d0, BWM d10, ctrl d10) |
| E | ACC | T | -0.47, -0.63, -0.59 <br> -0.19, -0.74, -0.31 | 151 | 6/6 | iid-amp |
| E | GCC | A | -0.92, -0.94, -0.89 <br> -0.90, -1.19, -0.80 | 134 | 6/6 | iid-amp |
| E | GCT | A | -0.94, -0.72, -0.78 <br> -0.94, -0.98, -0.79 | 118 | 6/6 | iid-amp |
| E | TAC | Y | -0.52, -0.27, -0.48 <br> -0.25, -0.60, -0.51 | 103 | 6/6 |  |
| E | TGC | C | -1.27, -0.98, -0.68 <br> -1.06, -1.00, -0.89 | 25 | 6/6 | iid-amp, rare-codon (all but ctrl d0) |
| E | CGC | R | -0.39, -0.49, -0.25 <br> -0.69, -0.55, -0.62 | 93 | 5/6 | iid-amp, rare-codon (BWM d0, ctrl d10) |
| E | TCC | S | -0.95, -0.36, -0.58 <br> -0.60, -0.58, -0.25 | 74 | 5/6 | iid-amp, rare-codon (BWM d0) |
| E | CTC | L | -0.35, -0.34, -0.22 <br> -0.46, -0.53, -0.13 | 178 | 4/6 | iid-amp |
| E | TTC | F | -0.34, -0.11, -0.15 <br> -0.16, -0.32, -0.22 | 159 | 4/6 |  |
| E | CGT | R | -0.13, -0.35, -0.25 <br> -0.36, -0.31, -0.66 | 143 | 4/6 |  |
| E | TCT | S | -0.73, -0.16, -0.28 <br> -0.52, -0.35, -0.56 | 45 | 4/6 | rare-codon (BWM d0, BWM d10, ctrl d10) |
| E | TGT | C | -0.65, -0.77, -0.06 <br> -0.75, -0.57, -0.56 | 9 | 2/6 | rare-codon (all) |

### Discordant — significant in >= 2 of 6 groups

| Site | codon | aa | log2_enrichment | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- | --- |
| A | GGA | G | +0.18, +0.07, +0.45 <br> -0.14, +0.13, +0.36 | 455 | 5/6 | iid-amp, bg-tight |
| A | ATC | I | -0.22, -0.01, -0.01 <br> -0.12, +0.21, +0.06 | 231 | 3/6 |  |
| A | TGC | C | +0.17, -0.44, -0.47 <br> -0.12, -0.51, +0.20 | 53 | 3/6 | rare-codon (all but ctrl d0) |
| A | AAT | N | -0.08, +0.41, +0.37 <br> +0.22, +0.51, +0.25 | 34 | 3/6 | rare-codon (BWM d0, BWM d10, ctrl d10) |
| A | AGC | S | +0.05, -0.72, -0.60 <br> -0.47, -0.43, -0.46 | 29 | 3/6 | rare-codon (all but ctrl d0) |
| A | TTT | F | -0.18, +0.49, +0.60 <br> +0.34, +0.46, +0.28 | 16 | 3/6 | rare-codon (all but ctrl d0) |
| A | CCA | P | +0.08, -0.07, -0.13 <br> -0.12, -0.22, +0.10 | 227 | 2/6 |  |
| A | CAA | Q | +0.08, +0.02, -0.36 <br> -0.02, -0.08, -0.59 | 126 | 2/6 |  |
| A | CGC | R | -0.23, -0.09, -0.05 <br> -0.20, -0.25, +0.24 | 104 | 2/6 |  |
| A | TGG | W | +0.77, +0.02, +0.33 <br> +0.60, -0.04, +0.16 | 78 | 2/6 | iid-amp, rare-codon (BWM d0, BWM d10, ctrl d5, ctrl d10) |
| A | GGT | G | +0.20, -0.02, +0.13 <br> -0.41, -0.16, -0.54 | 46 | 2/6 | rare-codon (all but ctrl d0) |
| A | TCG | S | +0.69, +0.12, +0.19 <br> +0.20, -0.03, +0.07 | 44 | 2/6 | rare-codon (all but ctrl d0) |
| A | GTG | V | +0.61, +0.16, -0.23 <br> +0.46, +0.28, +0.22 | 35 | 2/6 | rare-codon (all but ctrl d0) |
| P | AAG | K | +0.14, -0.03, -0.23 <br> +0.04, -0.23, -0.37 | 482 | 4/6 | bg-tight |
| P | GGA | G | +0.19, +0.03, +0.39 <br> -0.09, +0.07, +0.38 | 458 | 4/6 | iid-amp, bg-tight |
| P | AGA | R | +0.44, +0.29, -0.02 <br> +0.26, +0.13, +0.12 | 128 | 3/6 |  |
| P | CTG | L | -0.20, -0.46, +0.03 <br> -0.46, -0.84, -0.71 | 16 | 3/6 | rare-codon (all but ctrl d0) |
| P | TGT | C | -0.36, +0.54, +0.55 <br> +0.39, +0.45, +0.83 | 11 | 3/6 | rare-codon (all but ctrl d0) |
| P | ATC | I | +0.02, +0.02, -0.08 <br> -0.01, +0.16, -0.19 | 273 | 2/6 |  |
| P | ATG | M | +0.23, -0.05, -0.08 <br> +0.14, -0.17, -0.38 | 114 | 2/6 |  |
| P | GCA | A | +0.09, -0.62, -0.19 <br> -0.24, -0.41, -0.25 | 28 | 2/6 | rare-codon (all but ctrl d0) |
| E | GTT | V | -0.04, -0.20, -0.27 <br> +0.03, -0.27, -0.42 | 148 | 3/6 |  |
| E | GCA | A | -0.40, -0.44, +0.05 <br> -0.36, -0.57, -0.56 | 22 | 3/6 | rare-codon (all but ctrl d0) |
| E | GGA | G | -0.09, -0.17, +0.13 <br> -0.24, +0.03, -0.01 | 377 | 2/6 | iid-amp, bg-tight |
| E | ATC | I | +0.05, +0.10, -0.28 <br> +0.14, -0.11, +0.11 | 257 | 2/6 |  |

### Flag glossary
- `iid-amp` — i.i.d.-violation amplified; applied to every row with `p_adj` < 1e-10 (110 of 1098 rows, 10.0%). The binomial treats every counted position as an independent draw, but stall positions on the same transcript move together, so even a modest enrichment compresses to an astronomically tiny p. Read the row by `log2_enrichment` + reproducibility, not by how small the p is.
- `bg-tight` — common codon (`bg_freq` > 0.05; AAG and GGA cross this threshold in Top hits). At tens of thousands of positions the binomial null is very narrow, so a small departure from `bg_freq` lands far in the tail. Same upshot as `iid-amp`: the small p reflects a sharp null at large n, not a large effect.
- `rare-codon` — at least one of the 6 groups has `observed_count` < 100; the flag names the low-count groups, collapses to `(all but <group>)` when only one group clears the threshold, and to `(all)` when none do. Direction is reliable; magnitude is not. TGG (Trp, single-codon) cannot be rescued by synonym aggregation; ATA (Ile) is a genuinely rare codon carrying `rare-codon (all)`.

## Numbers at a glance
- `n_tests`: 1098 (6 groups x 3 sites x 61 sense codons)
- `n_significant` (adjusted-p < 0.05): 491 (44.7%)
- `n_significant` (adjusted-p < 0.10): 552 (50.3%)
- `min adjusted-p`: 3.5335639633970407e-97 (control_day_0, P site, GAT; `log2_enrichment` +0.828, k=1558 / n=27732) — also a cross-group concordance cell (mean `log2_enrichment` +0.98 across the 6 groups). Under `iid-violation-binomial` this row's p magnitude is co-amplified; this is not a magnitude claim.
- `p_floor`: n/a — no exact-test floor for the binomial at these n (iid violation amplifies the lower tail instead).
- Cross-group sign partition of the 183 (site, codon) cells: 50 concordant-enriched, 45 concordant-depleted, 88 discordant (each cross-group table displays the cells significant in >= 2 of 6 groups — 44, 37, and 25 respectively).
- Rows with `p_adj` < 1e-10: 110 of 1098 (10.0%); each carries `iid-amp` in its flag column per A.2.6.
- Per (group, site) BH family of 61 codons: hits at FDR<0.05 (of 61), min p_adj, and the codon achieving it:

| group | A site | P site | E site |
| --- | --- | --- | --- |
| BWM_day_0 | 27/61; 6.14e-16 (TAC) | 28/61; 7.95e-22 (GCC) | 16/61; 8.65e-42 (AAG) |
| BWM_day_5 | 27/61; 4.86e-37 (GAT) | 30/61; 1.75e-48 (GAT) | 23/61; 1.72e-34 (AAG) |
| BWM_day_10 | 28/61; 6.67e-21 (GCC) | 26/61; 5.70e-39 (GAT) | 16/61; 1.89e-22 (AAG) |
| control_day_0 | 42/61; 6.48e-60 (GCT) | 38/61; 3.53e-97 (GAT) | 30/61; 1.50e-79 (AAG) |
| control_day_5 | 31/61; 1.33e-24 (GCC) | 28/61; 4.22e-50 (GAT) | 26/61; 2.06e-75 (AAG) |
| control_day_10 | 25/61; 2.45e-15 (CTC) | 31/61; 1.07e-52 (GAT) | 19/61; 9.17e-32 (AAG) |

## Methods
Dylan proposed a one-sample binomial test with H0 `stall_freq = bg_freq`, k=stall_count, n=total_n, two-sided, BH-FDR correction within each (group, site) family of 61 sense codons; user confirmed in the original triage (carried over to this run). Effect column is `log2_enrichment` (= log2(observed `stall_freq` / `bg_freq`)) with a 1e-6 pseudocount on `bg_freq` to avoid log(0); a count-weighted variant `weighted_log2_enrichment` is present in the CSV but is not used for ranking here. The test answers "is codon X observed at stall sites at a different frequency than its `bg_freq` in the same group's transcriptome window?", not "is codon X's enrichment different between BWM and control" (the per-timepoint Fisher) and not "is the enrichment shifting across timepoints" (the between-timepoint Wilcoxon). Counts and totals are summed across replicates before the test (hence the pseudoreplication caveat). Top hits are presented as three cross-group tables that partition the 183 (site, codon) cells by the sign of `log2_enrichment` across the 6 groups: concordant enrichment (positive in all 6; 50 cells), concordant depletion (negative in all 6; 45 cells), and discordant (>= 1 sign disagreement; 88 cells). A cell enters the partition only if it has a value in all 6 groups. Each table shows every cell that reached `p_adj` < 0.05 in at least 2 of the 6 groups (`#sig` >= 2 — a reproducibility floor on the `#sig` axis, not a fixed row cap), ranked by `#sig` descending then `min count` (smallest `observed_count` across the 6 groups) descending, displayed in A / P / E site order; the three tables show 44, 37, and 25 cells respectively. Ranking by p magnitude is deliberately avoided because `iid-violation-binomial` and common-codon `bg_freq` tightness inflate p for modest effects at large n.

## Caveats
### Confirmed
- **pseudorep** (family-wide) — replicate counts within a (group, site) cell are pooled to form k and n, so the binomial variance is computed at the position-pool level rather than the replicate level; per-replicate variation is invisible to the statistic and p-values are anti-conservative. Inherited from family `within_condition_binomial`.
- **iid-violation-binomial** (family-wide) — the binomial assumes the `total_n` counted positions are i.i.d., but stall positions cluster within transcripts and transcripts vary in coverage, so the effective n is smaller than `total_n`. This is the dominant reason p-values compress to 1e-22 and beyond (110 rows below 1e-10). Inherited from family `within_condition_binomial`.
- **bg-pseudocount-1e-6** (family-wide) — `bg_freq` is floored at 1e-6 to keep log(0) finite; for genuinely absent codons this produces extreme `log2_enrichment` when k>0. Does not affect any Top-hits row (all have `bg_freq` > 0.001). Inherited from family `within_condition_binomial`.
- **bh-per-(group,site)** (family-wide) — BH correction is applied independently within each of the 18 (group, site) families of 61 codons, not across the full 1098-test grid. Two cells with the same raw p in different (group, site) families can land at different `p_adj`; cross-group p_adj rankings are not directly commensurable. Inherited from family `within_condition_binomial`.
- **larger-bh-family** (per-CSV) — each per-(group, site) BH family is 61 codons, ~3x larger than the AA file's 20-aa families. AA-level signals can distribute across synonyms and fall below the per-(group, site) FDR threshold here; if a clear AA-level hit in the sister CSV has no matching codon-level row, the likely reading is synonym-splitting rather than absence-of-signal.
- **rare-codon-low-count-instability** (per-CSV) — rare codons (ATA, TTA, AGG, GGG, CGG, ACG, TCG and similar) have unstable `log2_enrichment` because binomial variance at small k is dominated by counting noise. The cross-group tables carry a `rare-codon` flag on any cell where at least one of the 6 groups has `observed_count` < 100, and within each `#sig` tier they sort by `min count` descending so the weakest-supported cells fall to the bottom of their tier: A:ATA and A:TTA are concordant-enriched and clear the `#sig` >= 2 floor (`min count` 7 and 5 across the 6 groups), with A:ATA sorting high in the A-site block on its 6-of-6 significance and A:TTA falling to the last A-site row on its 2-of-6. Read every `rare-codon`-flagged row by direction, not magnitude.
- **p-magnitude-anchored-ranking** (per-CSV) — `p_adj` down to 3.53e-97 (control_day_0 P GAT) is best read as a joint readout of `log2_enrichment` magnitude x `total_n` x `iid-violation-binomial` amplification, not as biological strength. 110 of 1098 rows sit at `p_adj` < 1e-10. The Top-hits flag column applies `iid-amp` to every such row regardless of direction (no selective dampening) and adds `bg-tight` for `bg_freq` > 0.05 rows; the cross-group tables rank by `#sig` then `min count`, never by p magnitude.

### Considered but not applicable
*(none denied this run; no per-CSV proposals were rejected.)*

## For Chumeng (joint-reading hooks)
- Family: `within_condition_binomial` — sister CSV in this family that should be reconciled: `within_condition_binomial_aa.csv` (AA resolution; same design, same family-wide caveats).
- Open questions Chumeng should resolve at synthesis time:
  - Codon-vs-AA aggregation: at every site, does each AA's signal appear at a single codon (codon-specific) or across multiple synonyms (AA-aggregated)? Concrete sub-questions: at P site, does the Asp signal show up at GAT only or also GAC? At E site, does the Lys signal split between AAA and AAG with comparable magnitudes (the 6-group means here are AAA +0.78, AAG +0.60), or is the AA-level Lys hit driven primarily by one codon? Does the same pattern hold for Glu (GAA vs GAG)?
  - Cross-test concordance for the largest-magnitude shared-direction cells: do the 10 top cells (P:TCC, P:GCC, P:GAT, E:GCC, A:GCC, P:AAT, E:GCT, A:GCT, E:AAA, P:GGT) reappear with consistent direction in `per_timepoint_fisher_codon.csv` (BWM-vs-control at each timepoint) and in `between_timepoint_wilcoxon_*_codon.csv` (timepoint-shift contrasts)? If yes across all three test designs, the cell is reproducible at codon resolution; if a top binomial cell shows the alternative-explanation flag (`iid-amp`, `bg-tight`) and does NOT reappear in Fisher, that supports iid amplification rather than biology.
  - Discordant cells at codon level: 88 of the 183 (site, codon) cells show at least one cross-group sign disagreement; 25 of them clear the `#sig` >= 2 floor and appear in the discordant table. The largest by support is A:GGA (Gly, `min count` 455, `#sig` 5/6), positive in 5 of the 6 groups and negative only in control_day_0 (-0.14). Does A:GGA (or P:GGA, also discordant) show a direction flip in the per-timepoint Fisher's BWM-vs-control contrasts at any timepoint? If yes → a condition-specific effect at small magnitude; if no → noise at the magnitude floor.
  - Site-level depletion-of-Ala: GCC and GCT (both Ala) are concordantly depleted in all 6 groups at all 3 sites (6 of the 95 concordant cells, all 6 in the 45-cell concordant-depletion set). Does this reflect a global low Ala-codon frequency at stalls (an AA-level Ala-depletion the codon file resolves to its two most-common codons), or a codon-specific anti-stall preference? Cross-check: are GCA and GCG (the other two Ala synonyms, lower `bg_freq`) absent from the concordant set because their counts are too small or because they do not show the same direction?
  - File-vs-AA-file headline reconciliation: this codon file's largest-magnitude shared-direction cells span all three sites (P:TCC, P:GCC, P:GAT, E:GCC, A:GCC, P:AAT, E:GCT, A:GCT, E:AAA, P:GGT — 5 P, 3 E, 2 A). Does the AA file's cross-group concordance set emphasize the same site distribution, or does AA aggregation shift the magnitude-ranked list toward one site over another?
