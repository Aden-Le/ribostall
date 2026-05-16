---
input_csv: results/stall_sites/enrichment/analysis_stats/within_condition_binomial_codon.csv
family: within_condition_binomial
test_type: One-sample binomial test (k=stall_count out of n=total_n vs H0: p=bg_freq, two-sided), BH-FDR within each (group, site) family of 61 sense codons
test_type_source: user-confirmed
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
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "Each per-(group,site) BH family is 61 codons (~3x stricter than the AA file's 20 per family). AA-level signals can distribute across synonyms and fall below the per-(group,site) FDR threshold here. Sets up the codon-vs-AA power tradeoff: codon resolution gains specificity but loses some power per BH family."}
  - {label: "rare-codon-low-count-instability", proposed_by: dylan, status: confirmed, why: "Rare codons (ATA, TTA, AGG, GGG, CGG, ACG, TCG and similar) have stall_count < 50 in many (group, site) cells. Their log2_enrichment estimates are unstable because the binomial variance at small k is dominated by counting noise rather than the underlying frequency. Rows flagged `rare-codon` (k<50) in Top hits are reported but should be discounted relative to high-count rows (k>=50) when ranking by magnitude."}
  - {label: "p-magnitude-anchored-ranking", proposed_by: dylan, status: confirmed, why: "min p_adj = 3.5e-97 (control_day_0, P, GAT) reflects iid violation amplifying real but moderate per-site enrichment (log2 +0.828) at large stall count (k=1558 / n=27732). Codon-level extreme p magnitudes (110 rows with p_adj < 1e-10 across 1098 tests) are routine in this file because of high stall counts in common codons. Rank by effect size + count, never by p magnitude alone."}
caveats_considered: []
headline: "Codon-level binomial returns 491/1098 hits at FDR<0.05 (44.7%) split roughly evenly between enriched and depleted cells; 35 (codon, site) cells are concordant across all 6 groups (k>=50, p_adj<0.05, single sign), only 7 cells show any direction flip across groups and all of those have |log2|<=0.45. Largest-magnitude shared-direction cells (mean log2 across 6 groups) are P:TCC -1.04, P:GCC -1.02, P:GAT +0.98, E:GCC -0.94, A:GCC -0.90, P:AAT +0.88, E:GCT -0.86, A:GCT -0.82, E:AAA +0.78, P:GGT +0.74. File-min p_adj cell is control_day_0 P GAT (log2 +0.828, k=1558 of n=27732, p_adj 3.5e-97); the GAT-at-P enrichment is itself one of the 35 all-6-concordant cells (mean +0.984)."
user_directives:
  - "(invocation context) `flat-prior` token — apply A.2.1 through A.2.9 strictly; do not import findings from prior CSV interpretations as priors; read this CSV cold without consulting `_INDEX.md` cross-family hooks; rank features by (a) effect size in high-count rows (k>=50), (b) cross-synonym coherence at codon level, (c) reproducibility within this CSV's per-cell neighbours; report shared-direction features at equal billing with divergent features; for any p_adj < 1e-10 row name at least one alternative explanation."
  - "(per-CSV triage, carried over from original triage; not re-litigated this run) Test type confirmation → `Confirm: one-sample binomial vs bg_freq` (k=stall_count out of n=total_n, H0: p=bg_freq, two-sided, BH-FDR within each (group, site) family of 61 sense codons)."
  - "(per-CSV triage, carried over) Codon-specific caveats beyond family-wide → confirmed `larger-bh-family`, `rare-codon-low-count-instability`, `p-magnitude-anchored-ranking`."
  - "(per-CSV triage, carried over) Framing firmness → `Mixed`. Hits with consistent effect across groups + corroboration from the per-timepoint Fisher and between-condition Wilcoxon files are firm; binomial-only hits with extreme p but no cross-test corroboration are exploratory because of the iid violation."
  - "(per-CSV triage, this run) Spotlight → `No spotlight`. The prior invocation's AAG-at-E spotlight directive is revoked for this re-run; rank by data alone per A.2.3."
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
- (per-CSV triage, this run) "Spotlight" → none. Prior invocation's AAG-at-E spotlight directive is revoked for this re-run; data-ranked only per A.2.3.

## Headline
Across 1098 tests (6 groups x 3 sites x 61 sense codons) the file returns 491 hits at p_adj<0.05 (44.7%) and 552 at p_adj<0.10 (50.3%). 35 (codon, site) cells are concordant across all 6 groups under the same direction (k>=50, p_adj<0.05 in every group); only 7 cells show any direction flip across groups and all 7 have small max-magnitude (|log2|<=0.45). The 10 largest-magnitude shared-direction cells (by mean log2 across the 6 groups) are: P:TCC -1.04 (Ser), P:GCC -1.02 (Ala), P:GAT +0.98 (Asp), E:GCC -0.94 (Ala), A:GCC -0.90 (Ala), P:AAT +0.88 (Asn), E:GCT -0.86 (Ala), A:GCT -0.82 (Ala), E:AAA +0.78 (Lys), P:GGT +0.74 (Gly). File-min `p_adj` cell is control_day_0 P GAT (log2 +0.828, k=1558 / n=27732, p_adj=3.5e-97), itself one of the 35 all-6-concordant cells (mean log2 +0.984 across the 6 groups). Treat magnitude rankings as the primary read; the broad p_adj distribution from 3.5e-97 to ~1.0 is shape-driven by the iid-violation amplification at high k and is not a faithful priority signal across cells.

## Top hits

### Cross-group concordance (primary table)

35 (codon, site) cells are concordant in direction across all 6 groups at k>=50 and p_adj<0.05. Sorted by |mean log2| across groups; codon AA assignment in parentheses. `flag` carries `iid-amp` for any cell whose smallest per-group `p_adj` falls below 1e-10, `bg-tight` for cells whose `bg_freq` is above 5% (the iid amplification at high k pushes these to extreme p_adj routinely), and `rare-codon` if any one of the 6 group entries has k<100 (counting variability inside the concordant set).

| direction | cell (site:codon, AA) | mean log2 | per-group log2 (BWM_d0,d5,d10 / ctrl_d0,d5,d10) | k range | flag |
| --- | --- | --- | --- | --- | --- |
| depleted | P:TCC (Ser) | -1.039 | -1.28, -1.02, -1.03, -1.04, -0.89, -0.97 | 59-218 | iid-amp, rare-codon |
| depleted | P:GCC (Ala) | -1.016 | -1.16, -1.10, -0.92, -1.06, -0.85, -1.00 | 113-453 | iid-amp |
| enriched | P:GAT (Asp) | +0.984 | +0.88, +0.96, +1.13, +0.83, +1.00, +1.11 | 272-1558 | iid-amp |
| depleted | E:GCC (Ala) | -0.939 | -0.92, -0.94, -0.89, -0.90, -1.19, -0.80 | 134-506 | iid-amp |
| depleted | A:GCC (Ala) | -0.899 | -0.76, -1.05, -1.09, -0.85, -0.95, -0.70 | 126-525 | iid-amp |
| enriched | P:AAT (Asn) | +0.875 | +1.04, +0.94, +0.87, +0.84, +0.73, +0.83 | 74-662 | iid-amp, rare-codon |
| depleted | E:GCT (Ala) | -0.858 | -0.94, -0.72, -0.78, -0.94, -0.98, -0.79 | 118-558 | iid-amp |
| depleted | A:GCT (Ala) | -0.817 | -0.89, -0.79, -0.84, -0.88, -0.73, -0.77 | 122-581 | iid-amp |
| enriched | E:AAA (Lys) | +0.784 | +0.99, +0.78, +0.57, +0.81, +0.79, +0.77 | 105-867 | iid-amp |
| enriched | P:GGT (Gly) | +0.743 | +0.83, +0.66, +1.02, +0.48, +0.73, +0.72 | 81-412 | rare-codon |
| depleted | P:CTC (Leu) | -0.727 | -0.82, -0.64, -0.63, -0.70, -0.68, -0.89 | 128-476 | iid-amp |
| depleted | A:ACC (Thr) | -0.710 | -0.97, -0.77, -0.81, -0.45, -0.69, -0.58 | 107-493 | iid-amp |
| depleted | P:CAG (Gln) | -0.686 | -0.78, -0.64, -0.51, -0.65, -0.80, -0.73 | 50-258 | iid-amp, rare-codon |
| depleted | A:CTC (Leu) | -0.624 | -0.43, -0.48, -0.81, -0.43, -0.78, -0.81 | 139-573 | iid-amp |
| enriched | A:GAT (Asp) | +0.624 | +0.31, +0.86, +0.78, +0.43, +0.71, +0.67 | 184-1180 | iid-amp |
| depleted | P:GCT (Ala) | -0.612 | -0.72, -0.59, -0.72, -0.70, -0.47, -0.47 | 137-659 | iid-amp |
| enriched | E:AAG (Lys) | +0.600 | +0.69, +0.52, +0.52, +0.57, +0.75, +0.55 | 812-2495 | iid-amp, bg-tight |
| enriched | P:CAT (His) | +0.573 | +0.51, +0.80, +0.46, +0.67, +0.48, +0.52 | 52-343 | iid-amp, rare-codon |
| enriched | A:TTG (Leu) | +0.547 | +0.63, +0.56, +0.38, +0.73, +0.40, +0.58 | 96-653 | iid-amp, rare-codon |
| depleted | P:ACC (Thr) | -0.536 | -0.66, -0.65, -0.62, -0.31, -0.60, -0.38 | 132-546 | iid-amp |
| depleted | A:TCC (Ser) | -0.528 | -0.36, -0.55, -0.43, -0.43, -0.66, -0.69 | 154-303 | rare-codon |
| depleted | P:CGC (Arg) | -0.505 | -0.63, -0.41, -0.27, -0.69, -0.49, -0.55 | 92-225 | iid-amp, rare-codon |
| enriched | A:TAC (Tyr) | +0.500 | +0.31, +0.36, +0.65, +0.63, +0.48, +0.54 | 261-910 | iid-amp |
| depleted | E:CGC (Arg) | -0.498 | -0.35, -0.49, -0.36, -0.69, -0.61, -0.42 | 99-225 | iid-amp, rare-codon |
| enriched | E:AGA (Arg) | +0.442 | +0.62, +0.38, +0.27, +0.55, +0.50, +0.34 | 170-564 | iid-amp |
| enriched | A:GAA (Glu) | +0.422 | +0.34, +0.40, +0.34, +0.59, +0.49, +0.36 | 156-406 | iid-amp |
| enriched | E:AAC (Asn) | +0.418 | +0.45, +0.40, +0.35, +0.41, +0.39, +0.49 | 285-1076 | iid-amp |
| enriched | A:GAG (Glu) | +0.411 | +0.20, +0.39, +0.58, +0.49, +0.31, +0.50 | 481-918 | iid-amp |
| enriched | E:GAG (Glu) | +0.374 | +0.27, +0.35, +0.42, +0.34, +0.56, +0.30 | 430-839 | iid-amp |
| enriched | E:GAA (Glu) | +0.329 | +0.35, +0.42, +0.27, +0.32, +0.31, +0.27 | 156-489 |  |
| enriched | P:AAC (Asn) | +0.319 | +0.42, +0.34, +0.30, +0.31, +0.25, +0.30 | 226-665 |  |
| depleted | A:GTC (Val) | -0.287 | -0.20, -0.39, -0.26, -0.18, -0.34, -0.36 | 222-554 |  |
| enriched | P:TTC (Phe) | +0.208 | +0.27, +0.16, +0.30, +0.16, +0.18, +0.18 | 175-422 |  |
| depleted | A:TCT (Ser) | -0.428 | -0.55, -0.39, -0.59, -0.20, -0.43, -0.42 | 50-200 | rare-codon |
| depleted | P:CGT (Arg) | -0.419 | -0.41, -0.42, -0.45, -0.46, -0.36, -0.41 | 100-280 |  |

Direction-flip cells (cells where the sign changes between any two groups, k>=50, p_adj<0.05): 7 total, all small magnitude (max |log2| <= 0.45). Reported here at equal prominence to the concordant set per A.2.2:

| cell (site:codon, AA) | max_pos log2 | max_neg log2 | per-group log2 (ns where p_adj>=0.05 or k<50) |
| --- | --- | --- | --- |
| A:GGA (Gly) | +0.45 | -0.14 | +0.18, ns, +0.45, -0.14, +0.13, +0.36 |
| P:GGA (Gly) | +0.39 | -0.09 | +0.19, ns, +0.39, -0.09, ns, +0.38 |
| P:ATG (Met) | +0.14 | -0.38 | ns, ns, ns, +0.14, ns, -0.38 |
| P:AAG (Lys) | +0.14 | -0.37 | +0.14, ns, -0.23, ns, -0.23, -0.37 |
| E:ATC (Ile) | +0.14 | -0.28 | ns, ns, -0.28, +0.14, ns, ns |
| A:ATC (Ile) | +0.21 | -0.22 | -0.22, ns, ns, -0.12, +0.21, ns |
| P:ATC (Ile) | +0.16 | -0.19 | ns, ns, ns, ns, +0.16, -0.19 |

Note the empirical asymmetry: the file is dominated by cells where condition replicates (BWM, control) and timepoints (d0, d5, d10) all agree on direction at the same site/codon. There are no comparable-magnitude divergent cells in the data.

### Per-(group, site) tables

Each block below is a per-group view: top-5 enriched + top-5 depleted by |log2| within that (group, site) cell, restricted to k>=50 and p_adj<0.05. Codons with k<50 (rare-codon territory; flagged `rare-codon`) are excluded from these tables but are still in the n_significant counts.

<details>
<summary>BWM_day_0 (A / E / P)</summary>

#### BWM_day_0, A site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TAC (Y) | +0.824 | 6.14e-16 | iid-amp |
| enriched | TGG (W) | +0.767 | 2.51e-05 | rare-codon |
| enriched | CAT (H) | +0.741 | 8.01e-04 | rare-codon |
| enriched | TCG (S) | +0.694 | 4.18e-03 | rare-codon |
| enriched | TTG (L) | +0.629 | 1.89e-04 |  |
| depleted | AAC (N) | -1.057 | 1.31e-15 | iid-amp |
| depleted | ACC (T) | -0.967 | 8.59e-14 | iid-amp |
| depleted | GCT (A) | -0.891 | 2.91e-13 | iid-amp |
| depleted | GCC (A) | -0.755 | 1.37e-11 | iid-amp |
| depleted | TCT (S) | -0.574 | 1.16e-02 | rare-codon |

#### BWM_day_0, E site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | AAA (K) | +0.990 | 3.18e-10 |  |
| enriched | AAG (K) | +0.693 | 8.65e-42 | iid-amp, bg-tight |
| enriched | AGA (R) | +0.616 | 1.26e-06 |  |
| enriched | AAC (N) | +0.454 | 2.15e-06 |  |
| enriched | GAA (E) | +0.350 | 1.21e-02 |  |
| depleted | TCC (S) | -0.955 | 1.83e-09 | rare-codon |
| depleted | GCT (A) | -0.939 | 2.56e-14 | iid-amp |
| depleted | GCC (A) | -0.918 | 2.70e-15 | iid-amp |
| depleted | TAC (Y) | -0.517 | 6.76e-04 |  |
| depleted | ACC (T) | -0.471 | 1.37e-04 |  |

#### BWM_day_0, P site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | AAT (N) | +1.043 | 1.56e-07 | rare-codon |
| enriched | GAT (D) | +0.875 | 7.98e-19 | iid-amp |
| enriched | GGT (G) | +0.835 | 1.03e-05 | rare-codon |
| enriched | ATT (I) | +0.552 | 7.98e-04 |  |
| enriched | CAT (H) | +0.511 | 3.84e-02 | rare-codon |
| depleted | TCC (S) | -1.281 | 2.51e-14 | iid-amp, rare-codon |
| depleted | GCC (A) | -1.164 | 7.95e-22 | iid-amp |
| depleted | CTC (L) | -0.821 | 1.10e-11 | iid-amp |
| depleted | CAG (Q) | -0.781 | 2.09e-04 | rare-codon |
| depleted | GTC (V) | -0.730 | 1.31e-10 |  |

Per-cell n_sig counts (p_adj<0.05): A=27, E=16, P=28. Min p_adj cell: BWM_day_0 E AAG (8.65e-42).

</details>

<details>
<summary>BWM_day_5 (A / E / P)</summary>

#### BWM_day_5, A site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GAT (D) | +0.859 | 4.86e-37 | iid-amp |
| enriched | TAT (Y) | +0.848 | 1.74e-07 |  |
| enriched | TTG (L) | +0.558 | 1.09e-06 |  |
| enriched | CAT (H) | +0.535 | 1.15e-03 |  |
| enriched | TTT (F) | +0.489 | 2.15e-02 | rare-codon |
| depleted | GCC (A) | -1.047 | 1.50e-32 | iid-amp |
| depleted | GCT (A) | -0.793 | 3.78e-21 | iid-amp |
| depleted | ACC (T) | -0.771 | 1.54e-17 | iid-amp |
| depleted | TCC (S) | -0.550 | 6.90e-07 |  |
| depleted | CTC (L) | -0.482 | 1.10e-08 |  |

#### BWM_day_5, E site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | AAA (K) | +0.781 | 3.69e-13 | iid-amp |
| enriched | AAG (K) | +0.516 | 1.72e-34 | iid-amp, bg-tight |
| enriched | AAT (N) | +0.475 | 4.06e-04 |  |
| enriched | GAA (E) | +0.419 | 1.96e-07 |  |
| enriched | AAC (N) | +0.396 | 8.20e-09 |  |
| depleted | TGC (C) | -0.980 | 2.54e-08 | rare-codon |
| depleted | GCC (A) | -0.942 | 1.27e-27 | iid-amp |
| depleted | GCT (A) | -0.721 | 4.09e-18 | iid-amp |
| depleted | ACC (T) | -0.630 | 1.22e-12 | iid-amp |
| depleted | CGC (R) | -0.485 | 5.85e-05 |  |

#### BWM_day_5, P site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GAT (D) | +0.958 | 1.75e-48 | iid-amp |
| enriched | AAT (N) | +0.944 | 3.98e-17 | iid-amp |
| enriched | TTT (F) | +0.932 | 2.16e-07 | rare-codon |
| enriched | TAT (Y) | +0.848 | 1.11e-07 |  |
| enriched | CAT (H) | +0.804 | 3.51e-08 |  |
| depleted | GCC (A) | -1.102 | 3.12e-35 | iid-amp |
| depleted | TCC (S) | -1.019 | 6.16e-18 | iid-amp |
| depleted | ACC (T) | -0.647 | 2.49e-13 | iid-amp |
| depleted | CTC (L) | -0.642 | 3.66e-14 | iid-amp |
| depleted | CAG (Q) | -0.642 | 8.40e-07 |  |

Per-cell n_sig counts (p_adj<0.05): A=27, E=23, P=30. Min p_adj cell: BWM_day_5 P GAT (1.75e-48).

</details>

<details>
<summary>BWM_day_10 (A / E / P)</summary>

#### BWM_day_10, A site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GAT (D) | +0.776 | 7.75e-16 | iid-amp |
| enriched | ATT (I) | +0.673 | 2.99e-06 |  |
| enriched | GAG (E) | +0.584 | 3.75e-16 | iid-amp |
| enriched | CAT (H) | +0.455 | 3.99e-02 | rare-codon |
| enriched | GGA (G) | +0.447 | 8.27e-12 | iid-amp, bg-tight |
| depleted | GCC (A) | -1.090 | 6.67e-21 | iid-amp |
| depleted | GCT (A) | -0.842 | 2.49e-13 | iid-amp |
| depleted | CTC (L) | -0.808 | 2.70e-12 | iid-amp |
| depleted | ACC (T) | -0.807 | 2.97e-12 | iid-amp |
| depleted | TCT (S) | -0.594 | 3.32e-03 | rare-codon |

#### BWM_day_10, E site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | AAA (K) | +0.571 | 8.28e-04 |  |
| enriched | AAG (K) | +0.519 | 1.89e-22 | iid-amp, bg-tight |
| enriched | AAT (N) | +0.478 | 1.96e-02 | rare-codon |
| enriched | GAG (E) | +0.422 | 3.96e-08 |  |
| enriched | AAC (N) | +0.352 | 1.85e-04 |  |
| depleted | GCC (A) | -0.887 | 2.14e-15 | iid-amp |
| depleted | GCT (A) | -0.781 | 1.06e-11 | iid-amp |
| depleted | ACC (T) | -0.593 | 2.41e-07 |  |
| depleted | TCC (S) | -0.575 | 7.79e-05 |  |
| depleted | TAC (Y) | -0.482 | 4.68e-04 |  |

#### BWM_day_10, P site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GAT (D) | +1.128 | 5.70e-39 | iid-amp |
| enriched | GGT (G) | +1.024 | 9.83e-09 | rare-codon |
| enriched | AAT (N) | +0.866 | 9.64e-08 |  |
| enriched | AAA (K) | +0.677 | 1.54e-05 |  |
| enriched | AGC (S) | +0.521 | 1.58e-02 | rare-codon |
| depleted | TCC (S) | -1.034 | 7.17e-12 | iid-amp, rare-codon |
| depleted | GCC (A) | -0.917 | 3.19e-16 | iid-amp |
| depleted | GCT (A) | -0.722 | 2.05e-10 |  |
| depleted | CTC (L) | -0.632 | 2.00e-08 |  |
| depleted | ACC (T) | -0.621 | 3.45e-08 |  |

Per-cell n_sig counts (p_adj<0.05): A=28, E=16, P=26. Min p_adj cell: BWM_day_10 P GAT (5.70e-39).

</details>

<details>
<summary>control_day_0 (A / E / P)</summary>

#### control_day_0, A site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | ATA (I) | +1.217 | 3.57e-07 | rare-codon |
| enriched | TTA (L) | +1.090 | 1.19e-06 | rare-codon |
| enriched | TAT (Y) | +0.932 | 1.05e-31 | iid-amp |
| enriched | TTG (L) | +0.731 | 2.86e-32 | iid-amp |
| enriched | TAC (Y) | +0.629 | 3.94e-34 | iid-amp |
| depleted | GCT (A) | -0.879 | 6.48e-60 | iid-amp |
| depleted | GCC (A) | -0.849 | 2.50e-50 | iid-amp |
| depleted | AAC (N) | -0.478 | 1.03e-16 | iid-amp |
| depleted | AGC (S) | -0.469 | 2.79e-04 |  |
| depleted | ACC (T) | -0.452 | 7.13e-13 | iid-amp |

#### control_day_0, E site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | AAA (K) | +0.807 | 2.38e-51 | iid-amp |
| enriched | AAG (K) | +0.567 | 1.50e-79 | iid-amp, bg-tight |
| enriched | AGA (R) | +0.555 | 4.14e-17 | iid-amp |
| enriched | AAC (N) | +0.411 | 1.15e-18 | iid-amp |
| enriched | AAT (N) | +0.397 | 1.82e-08 |  |
| depleted | TGC (C) | -1.059 | 4.85e-20 | iid-amp |
| depleted | GCT (A) | -0.937 | 9.08e-67 | iid-amp |
| depleted | GCC (A) | -0.902 | 1.30e-55 | iid-amp |
| depleted | TGT (C) | -0.750 | 2.86e-05 | rare-codon |
| depleted | CGC (R) | -0.693 | 3.26e-14 | iid-amp |

#### control_day_0, P site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TAT (Y) | +0.896 | 5.45e-29 | iid-amp |
| enriched | AAT (N) | +0.837 | 1.33e-41 | iid-amp |
| enriched | GAT (D) | +0.828 | 3.53e-97 | iid-amp |
| enriched | CAT (H) | +0.671 | 4.89e-15 | iid-amp |
| enriched | AAA (K) | +0.521 | 2.85e-19 | iid-amp |
| depleted | GCC (A) | -1.062 | 1.40e-71 | iid-amp |
| depleted | TCC (S) | -1.044 | 4.44e-33 | iid-amp |
| depleted | CTC (L) | -0.702 | 1.73e-30 | iid-amp |
| depleted | GCT (A) | -0.697 | 1.79e-41 | iid-amp |
| depleted | CGC (R) | -0.693 | 2.26e-14 | iid-amp |

Per-cell n_sig counts (p_adj<0.05): A=42, E=30, P=38. Min p_adj cell: control_day_0 P GAT (3.53e-97 — file minimum).

</details>

<details>
<summary>control_day_5 (A / E / P)</summary>

#### control_day_5, A site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TAT (Y) | +0.761 | 9.61e-06 | rare-codon |
| enriched | GAT (D) | +0.706 | 7.29e-22 | iid-amp |
| enriched | CAT (H) | +0.522 | 1.99e-03 |  |
| enriched | AAT (N) | +0.509 | 8.97e-05 |  |
| enriched | GAA (E) | +0.492 | 2.99e-10 |  |
| depleted | GCC (A) | -0.952 | 1.33e-24 | iid-amp |
| depleted | CTC (L) | -0.784 | 8.27e-18 | iid-amp |
| depleted | GCT (A) | -0.727 | 4.96e-17 | iid-amp |
| depleted | ACC (T) | -0.685 | 6.39e-14 | iid-amp |
| depleted | TCC (S) | -0.664 | 3.91e-09 |  |

#### control_day_5, E site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | AAA (K) | +0.788 | 1.40e-12 | iid-amp |
| enriched | AAG (K) | +0.753 | 2.06e-75 | iid-amp, bg-tight |
| enriched | GAG (E) | +0.560 | 1.30e-24 | iid-amp |
| enriched | AAT (N) | +0.536 | 2.40e-05 |  |
| enriched | AGA (R) | +0.500 | 7.95e-07 |  |
| depleted | GCC (A) | -1.186 | 9.09e-35 | iid-amp |
| depleted | TGC (C) | -0.999 | 6.16e-09 | rare-codon |
| depleted | GCT (A) | -0.977 | 3.14e-27 | iid-amp |
| depleted | ACC (T) | -0.737 | 1.18e-15 | iid-amp |
| depleted | TAC (Y) | -0.599 | 5.36e-09 |  |

#### control_day_5, P site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GAT (D) | +1.001 | 4.22e-50 | iid-amp |
| enriched | TTT (F) | +0.780 | 4.86e-05 | rare-codon |
| enriched | GGT (G) | +0.731 | 9.65e-08 |  |
| enriched | AAT (N) | +0.726 | 1.70e-09 |  |
| enriched | TAT (Y) | +0.617 | 4.67e-04 | rare-codon |
| depleted | TCC (S) | -0.887 | 5.02e-14 | iid-amp |
| depleted | GCC (A) | -0.848 | 1.03e-20 | iid-amp |
| depleted | CAG (Q) | -0.801 | 2.08e-08 | rare-codon |
| depleted | TGG (W) | -0.717 | 1.42e-04 | rare-codon |
| depleted | CTC (L) | -0.676 | 5.02e-14 | iid-amp |

Per-cell n_sig counts (p_adj<0.05): A=31, E=26, P=28. Min p_adj cell: control_day_5 E AAG (2.06e-75).

</details>

<details>
<summary>control_day_10 (A / E / P)</summary>

#### control_day_10, A site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TAT (Y) | +0.887 | 1.69e-05 | rare-codon |
| enriched | GAT (D) | +0.666 | 2.45e-15 | iid-amp |
| enriched | TTG (L) | +0.578 | 1.82e-05 |  |
| enriched | TAC (Y) | +0.538 | 6.35e-09 |  |
| enriched | ATT (I) | +0.435 | 8.12e-04 |  |
| depleted | CTC (L) | -0.808 | 2.45e-15 | iid-amp |
| depleted | GCT (A) | -0.770 | 2.45e-15 | iid-amp |
| depleted | GCC (A) | -0.702 | 3.93e-14 | iid-amp |
| depleted | AAC (N) | -0.630 | 8.39e-10 |  |
| depleted | CAC (H) | -0.620 | 5.83e-05 | rare-codon |

#### control_day_10, E site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | AAA (K) | +0.766 | 2.15e-09 |  |
| enriched | AAG (K) | +0.553 | 9.17e-32 | iid-amp, bg-tight |
| enriched | AAT (N) | +0.498 | 2.60e-03 |  |
| enriched | AAC (N) | +0.492 | 4.39e-10 |  |
| enriched | GAA (E) | +0.427 | 1.14e-05 |  |
| depleted | GCC (A) | -0.798 | 3.75e-17 | iid-amp |
| depleted | GCT (A) | -0.792 | 4.59e-16 | iid-amp |
| depleted | CGT (R) | -0.661 | 2.43e-08 |  |
| depleted | CGC (R) | -0.625 | 2.56e-05 | rare-codon |
| depleted | TCT (S) | -0.557 | 9.42e-04 | rare-codon |

#### control_day_10, P site
| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TAT (Y) | +1.122 | 3.30e-09 | rare-codon |
| enriched | GAT (D) | +1.114 | 1.07e-52 | iid-amp |
| enriched | AAT (N) | +0.833 | 4.21e-09 |  |
| enriched | GGT (G) | +0.720 | 4.59e-06 |  |
| enriched | CAT (H) | +0.518 | 3.21e-03 | rare-codon |
| depleted | GCC (A) | -1.003 | 9.55e-25 | iid-amp |
| depleted | TCC (S) | -0.970 | 2.99e-12 | iid-amp, rare-codon |
| depleted | CTC (L) | -0.890 | 2.23e-18 | iid-amp |
| depleted | GTC (V) | -0.743 | 1.26e-14 | iid-amp |
| depleted | CAG (Q) | -0.730 | 4.86e-06 | rare-codon |

Per-cell n_sig counts (p_adj<0.05): A=25, E=19, P=31. Min p_adj cell: control_day_10 P GAT (1.07e-52).

</details>

## Numbers at a glance
- `n_tests`: 1098 (6 groups x 3 sites x 61 sense codons)
- `n_significant` (adjusted-p < 0.05): 491 (44.7%)
- `n_significant` (adjusted-p < 0.10): 552 (50.3%)
- `min adjusted-p`: 3.5335639633970407e-97 (control_day_0 P GAT)
- `p_floor`: n/a — user did not flag a CSV-level floor (binomial test has no analogous discrete floor at this n; iid violation amplifies the lower tail instead)
- Cells concordant in all 6 groups (k>=50, p_adj<0.05, single sign): 35 (20 depleted, 15 enriched)
- Cells with sign flip across groups (k>=50, p_adj<0.05 in >=2 groups): 7, all with max |log2| <= 0.45
- Rows with `p_adj < 1e-10`: 110 of 1098 (10.0%); each carries `iid-amp` in its flag column per A.2.6
- Per (group, site) summary (n_sig at FDR<0.05 / 61, min p_adj):

| group | A site | E site | P site |
| --- | --- | --- | --- |
| BWM_day_0 | 27 (6.14e-16) | 16 (8.65e-42) | 28 (7.95e-22) |
| BWM_day_5 | 27 (4.86e-37) | 23 (1.72e-34) | 30 (1.75e-48) |
| BWM_day_10 | 28 (6.67e-21) | 16 (1.89e-22) | 26 (5.70e-39) |
| control_day_0 | 42 (6.48e-60) | 30 (1.50e-79) | 38 (3.53e-97) |
| control_day_5 | 31 (1.33e-24) | 26 (2.06e-75) | 28 (4.22e-50) |
| control_day_10 | 25 (2.45e-15) | 19 (9.17e-32) | 31 (1.07e-52) |

## Methods
Dylan proposed a one-sample binomial test with H0 `stall_freq = bg_freq`, k=stall_count, n=total_n, two-sided, BH-FDR correction within each (group, site) family of 61 sense codons; user confirmed in the original triage and the answer was carried over to this re-run without re-litigation. Effect column is `log2_enrichment` (= log2(stall_freq / bg_freq)) with a 1e-6 pseudocount on `bg_freq` to avoid log(0). The `weighted_log2_enrichment` column scales by `(stall_freq - bg_freq)` to penalize rare-codon rows; reported in the file but not used as the primary effect for this interpretation. The test answers "is codon X observed at this site at a per-stall-event frequency that differs from its background-codon frequency in this group's transcripts?". It does *not* answer "is this codon's enrichment different between BWM and control" (that is the per-timepoint Fisher) and *not* "is the enrichment shifting across timepoints" (that is the between-timepoint Wilcoxon).

## Caveats
### Confirmed
- **pseudorep** (family-wide) — replicate counts within a (group, site) cell are pooled to form k and n, so the binomial sample variance is computed at the position-pool level rather than the replicate level. The replicate-level standard error is not represented in p; sister tests in the family inherit the same pooling. Inherited from family `within_condition_binomial`.
- **iid-violation-binomial** (family-wide) — the binomial assumes independent Bernoulli trials per stall event, but stall positions cluster on motifs and within transcripts, so the effective n is smaller than total_n. The test under-estimates p in proportion to the cluster strength; this is what produces the 110 rows below 1e-10 at moderate effect sizes. Inherited from family `within_condition_binomial`.
- **bg-pseudocount-1e-6** (family-wide) — `bg_freq` is computed with a 1e-6 pseudocount per codon to keep log(0) finite. For codons whose true bg_freq is ~1e-6 (e.g. unobserved or near-zero codons), `log2_enrichment` is dominated by the pseudocount and is not informative. Affects only very rare codons; the high-count rows in the Top hits are unaffected. Inherited from family `within_condition_binomial`.
- **bh-per-(group,site)** (family-wide) — BH correction is applied independently within each of the 18 (group, site) families of 61 codons, not across the full 1098-test grid. Two cells with the same raw p in different groups can therefore receive different `p_adj` values; cross-group p_adj rankings are not directly commensurable. Inherited from family `within_condition_binomial`.
- **larger-bh-family** (per-CSV) — each per-(group, site) BH family is 61 codons, ~3x larger than the AA file's 20-AA families. AA-level signals can distribute across synonyms and fall below the per-(group, site) FDR threshold here; AA aggregation is the place to look if the AA file shows a hit that the codon file's per-codon rows do not.
- **rare-codon-low-count-instability** (per-CSV) — rare codons (e.g. ATA k=51, TTA k=57 in control_day_0 A; many cells under k<50) have unstable log2 estimates because binomial variance at small k is dominated by counting noise. Top hits tables in the per-(group, site) blocks are restricted to k>=50; the cross-group concordance table is restricted to k>=50 in every one of the 6 groups. Two A-site rare-codon rows (ATA, TTA in control_day_0) appear in the per-(group, site) tables with `rare-codon` flags because k is just above 50 in that one group and well below in others.
- **p-magnitude-anchored-ranking** (per-CSV) — the file's min `p_adj` of 3.5e-97 (control_day_0 P GAT) reflects the iid-violation inflation of significance at large k (n=27732, k=1558) for an effect of moderate size (log2 +0.828). 110 of 1098 rows are at `p_adj < 1e-10` and the lower tail spans 87 orders of magnitude. Rank features by effect size + count + cross-group reproducibility; treat `p_adj` magnitude only as a coarse "strong / weak" indicator within a single (group, site) family, never as a feature-prominence axis across cells.

### Considered but not applicable
*(none denied this run; no per-CSV proposals were rejected.)*

## For Chumeng (joint-reading hooks)
- Family: `within_condition_binomial` — sister CSV in this family that should be reconciled: `within_condition_binomial_aa.csv` (AA resolution; same design, same family-wide caveats).
- Open questions Chumeng should resolve at synthesis time:
  - Codon-vs-AA aggregation: at every site, does each AA's signal appear at a single codon (codon-specific) or across multiple synonyms (AA-aggregated)? Concrete sub-questions: at P site, does the Asp signal show up at GAT only or also GAC? At E site, does the Lys signal split between AAA and AAG with comparable magnitudes (the 6-group means here are AAA +0.78, AAG +0.60), or is the AA-level Lys hit driven primarily by one codon? Does the same pattern hold for Glu (GAA vs GAG)?
  - Cross-test concordance for the largest-magnitude shared-direction cells: do the 10 top cells (P:TCC, P:GCC, P:GAT, E:GCC, A:GCC, P:AAT, E:GCT, A:GCT, E:AAA, P:GGT) reappear with consistent direction in `per_timepoint_fisher_codon.csv` (BWM-vs-control at each timepoint) and in `between_timepoint_wilcoxon_*_codon.csv` (timepoint-shift contrasts)? If yes across all three test designs, the cell is reproducible at codon resolution; if a top binomial cell shows the alternative-explanation flag (`iid-amp`, `bg-tight`) and does NOT reappear in Fisher, that supports iid amplification rather than biology.
  - Direction-flip cells at codon level: 7 cells show sign flips between groups, all with small max-magnitude (|log2| <= 0.45). The largest, GGA at A and P sites (Gly), is enriched in 4 groups and depleted only in control_day_0. Does GGA at A or P show direction flips in the per-timepoint Fisher's BWM-vs-control contrasts at any timepoint? If yes → genuine condition-specific effect at small magnitude; if no → likely just noise at the magnitude floor.
  - Site-level depletion-of-Ala: GCC and GCT (both Ala) appear concordantly depleted in all 6 groups at all 3 sites (6 of 35 concordant cells are GCC or GCT). Does this reflect a global low Ala-codon frequency at stalls (i.e. is this an AA-level Ala-depletion that the codon file resolves to its two most-common codons), or is it a codon-specific anti-stall preference? Cross-check: are GCA and GCG (the other two Ala synonyms, lower bg_freq) absent from the concordant set because their counts are too small or because they actually do not show the same direction?
  - File-vs-AA-file headline reconciliation: this codon file's largest-magnitude shared-direction cells span all three sites (P:TCC, P:GCC, P:GAT, E:GCC, A:GCC, P:AAT, E:GCT, A:GCT, E:AAA, P:GGT — 5 P, 3 E, 2 A). Does the AA file's cross-group concordance table emphasize the same site distribution, or does AA aggregation shift the magnitude-ranked list toward one site over others?
