---
input_csv: results/global_occupancy/analysis_corrected/aa_timepoint_fisher_within_condition_d5_vs_d0.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), within-condition day_5-vs-day_0 contrast per (condition, site, aa); BH-FDR within each (condition, site) family of 20 AAs
test_type_source: user-confirmed
n_tests: 120
n_significant_fdr05: 115
n_significant_fdr10: 115
min_p_adj: 0.0
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "This contrast is shared-direction-dominant (both conditions move together on G/W/P depletion and N/D/M/I enrichment), but several cells still split direction: E:Y (BWM log2_OR=+0.099 vs control -0.131), A:C (-0.113 vs +0.095), E:K (-0.080 vs +0.101), P:M (+0.154 vs -0.041), E:H (+0.132 vs -0.049), E:T (+0.070 vs -0.111). Reported alongside the shared-direction cells per A.2.2. User-confirmed."}
  - {label: "near-universal-sig-large-N", proposed_by: dylan, status: confirmed, why: "115/120 tests clear FDR<0.05 (the most of any file in the family) and min p_adj underflows to 0.0 (13 cells) at whole-transcriptome pooled N. Section membership is barely discriminating here; rank by |Effect change|, not by section or p. User-confirmed."}
caveats_considered:
  - {label: "rare-aa-low-count", proposed_by: dylan, status: denied, why: "All (condition, site, aa) cells have both day_5 and day_0 counts well above the within_condition_sig_split thresholds (BWM<100 / control<200); no low-count flag fires.", user_note: "Recorded not-applicable at AA resolution (codon sister differs at in-frame stop TGA)."}
headline: "115/120 within-condition Fisher tests sig at FDR<0.05 (BWM 56/60, control 59/60; 55 cells sig in both, 1 BWM-only, 4 control-only) -- the most coordinated file in the family; min p_adj underflows to 0.0 (13 cells) at whole-transcriptome N. This contrast is shared-direction-dominant: both conditions deplete G (A -0.253/-0.226, P -0.377/-0.229), W (A -0.538/-0.255) and P (E -0.327/-0.340), and enrich N (A +0.251/+0.227), M (A +0.186/+0.230), D (A +0.273/+0.074). Largest BWM-vs-control divergences are smaller: E:Y (+0.099 vs -0.131), A:C (-0.113 vs +0.095), E:K (-0.080 vs +0.101)."
user_directives:
  - "(per-CSV triage) 'Confirm test type for the family?' -> 'Fisher's exact, BH per (condition, site)' (applies to all 6 files)."
  - "(per-CSV triage) 'Per-CSV caveats beyond the 4 locked family caveats?' -> confirmed control-vs-BWM-divergent-direction and near-universal-sig-large-N; rare-aa-low-count declined for AA files."
  - "(per-CSV triage) 'How firmly should this family read?' -> 'Firm' (significant cells read as established; still rank by |Effect change|, not p)."
  - "(per-CSV triage) Top-hits table source -> user authorised running `scripts/within_condition_sig_split.py` to generate the three-section paired tables transcribed below."
---

# Interpretation — aa_timepoint_fisher_within_condition_d5_vs_d0

> Source: `results/global_occupancy/analysis_corrected/aa_timepoint_fisher_within_condition_d5_vs_d0.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), within-condition day_5-vs-day_0 contrast per (condition, site, aa); BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (per-CSV triage) "Confirm test type for the family?" → "Fisher's exact, BH per (condition, site)" (confirmed for all 6 family files).
- (per-CSV triage) "Per-CSV caveats beyond the 4 locked family caveats?" → confirmed `control-vs-BWM-divergent-direction` and `near-universal-sig-large-N`; `rare-aa-low-count` declined for AA files and recorded considered-not-applicable.
- (per-CSV triage) "How firmly should this family read?" → "Firm". Significant cells read as established; the A.2.x large-N discipline still applies — rank by `|Effect change|`, not p.
- (per-CSV triage) Top-hits table source → user authorised running `scripts/within_condition_sig_split.py` (a display-only reshape of the existing `odds_ratio`/`p_adj` columns — no statistics re-run) to generate the three-section paired tables below.

## Headline
At AA-level within-condition Fisher d5 vs d0, 115/120 tests clear FDR<0.05 (BWM 56/60, control 59/60) — the most of any file in the family: 55 site x aa cells significant in both conditions, 1 BWM-only, 4 control-only. Minimum adjusted p underflows to 0.0 (13 cells) at whole-transcriptome pooled N (totals 2.0M-3.4M); the informative axis is `Effect change`, not p.

This contrast is shared-direction-dominant — BWM and control move the same way on most large cells. Both conditions deplete G at every site (A -0.253 / -0.226, P -0.377 / -0.229, E -0.204 / -0.112), deplete W at site A (-0.538 / -0.255) and P at site E (-0.327 / -0.340), and enrich N (A +0.251 / +0.227), M (A +0.186 / +0.230), D (A +0.273 / +0.074) and I (A +0.200 / +0.184). Reported at equal billing (A.2.2), the largest BWM-vs-control divergences are smaller and scattered: E:Y (BWM +0.099 vs control -0.131), A:C (BWM -0.113 vs control +0.095), E:K (BWM -0.080 vs control +0.101), P:M (BWM +0.154 vs control -0.041), E:H (BWM +0.132 vs control -0.049), E:T (BWM +0.070 vs control -0.111).

## Top hits

`log2_OR` is the within-condition Fisher effect for the **day_5 vs day_0** contrast: positive = enriched at day_5 relative to day_0, negative = depleted. Each row pairs the BWM and control value for one (site, amino acid) cell; `Effect change` = BWM `log2_OR` − control `log2_OR`. Rows grouped by `site` in A → P → E order, then sorted by `Effect change` descending. Cells significant (FDR<0.05) in neither condition are omitted. Generated by `scripts/within_condition_sig_split.py`.

Every cell shown is FDR-significant in at least one arm at whole-transcriptome N (both-section cells overwhelmingly p_adj << 1e-10), so per A.2.6/A.2.4 the large-N anti-conservatism alternative explanation applies symmetrically and ranking is by `|Effect change|`, not p. The `Flag` column carries only `low-count` (none fires at AA resolution).

### Significant in both conditions (n = 55 site x amino acid cells)

<details>
<summary>Full 55-cell both-conditions table (A / P / E, sorted by Effect change desc)</summary>

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | P (Pro) | -0.154 | -0.379 | +0.225 |  |
| A | D (Asp) | +0.273 | +0.074 | +0.199 |  |
| A | A (Ala) | -0.089 | -0.244 | +0.155 |  |
| A | V (Val) | +0.090 | -0.059 | +0.149 |  |
| A | F (Phe) | +0.098 | +0.052 | +0.045 |  |
| A | T (Thr) | +0.070 | +0.035 | +0.034 |  |
| A | H (His) | -0.026 | -0.052 | +0.026 |  |
| A | N (Asn) | +0.251 | +0.227 | +0.024 |  |
| A | L (Leu) | -0.099 | -0.118 | +0.018 |  |
| A | I (Ile) | +0.200 | +0.184 | +0.016 |  |
| A | Q (Gln) | -0.025 | -0.031 | +0.006 |  |
| A | R (Arg) | -0.021 | -0.024 | +0.003 |  |
| A | S (Ser) | +0.023 | +0.048 | -0.025 |  |
| A | G (Gly) | -0.253 | -0.226 | -0.028 |  |
| A | M (Met) | +0.186 | +0.230 | -0.044 |  |
| A | K (Lys) | -0.045 | +0.114 | -0.159 |  |
| A | C (Cys) | -0.113 | +0.095 | -0.208 |  |
| A | W (Trp) | -0.538 | -0.255 | -0.284 |  |
| P | H (His) | +0.084 | -0.112 | +0.196 |  |
| P | M (Met) | +0.154 | -0.041 | +0.195 |  |
| P | K (Lys) | -0.021 | -0.201 | +0.181 |  |
| P | P (Pro) | -0.072 | -0.243 | +0.171 |  |
| P | L (Leu) | +0.170 | +0.059 | +0.111 |  |
| P | Y (Tyr) | +0.069 | -0.031 | +0.100 |  |
| P | T (Thr) | +0.060 | -0.031 | +0.091 |  |
| P | R (Arg) | -0.048 | -0.091 | +0.043 |  |
| P | E (Glu) | +0.058 | +0.058 | +0.001 |  |
| P | S (Ser) | +0.094 | +0.102 | -0.008 |  |
| P | C (Cys) | +0.072 | +0.115 | -0.043 |  |
| P | A (Ala) | -0.088 | -0.043 | -0.045 |  |
| P | I (Ile) | +0.070 | +0.121 | -0.051 |  |
| P | N (Asn) | +0.082 | +0.137 | -0.055 |  |
| P | G (Gly) | -0.377 | -0.229 | -0.149 |  |
| P | V (Val) | -0.015 | +0.134 | -0.149 |  |
| P | F (Phe) | +0.037 | +0.210 | -0.173 |  |
| P | D (Asp) | -0.061 | +0.154 | -0.216 |  |
| E | Y (Tyr) | +0.099 | -0.131 | +0.230 |  |
| E | H (His) | +0.132 | -0.049 | +0.181 |  |
| E | T (Thr) | +0.070 | -0.111 | +0.180 |  |
| E | A (Ala) | -0.052 | -0.199 | +0.147 |  |
| E | F (Phe) | +0.062 | -0.071 | +0.133 |  |
| E | S (Ser) | +0.198 | +0.082 | +0.117 |  |
| E | V (Val) | -0.024 | -0.064 | +0.039 |  |
| E | P (Pro) | -0.327 | -0.340 | +0.013 |  |
| E | L (Leu) | +0.069 | +0.057 | +0.012 |  |
| E | Q (Gln) | +0.065 | +0.058 | +0.007 |  |
| E | D (Asp) | +0.035 | +0.056 | -0.022 |  |
| E | C (Cys) | +0.064 | +0.094 | -0.030 |  |
| E | E (Glu) | +0.057 | +0.106 | -0.049 |  |
| E | N (Asn) | +0.059 | +0.118 | -0.059 |  |
| E | G (Gly) | -0.204 | -0.112 | -0.092 |  |
| E | R (Arg) | -0.051 | +0.080 | -0.130 |  |
| E | M (Met) | +0.083 | +0.235 | -0.152 |  |
| E | K (Lys) | -0.080 | +0.101 | -0.181 |  |
| E | W (Trp) | -0.058 | +0.151 | -0.209 |  |

</details>

### Significant in BWM only (n = 1 site x amino acid cell)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | Y (Tyr) | -0.151 | -0.002 | -0.149 |  |

### Significant in control only (n = 4 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | E (Glu) | +0.007 | +0.139 | -0.131 |  |
| P | W (Trp) | +0.013 | -0.207 | +0.219 |  |
| P | Q (Gln) | +0.010 | -0.175 | +0.185 |  |
| E | I (Ile) | +0.005 | -0.057 | +0.062 |  |

## Numbers at a glance
- `n_tests`: 120 (60 per condition; 3 sites x 20 AAs each)
- `n_significant` (adjusted-p < 0.05): 115 (BWM 56/60, control 59/60)
- `n_significant` (adjusted-p < 0.10): 115 (no cell sits in the [0.05, 0.10) band)
- `min adjusted-p`: 0.0 (underflow; 13 cells at exactly 0.0). Smallest non-zero p_adj is 1.986e-296 (A,control,I).
- `p_floor`: n/a — Fisher with pooled N in the millions has no meaningful floor; the dominant concern is `large-N-Fisher-anticonservative` (family-wide).
- Per (condition, site) sig at FDR<0.05: BWM A 19/20, P 18/20, E 19/20; control A 19/20, P 20/20, E 20/20.
- Section split: 55 cells sig in both, 1 BWM-only, 4 control-only (60 of 60 site x aa cells sig in ≥1 condition).

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 of (aa_count, total − aa_count) at day_5 vs day_0 within each condition; user confirmed. `global_codon_occ_stats.py` (Analysis 3b) pools replicate counts per (condition, timepoint) before the test and applies BH-FDR within each (condition, site) family of 20 AAs; each of the 6 (condition, site) sub-families is corrected independently. Effect is `log2_OR` (>0 = enriched at day_5 relative to day_0, <0 = depleted). The test does not compare BWM against control directly (that is the `control-vs-BWM-divergent-direction` reading) and does not test enrichment vs the transcriptomic background (the `within_condition_binomial` family).

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* — 2 replicates per (condition, timepoint) summed into the 2x2 before Fisher; anti-conservative p. (Inherited — see `_INDEX.md`.)
- **large-N-Fisher-anticonservative** *(family-wide)* — pooled totals are whole-transcriptome footprint counts (day_5 ~2.01M BWM / ~2.41M control; day_0 ~2.11M BWM / ~3.37M control); 13 cells underflow to 0.0 (the most in the family). `log2_OR` is the primary effect column. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* — BH within each of the 6 (condition, site) families of 20 AAs; `p_adj` is corrected within the sub-family. (Inherited.)
- **within-condition-clean** *(family-wide)* — no condition/timepoint pooling across the contrast; structurally cleaner than the Wilcoxon families. (Inherited.)
- **control-vs-BWM-divergent-direction** *(per-CSV)* — flagged for Chumeng. This contrast is shared-direction-dominant, so divergences are fewer and smaller than in d10_vs_d0 / d10_vs_d5: E:Y (BWM `log2_OR`=+0.099 / control -0.131), A:C (-0.113 / +0.095), E:K (-0.080 / +0.101), P:M (+0.154 / -0.041), E:H (+0.132 / -0.049), E:T (+0.070 / -0.111).
- **near-universal-sig-large-N** *(per-CSV)* — 115/120 tests clear FDR<0.05 (highest in the family); section membership is barely discriminating; rank by `|Effect change|`.

### Considered but not applicable
- **rare-aa-low-count** — every (condition, site, aa) cell has both day_5 and day_0 counts well above the `within_condition_sig_split` thresholds (BWM<100 / control<200); no low-count flag fires at AA resolution (codon sister differs at in-frame stop TGA).

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: `codon_timepoint_fisher_within_condition_d5_vs_d0` (codon refinement of this contrast), plus the d10_vs_d0 and d10_vs_d5 AA files (additivity check: does d10_vs_d0 ≈ d5_vs_d0 + d10_vs_d5 per cell?). See `## Joint-reading suggestions` in [`_INDEX.md`](_INDEX.md).
- Falsifier on the coordinated signal: G is depleted at all three sites in both conditions, W depleted at A in both, P depleted at E in both. Do these reappear as stable baselines across all 6 groups in `within_condition_binomial` (then they are baseline composition shifts both conditions share, not a d5-vs-d0 effect), or are they group-variable?
- Falsifier on resolution: does the site-A enrichment of N/D/I/M and depletion of G/W localise to specific synonyms at codon resolution, or move with the amino acid as a block?
- Falsifier on the divergences (E:Y, A:C, E:K): these are the only cells where BWM and control split direction in this otherwise coordinated contrast. Do any of them reproduce in `per_timepoint_fisher` (fixed-timepoint BWM-vs-control) with consistent sign, or only here?
