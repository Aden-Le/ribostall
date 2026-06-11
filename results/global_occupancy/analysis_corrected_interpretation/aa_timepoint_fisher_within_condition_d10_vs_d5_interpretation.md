---
input_csv: results/global_occupancy/analysis_corrected/aa_timepoint_fisher_within_condition_d10_vs_d5.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), within-condition day_10-vs-day_5 contrast per (condition, site, aa); BH-FDR within each (condition, site) family of 20 AAs
test_type_source: user-confirmed
n_tests: 120
n_significant_fdr05: 107
n_significant_fdr10: 107
min_p_adj: 0.0
p_floor: null
pseudoreplicated: true
synced_from_olive_qmd: 2026-06-10
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "In the day_10-vs-day_5 contrast, BWM and control split direction at sites A and E: A:P (BWM log2_OR=-0.077 vs control +0.376), A:W (-0.048 vs +0.342), E:W (+0.141 vs -0.152), E:M (+0.051 vs -0.183), E:P (-0.061 vs +0.254). The within-condition design isolates the two conditions' trajectories; divergent cells are the design-target reading, reported alongside shared-direction cells per A.2.2. User-confirmed."}
  - {label: "near-universal-sig-large-N", proposed_by: dylan, status: confirmed, why: "107/120 tests clear FDR<0.05 and min p_adj underflows to 0.0 (5 cells) at whole-transcriptome pooled N. Section membership (both / BWM-only / control-only) is weakly discriminating; rank by |Effect change|, not by section or p. User-confirmed."}
caveats_considered:
  - {label: "rare-aa-low-count", proposed_by: dylan, status: denied, why: "All (condition, site, aa) cells have both day_10 and day_5 counts well above the within_condition_sig_split thresholds (BWM<100 / control<200); no low-count flag fires.", user_note: "Recorded not-applicable at AA resolution (codon sister differs at the in-frame stop TGA cells)."}
headline: "107/120 within-condition Fisher tests sig at FDR<0.05 (BWM 51/60, control 56/60; 48 cells sig in both, 3 BWM-only, 8 control-only); min p_adj underflows to 0.0 (5 cells) at whole-transcriptome N, so effects rank, not p. Largest BWM-vs-control divergences split direction at sites A and E: A:P (BWM log2_OR=-0.077 vs control +0.376), A:W (-0.048 vs +0.342), E:W (+0.141 vs -0.152), E:M (+0.051 vs -0.183). Largest shared-direction cells: A:G both enriched (+0.159 / +0.163), P:G both enriched (+0.153 / +0.176), P:I both depleted (-0.154 / -0.259)."
user_directives:
  - "(per-CSV triage) 'Confirm test type for the family?' -> 'Fisher's exact, BH per (condition, site)' (applies to all 6 files)."
  - "(per-CSV triage) 'Per-CSV caveats beyond the 4 locked family caveats?' -> confirmed control-vs-BWM-divergent-direction and near-universal-sig-large-N; rare-aa-low-count declined for AA files."
  - "(per-CSV triage) 'How firmly should this family read?' -> 'Firm' (significant cells read as established; still rank by |Effect change|, not p)."
  - "(per-CSV triage) Top-hits table source -> user authorised running `scripts/within_condition_sig_split.py` to generate the three-section paired tables transcribed below."
  - "(readback) Reconciled shared content from the corrected .qmd on 2026-06-10 -> no shared-number changes (Stages 3-6 found zero corrections); three-section tables already match Olive's layout; AA full-name expansion + Top-hits intro run-on split are Olive-only, not pulled in; Dylan keeps three-letter abbreviations and the <details> both-section wrapper; no asymptotic entry (Fisher)."
---

# Interpretation — aa_timepoint_fisher_within_condition_d10_vs_d5

> Source: `results/global_occupancy/analysis_corrected/aa_timepoint_fisher_within_condition_d10_vs_d5.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), within-condition day_10-vs-day_5 contrast per (condition, site, aa); BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (per-CSV triage) "Confirm test type for the family?" → "Fisher's exact, BH per (condition, site)" (confirmed for all 6 family files).
- (per-CSV triage) "Per-CSV caveats beyond the 4 locked family caveats?" → confirmed `control-vs-BWM-divergent-direction` and `near-universal-sig-large-N`; `rare-aa-low-count` declined for AA files and recorded considered-not-applicable.
- (per-CSV triage) "How firmly should this family read?" → "Firm". Significant cells read as established; the A.2.x large-N discipline still applies — rank by `|Effect change|`, not p.
- (per-CSV triage) Top-hits table source → user authorised running `scripts/within_condition_sig_split.py` (a display-only reshape of the existing `odds_ratio`/`p_adj` columns — no statistics re-run) to generate the three-section paired tables below.
- (readback) "Reconciled shared content from the corrected .qmd on 2026-06-10" → "No shared-number changes — Olive Stages 3-6 found zero corrections, so every shared value already agreed. The three-section paired tables already match Olive's layout (no table adoption owed). The AA full-name expansion (every appearance) and the Top-hits intro run-on split applied at Olive Stage 5 are Olive-only; Dylan keeps the bare three-letter abbreviations and the `<details>` both-section wrapper. No asymptotic-with-ties entry (Fisher family). Numbers audit: all shared numbers reconciled to the `.qmd`, the `For Chumeng` control-arm values verified against the raw CSV, none flagged un-verifiable."

## Headline
At AA-level within-condition Fisher d10 vs d5, 107/120 tests clear FDR<0.05 (BWM 51/60, control 56/60): 48 site x aa cells significant in both conditions, 3 BWM-only, 8 control-only. Minimum adjusted p underflows to 0.0 (5 cells) at whole-transcriptome pooled N (totals 1.3M-2.4M), so the informative axis is `Effect change`, not p; effects are small.

The largest BWM-vs-control divergences split direction between the conditions at sites A and E: A:P (BWM `log2_OR`=-0.077 vs control +0.376 enriched at d10), A:W (BWM -0.048 vs control +0.342), E:P (BWM -0.061 vs control +0.254), E:W (BWM +0.141 enriched vs control -0.152 depleted), E:M (BWM +0.051 vs control -0.183), A:L (BWM -0.083 vs control +0.161). Reported at equal billing (A.2.2), the largest shared-direction cells (both arms same sign) are A:G both enriched (BWM +0.159 / control +0.163), P:G both enriched (+0.153 / +0.176), P:I both depleted (-0.154 / -0.259), P:E both enriched (+0.149 / +0.044), E:A both enriched (+0.042 / +0.213).

## Top hits

`log2_OR` is the within-condition Fisher effect for the **day_10 vs day_5** contrast: positive = enriched at day_10 relative to day_5, negative = depleted. Each row pairs the BWM and control value for one (site, amino acid) cell; `Effect change` = BWM `log2_OR` − control `log2_OR`. Rows grouped by `site` in A → P → E order, then sorted by `Effect change` descending. Cells significant (FDR<0.05) in neither condition are omitted. Generated by `scripts/within_condition_sig_split.py`.

Every cell shown is FDR-significant in at least one arm at whole-transcriptome N (both-section cells overwhelmingly p_adj << 1e-10), so per A.2.6/A.2.4 the large-N anti-conservatism alternative explanation applies symmetrically to every such row and ranking is by `|Effect change|`, not p. The `Flag` column carries only `low-count` (none fires at AA resolution).

### Significant in both conditions (n = 48 site x amino acid cells)

<details>
<summary>Full 48-cell both-conditions table (A / P / E, sorted by Effect change desc)</summary>

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | K (Lys) | -0.038 | -0.260 | +0.222 |  |
| A | E (Glu) | +0.147 | -0.042 | +0.189 |  |
| A | M (Met) | +0.021 | -0.134 | +0.155 |  |
| A | Q (Gln) | -0.078 | -0.149 | +0.071 |  |
| A | D (Asp) | -0.044 | -0.071 | +0.027 |  |
| A | S (Ser) | +0.040 | +0.034 | +0.005 |  |
| A | C (Cys) | +0.066 | +0.069 | -0.003 |  |
| A | G (Gly) | +0.159 | +0.163 | -0.004 |  |
| A | T (Thr) | -0.056 | -0.046 | -0.009 |  |
| A | R (Arg) | +0.028 | +0.097 | -0.069 |  |
| A | Y (Tyr) | -0.110 | +0.020 | -0.130 |  |
| A | L (Leu) | -0.083 | +0.161 | -0.244 |  |
| A | W (Trp) | -0.048 | +0.342 | -0.390 |  |
| A | P (Pro) | -0.077 | +0.376 | -0.453 |  |
| P | V (Val) | +0.029 | -0.186 | +0.215 |  |
| P | F (Phe) | -0.026 | -0.159 | +0.134 |  |
| P | E (Glu) | +0.149 | +0.044 | +0.106 |  |
| P | I (Ile) | -0.154 | -0.259 | +0.105 |  |
| P | L (Leu) | -0.077 | -0.163 | +0.085 |  |
| P | D (Asp) | +0.123 | +0.069 | +0.054 |  |
| P | A (Ala) | +0.074 | +0.077 | -0.003 |  |
| P | C (Cys) | +0.040 | +0.061 | -0.021 |  |
| P | G (Gly) | +0.153 | +0.176 | -0.024 |  |
| P | M (Met) | -0.051 | -0.026 | -0.026 |  |
| P | N (Asn) | -0.095 | -0.054 | -0.041 |  |
| P | T (Thr) | -0.016 | +0.039 | -0.055 |  |
| P | K (Lys) | -0.098 | +0.030 | -0.129 |  |
| P | R (Arg) | +0.014 | +0.147 | -0.133 |  |
| P | P (Pro) | +0.048 | +0.198 | -0.150 |  |
| P | Q (Gln) | -0.042 | +0.130 | -0.172 |  |
| P | H (His) | -0.118 | +0.070 | -0.189 |  |
| E | W (Trp) | +0.141 | -0.152 | +0.293 |  |
| E | M (Met) | +0.051 | -0.183 | +0.234 |  |
| E | G (Gly) | +0.136 | -0.021 | +0.156 |  |
| E | K (Lys) | -0.049 | -0.167 | +0.118 |  |
| E | R (Arg) | +0.036 | -0.073 | +0.109 |  |
| E | C (Cys) | +0.045 | -0.042 | +0.088 |  |
| E | E (Glu) | +0.015 | -0.060 | +0.076 |  |
| E | N (Asn) | -0.041 | -0.110 | +0.069 |  |
| E | S (Ser) | +0.047 | +0.037 | +0.011 |  |
| E | L (Leu) | +0.024 | +0.020 | +0.005 |  |
| E | V (Val) | -0.014 | +0.047 | -0.061 |  |
| E | I (Ile) | -0.075 | +0.023 | -0.098 |  |
| E | F (Phe) | -0.072 | +0.088 | -0.160 |  |
| E | T (Thr) | -0.038 | +0.132 | -0.170 |  |
| E | A (Ala) | +0.042 | +0.213 | -0.171 |  |
| E | Y (Tyr) | -0.148 | +0.048 | -0.197 |  |
| E | P (Pro) | -0.061 | +0.254 | -0.315 |  |

</details>

### Significant in BWM only (n = 3 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | H (His) | -0.068 | +0.014 | -0.082 |  |
| P | Y (Tyr) | -0.136 | +0.005 | -0.142 |  |
| E | D (Asp) | +0.032 | -0.003 | +0.035 |  |

### Significant in control only (n = 8 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | N (Asn) | -0.001 | -0.356 | +0.355 |  |
| A | I (Ile) | +0.010 | -0.201 | +0.210 |  |
| A | F (Phe) | +0.009 | +0.041 | -0.033 |  |
| A | A (Ala) | -0.008 | +0.283 | -0.291 |  |
| P | S (Ser) | -0.010 | -0.058 | +0.049 |  |
| P | W (Trp) | +0.022 | +0.100 | -0.078 |  |
| E | Q (Gln) | +0.011 | -0.057 | +0.068 |  |
| E | H (His) | -0.007 | +0.029 | -0.036 |  |

## Numbers at a glance
- `n_tests`: 120 (60 per condition; 3 sites x 20 AAs each)
- `n_significant` (adjusted-p < 0.05): 107 (BWM 51/60, control 56/60)
- `n_significant` (adjusted-p < 0.10): 107 (no cell sits in the [0.05, 0.10) band)
- `min adjusted-p`: 0.0 (underflow; 5 cells at exactly 0.0). Smallest non-zero p_adj is 5.257e-235 (E,control,P).
- `p_floor`: n/a — Fisher with pooled N in the millions has no meaningful floor; the dominant concern is `large-N-Fisher-anticonservative` (family-wide).
- Per (condition, site) sig at FDR<0.05: BWM A 15/20, P 18/20, E 18/20; control A 18/20, P 19/20, E 19/20.
- Section split: 48 cells sig in both, 3 BWM-only, 8 control-only (59 of 60 site x aa cells sig in ≥1 condition).

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 of (aa_count, total − aa_count) at day_10 vs day_5 within each condition; user confirmed. `global_codon_occ_stats.py` (Analysis 3b) pools replicate counts per (condition, timepoint) before the test and applies BH-FDR within each (condition, site) family of 20 AAs; each of the 6 (condition, site) sub-families is corrected independently. Effect is `log2_OR` (>0 = enriched at day_10 relative to day_5, <0 = depleted). The test does not compare BWM against control directly (that is the `control-vs-BWM-divergent-direction` reading) and does not test enrichment vs the transcriptomic background (the `within_condition_binomial` family).

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* — 2 replicates per (condition, timepoint) summed into the 2x2 before Fisher; anti-conservative p. (Inherited — see `_INDEX.md`.)
- **large-N-Fisher-anticonservative** *(family-wide)* — pooled totals are whole-transcriptome footprint counts (day_10 ~1.56M BWM / ~1.26M control; day_5 ~2.01M BWM / ~2.41M control); 5 cells underflow to 0.0. `log2_OR` is the primary effect column. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* — BH within each of the 6 (condition, site) families of 20 AAs; `p_adj` is corrected within the sub-family, not across the 120-row file. (Inherited.)
- **within-condition-clean** *(family-wide)* — no condition/timepoint pooling across the contrast; structurally cleaner than the Wilcoxon families. (Inherited.)
- **control-vs-BWM-divergent-direction** *(per-CSV)* — flagged for Chumeng. Direction-split cells span sites A and E: A:P (BWM `log2_OR`=-0.077 / control +0.376), A:W (-0.048 / +0.342), E:P (-0.061 / +0.254), E:W (+0.141 / -0.152), E:M (+0.051 / -0.183), A:L (-0.083 / +0.161). The design-target reading.
- **near-universal-sig-large-N** *(per-CSV)* — 107/120 tests clear FDR<0.05, so section membership is weakly discriminating; rank by `|Effect change|`.

### Considered but not applicable
- **rare-aa-low-count** — every (condition, site, aa) cell has both day_10 and day_5 counts well above the `within_condition_sig_split` thresholds (BWM<100 / control<200); no low-count flag fires at AA resolution (codon sister differs at in-frame stop TGA).

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: `codon_timepoint_fisher_within_condition_d10_vs_d5` (codon refinement of this contrast), plus the d10_vs_d0 and d5_vs_d0 AA files (for monotonicity / biphasic checks: does d10_vs_d0 ≈ d5_vs_d0 + d10_vs_d5 per cell?). See `## Joint-reading suggestions` in [`_INDEX.md`](_INDEX.md).
- Falsifier: this contrast's control-only enrichments at site A (A:P control +0.376, A:W control +0.342, A:A control +0.283 — all flat in BWM) and the E-site direction splits (E:W, E:M) — do they reproduce at codon resolution localised to one synonym, or spread across synonyms?
- Falsifier on monotonicity: A:W and A:P show control enriched at d10 vs d5 here, while in d10_vs_d0 the same A:W is BWM-depleted/control-up. Does the d5_vs_d0 file place these on a consistent additive trajectory, or is the d5→d10 step the opposite sign of the d0→d5 step (biphasic)? Chumeng resolves; Dylan reports the per-cell d10-vs-d5 numbers only.
- Falsifier for the shared-direction cells (A:G↑, P:G↑, P:I↓): do these appear as stable baselines across all 6 groups in `within_condition_binomial`? A baseline both conditions carry is not a d10-vs-d5 effect.
