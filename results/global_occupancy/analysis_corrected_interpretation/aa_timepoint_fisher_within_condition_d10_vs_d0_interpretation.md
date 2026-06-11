---
input_csv: results/global_occupancy/analysis_corrected/aa_timepoint_fisher_within_condition_d10_vs_d0.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), within-condition day_10-vs-day_0 contrast per (condition, site, aa); BH-FDR within each (condition, site) family of 20 AAs
test_type_source: user-confirmed
n_tests: 120
n_significant_fdr05: 104
n_significant_fdr10: 108
min_p_adj: 0.0
p_floor: null
pseudoreplicated: true
synced_from_olive_qmd: 2026-06-08
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Several site x aa cells have BWM and control moving opposite ways (or one moving while the other is flat) in the day_10-vs-day_0 contrast, e.g. A:N (BWM log2_OR=+0.250 vs control -0.129), A:W (BWM -0.586 vs control +0.088), A:Y (BWM -0.262 vs control +0.017). The within-condition design isolates the two conditions' trajectories; divergent cells are the design-target reading and are reported alongside shared-direction cells per A.2.2. User-confirmed."}
  - {label: "near-universal-sig-large-N", proposed_by: dylan, status: confirmed, why: "104/120 tests clear FDR<0.05 and min p_adj underflows to 0.0 (4 cells) at whole-transcriptome pooled N (totals 1.3M-3.4M). 'Significant in both / BWM-only / control-only' section membership is therefore weakly discriminating; rank by |Effect change| magnitude, not by which section a cell lands in or by p. User-confirmed."}
caveats_considered:
  - {label: "rare-aa-low-count", proposed_by: dylan, status: denied, why: "Every (condition, site, aa) cell has both day_10 and day_0 counts in the thousands-to-hundreds-of-thousands (smallest arm here, control P:W day_10=9188); none falls below the within_condition_sig_split thresholds (BWM<100 / control<200), so no low-count flag fires.", user_note: "Recorded not-applicable at AA resolution (unlike the codon sister, where in-frame stop TGA cells trip the low-count flag)."}
headline: "104/120 within-condition Fisher tests sig at FDR<0.05 (BWM 54/60, control 50/60; 46 cells sig in both, 8 BWM-only, 4 control-only); min p_adj underflows to 0.0 (4 cells) at whole-transcriptome N so effects, not p, rank. Largest BWM-vs-control divergences sit at site A: A:W (BWM log2_OR=-0.586 vs control +0.088), A:N (+0.250 vs -0.129), A:Y (-0.262 vs +0.017). Largest shared-direction cells: E:P both depleted (BWM -0.387 / control -0.086), E:S both enriched (+0.246 / +0.118), P:E both enriched (+0.208 / +0.101)."
user_directives:
  - "(per-CSV triage) 'Confirm test type for the family?' -> 'Fisher's exact, BH per (condition, site)' (applies to all 6 files)."
  - "(per-CSV triage) 'Per-CSV caveats beyond the 4 locked family caveats?' -> confirmed control-vs-BWM-divergent-direction and near-universal-sig-large-N; rare-aa-low-count declined for AA files (counts huge) and recorded considered-not-applicable."
  - "(per-CSV triage) 'How firmly should this family read?' -> 'Firm' (read significant cells as established; still honour the large-N discipline -- rank by |Effect change|, not p)."
  - "(per-CSV triage) Top-hits table source -> user authorised running `scripts/within_condition_sig_split.py` (a display-only reshape of existing odds_ratio/p_adj columns; no stats re-run) to generate the three-section paired tables transcribed below."
  - "(readback) Reconciled shared content from the corrected .qmd on 2026-06-08 -> tables already aligned (three-section 46/8/4, no raw-p); corrected the rare-aa-low-count smallest-arm cite BWM E:W day_10=15966 -> control P:W day_10=9188 (front-matter + Caveats body); no asymptotic entry (Fisher); Olive-only sections not imported."
---

# Interpretation — aa_timepoint_fisher_within_condition_d10_vs_d0

> Source: `results/global_occupancy/analysis_corrected/aa_timepoint_fisher_within_condition_d10_vs_d0.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), within-condition day_10-vs-day_0 contrast per (condition, site, aa); BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (per-CSV triage) "Confirm test type for the family?" → "Fisher's exact, BH per (condition, site)" (confirmed for all 6 family files).
- (per-CSV triage) "Per-CSV caveats beyond the 4 locked family caveats?" → confirmed `control-vs-BWM-divergent-direction` and `near-universal-sig-large-N`; `rare-aa-low-count` declined for AA files (counts in the thousands+) and recorded considered-not-applicable.
- (per-CSV triage) "How firmly should this family read?" → "Firm". Significant cells read as established; the A.2.x large-N discipline still applies — rank by `|Effect change|`, not by p magnitude.
- (per-CSV triage) Top-hits table source → user authorised running `scripts/within_condition_sig_split.py` (a display-only reshape of the existing `odds_ratio`/`p_adj` columns — no statistics are re-run) to generate the three-section paired tables transcribed below.
- (readback) "Reconciled shared content from the corrected .qmd on 2026-06-08" → "Tables already aligned (three-section 46/8/4, no raw-p column); corrected the `rare-aa-low-count` smallest-arm cite BWM E:W day_10=15966 → control P:W day_10=9188 (front-matter + Caveats body); no asymptotic-with-ties entry (Fisher family); Olive-only Biological interpretation / composite / plots not imported; Dylan-only sections kept."

## Headline
At AA-level within-condition Fisher d10 vs d0, 104/120 tests clear FDR<0.05 (BWM 54/60, control 50/60): 46 site x aa cells are significant in both conditions, 8 in BWM only, 4 in control only. Minimum adjusted p underflows to 0.0 (4 cells) at whole-transcriptome pooled N (totals 1.3M-3.4M), so p magnitude is uninformative here and the informative axis is `Effect change` (BWM `log2_OR` − control `log2_OR`); effects are small-to-moderate.

The largest BWM-vs-control divergences concentrate at site A: A:W (BWM `log2_OR`=-0.586 depleted at d10 vs control +0.088 enriched), A:N (BWM +0.250 vs control -0.129), A:Y (BWM -0.262 vs control +0.017), A:L (BWM -0.182 vs control +0.044), A:I (BWM +0.209 vs control -0.017) — each a direction split between conditions. Reported at equal billing (A.2.2), the largest shared-direction cells (both arms same sign) are E:P both depleted (BWM -0.387 / control -0.086), P:G both depleted (-0.225 / -0.052), E:S both enriched (+0.246 / +0.118), P:E both enriched (+0.208 / +0.101), A:M both enriched (+0.207 / +0.096).

## Top hits

`log2_OR` is the within-condition Fisher effect for the **day_10 vs day_0** contrast: positive = enriched at day_10 relative to day_0, negative = depleted. Each row pairs the BWM and control value for one (site, amino acid) cell; `Effect change` = BWM `log2_OR` − control `log2_OR` (large magnitude = the two conditions' d10-vs-d0 trajectories diverge). Rows are grouped by `site` in A → P → E order, then sorted by `Effect change` descending. Cells significant (FDR<0.05) in neither condition are omitted. Generated by `scripts/within_condition_sig_split.py`.

Every cell shown is FDR-significant in at least one arm at whole-transcriptome N, and the both-section cells overwhelmingly sit at p_adj << 1e-10. Per A.2.6/A.2.4 the large-N anti-conservatism alternative explanation applies to every such row symmetrically, so ranking is by `|Effect change|`, not by p. The `Flag` column carries only `low-count` (none fires at AA resolution).

### Significant in both conditions (n = 46 site x amino acid cells)

<details>
<summary>Full 46-cell both-conditions table (A / P / E, sorted by Effect change desc)</summary>

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | N (Asn) | +0.250 | -0.129 | +0.379 |  |
| A | I (Ile) | +0.209 | -0.017 | +0.226 |  |
| A | V (Val) | +0.089 | -0.057 | +0.146 |  |
| A | M (Met) | +0.207 | +0.096 | +0.111 |  |
| A | Q (Gln) | -0.103 | -0.179 | +0.077 |  |
| A | K (Lys) | -0.083 | -0.146 | +0.063 |  |
| A | E (Glu) | +0.154 | +0.097 | +0.057 |  |
| A | F (Phe) | +0.107 | +0.094 | +0.013 |  |
| A | S (Ser) | +0.063 | +0.083 | -0.020 |  |
| A | G (Gly) | -0.095 | -0.063 | -0.032 |  |
| A | H (His) | -0.094 | -0.038 | -0.056 |  |
| A | A (Ala) | -0.098 | +0.038 | -0.136 |  |
| A | C (Cys) | -0.046 | +0.164 | -0.211 |  |
| A | L (Leu) | -0.182 | +0.044 | -0.226 |  |
| A | Y (Tyr) | -0.262 | +0.017 | -0.279 |  |
| A | W (Trp) | -0.586 | +0.088 | -0.674 |  |
| P | L (Leu) | +0.093 | -0.104 | +0.197 |  |
| P | M (Met) | +0.103 | -0.066 | +0.170 |  |
| P | W (Trp) | +0.035 | -0.107 | +0.141 |  |
| P | E (Glu) | +0.208 | +0.101 | +0.106 |  |
| P | V (Val) | +0.014 | -0.052 | +0.065 |  |
| P | I (Ile) | -0.084 | -0.137 | +0.054 |  |
| P | K (Lys) | -0.119 | -0.171 | +0.052 |  |
| P | S (Ser) | +0.084 | +0.044 | +0.040 |  |
| P | P (Pro) | -0.024 | -0.045 | +0.022 |  |
| P | Q (Gln) | -0.032 | -0.044 | +0.012 |  |
| P | H (His) | -0.034 | -0.042 | +0.008 |  |
| P | Y (Tyr) | -0.067 | -0.026 | -0.042 |  |
| P | A (Ala) | -0.014 | +0.034 | -0.048 |  |
| P | C (Cys) | +0.112 | +0.175 | -0.064 |  |
| P | R (Arg) | -0.034 | +0.055 | -0.090 |  |
| P | D (Asp) | +0.061 | +0.223 | -0.162 |  |
| P | G (Gly) | -0.225 | -0.052 | -0.173 |  |
| E | S (Ser) | +0.246 | +0.118 | +0.128 |  |
| E | M (Met) | +0.134 | +0.053 | +0.082 |  |
| E | G (Gly) | -0.069 | -0.133 | +0.064 |  |
| E | C (Cys) | +0.110 | +0.052 | +0.058 |  |
| E | Y (Tyr) | -0.050 | -0.083 | +0.033 |  |
| E | E (Glu) | +0.072 | +0.046 | +0.026 |  |
| E | L (Leu) | +0.093 | +0.076 | +0.017 |  |
| E | D (Asp) | +0.066 | +0.054 | +0.013 |  |
| E | T (Thr) | +0.031 | +0.021 | +0.010 |  |
| E | V (Val) | -0.038 | -0.017 | -0.022 |  |
| E | I (Ile) | -0.070 | -0.034 | -0.036 |  |
| E | K (Lys) | -0.129 | -0.066 | -0.063 |  |
| E | P (Pro) | -0.387 | -0.086 | -0.302 |  |

</details>

### Significant in BWM only (n = 8 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | D (Asp) | +0.229 | +0.003 | +0.226 |  |
| A | P (Pro) | -0.231 | -0.003 | -0.228 |  |
| P | T (Thr) | +0.044 | +0.008 | +0.037 |  |
| E | H (His) | +0.125 | -0.020 | +0.145 |  |
| E | W (Trp) | +0.083 | -0.001 | +0.084 |  |
| E | Q (Gln) | +0.076 | +0.001 | +0.075 |  |
| E | N (Asn) | +0.018 | +0.009 | +0.009 |  |
| E | R (Arg) | -0.015 | +0.007 | -0.021 |  |

### Significant in control only (n = 4 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | R (Arg) | +0.007 | +0.073 | -0.066 |  |
| P | F (Phe) | +0.011 | +0.050 | -0.039 |  |
| P | N (Asn) | -0.013 | +0.083 | -0.096 |  |
| E | A (Ala) | -0.010 | +0.015 | -0.024 |  |

## Numbers at a glance
- `n_tests`: 120 (60 per condition; 3 sites x 20 AAs each)
- `n_significant` (adjusted-p < 0.05): 104 (BWM 54/60, control 50/60)
- `n_significant` (adjusted-p < 0.10): 108
- `min adjusted-p`: 0.0 (underflow; 4 cells at exactly 0.0). Smallest non-zero p_adj is 2.087e-318 (E,BWM,S).
- `p_floor`: n/a — Fisher with pooled N in the millions has no meaningful floor; the dominant statistical-design concern is `large-N-Fisher-anticonservative` (family-wide).
- Per (condition, site) sig at FDR<0.05: BWM A 18/20, P 18/20, E 18/20; control A 17/20, P 19/20, E 14/20.
- Section split: 46 cells sig in both conditions, 8 BWM-only, 4 control-only (58 of 60 site x aa cells sig in ≥1 condition).

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 of (aa_count, total − aa_count) at day_10 vs day_0 within each condition; user confirmed. `global_codon_occ_stats.py` (Analysis 3b) pools the replicate counts per (condition, timepoint) before the test and applies BH-FDR within each (condition, site) family of 20 AAs (corrected by `merge_global_occupancy_analysis.py`); each of the 6 (condition, site) sub-families is corrected independently, not across the full 120-row file. The effect is reported as `log2_OR` (log2 of the CSV `odds_ratio` column): >0 = enriched at day_10 relative to day_0 within that condition, <0 = depleted. The test does **not** answer whether BWM and control move in the same direction (that BWM-vs-control comparison is the `control-vs-BWM-divergent-direction` reading) and does not test whether occupancy is enriched vs the transcriptomic background (that is the `within_condition_binomial` family).

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* — the 2 biological replicates per (condition, timepoint) are summed into the 2x2 before Fisher; per-replicate variance is not in the test statistic, so p-values are anti-conservative. (Inherited — see `_INDEX.md`.)
- **large-N-Fisher-anticonservative** *(family-wide)* — pooled totals are whole-transcriptome footprint counts (day_10 ~1.56M BWM / ~1.26M control; day_0 ~2.11M BWM / ~3.37M control). Fisher returns vanishing p (here 4 cells underflow to 0.0) for tiny relative deviations; `log2_OR` is the primary effect column, p magnitude is not a ranking axis. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* — BH is applied within each of the 6 (condition, site) families of 20 AAs; `p_adj` means "corrected within this sub-family", not across the 120-row file. (Inherited.)
- **within-condition-clean** *(family-wide)* — no condition or timepoint pooling across the contrast; each Fisher is held within one (condition, contrast) cell. Structurally cleaner than the between-condition / between-timepoint Wilcoxon families. (Inherited.)
- **control-vs-BWM-divergent-direction** *(per-CSV)* — flagged for Chumeng's reconciliation. Direction-split cells (both arms meaningful, opposite signs) cluster at site A: A:N (BWM `log2_OR`=+0.250 / control -0.129), A:W (BWM -0.586 / control +0.088), A:Y (BWM -0.262 / control +0.017), A:L (BWM -0.182 / control +0.044), A:I (BWM +0.209 / control -0.017), A:C (BWM -0.046 / control +0.164). These are the design-target reading of the within-condition contrast.
- **near-universal-sig-large-N** *(per-CSV)* — 104/120 tests clear FDR<0.05, so "both / BWM-only / control-only" section membership is weakly discriminating; the real ranking axis is `|Effect change|`. Sharpens the family-wide large-N caveat for this design.

### Considered but not applicable
- **rare-aa-low-count** — every (condition, site, aa) cell has both day_10 and day_0 counts well above the `within_condition_sig_split` thresholds (BWM<100 / control<200); the smallest arm is control P:W day_10=9188. No low-count flag fires at AA resolution (the codon sister differs — in-frame stop TGA trips it).

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs to reconcile this file against: `codon_timepoint_fisher_within_condition_d10_vs_d0` (codon refinement of the same contrast), and the d10_vs_d5 + d5_vs_d0 contrasts at the same AA resolution (for monotonicity / biphasic checks). See the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- Falsifier: do the site-A direction splits (A:N BWM↑/control↓, A:W BWM↓/control↑, A:Y BWM↓/control↑) reappear at the codon level localised to one synonym (codon-usage signature) or spread across synonyms (amino-acid-level)? If they vanish at codon resolution they are AA-aggregation artefacts.
- Falsifier for `near-universal-sig-large-N`: at this N nearly every cell is FDR-significant. Does the same near-universal significance hold in the d10_vs_d5 and d5_vs_d0 AA files? If yes, section membership is a property of the large N, not contrast-specific biology.
- Falsifier for the shared-direction cells (E:P both↓, E:S both↑, P:E both↑): do these reappear as stable baselines across all 6 groups in `within_condition_binomial`, or as group-variable cells? A baseline that both conditions carry is not a d10-vs-d0 effect.
- Falsifier on direction: does any site-A divergence (e.g. A:N) reproduce with consistent sign in `per_timepoint_fisher` (BWM-vs-control at a fixed timepoint)? A divergence seen only in the within-condition contrast but not in the per-timepoint comparison would need a per-condition-trajectory explanation rather than a fixed-timepoint condition difference.
