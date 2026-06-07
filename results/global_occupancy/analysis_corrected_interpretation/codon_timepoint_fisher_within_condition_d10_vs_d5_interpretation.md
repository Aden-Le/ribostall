---
input_csv: results/global_occupancy/analysis_corrected/codon_timepoint_fisher_within_condition_d10_vs_d5.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), within-condition day_10-vs-day_5 contrast per (condition, site, codon); BH-FDR within each (condition, site) family of ~62 codons
test_type_source: user-confirmed
n_tests: 372
n_significant_fdr05: 297
n_significant_fdr10: 313
min_p_adj: 1.273e-313
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Sense codons where BWM and control split direction in the day_10-vs-day_5 contrast (design-target reading), reported alongside shared-direction cells per A.2.2. See the file's Caveats section for the listed cells. User-confirmed."}
  - {label: "near-universal-sig-large-N", proposed_by: dylan, status: confirmed, why: "297/372 tests clear FDR<0.05 at whole-transcriptome pooled N; both/BWM-only/control-only section membership is weakly discriminating, so rank by |Effect change|, not by section or p. User-confirmed."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "The within_condition_sig_split low-count flag (BWM<50 / control<50) fires only on the in-frame stop codon TGA: A:TGA, P:TGA and E:TGA (all low-count control). All sense codons clear the threshold, so this caveat and stop-codon-instability mark the same TGA cells. User-confirmed."}
  - {label: "stop-codon-instability", proposed_by: dylan, status: confirmed, why: "The in-frame stop TGA carries the largest |log2_OR| swings in the file (A:TGA (BWM log2_OR=+0.536 vs control -1.702), P:TGA (BWM +0.633 vs control -1.284), E:TGA (BWM +0.950 vs control -1.334)) but rests on small counts in at least one arm; its magnitude is unstable and must not anchor the read. User-confirmed."}
caveats_considered: []
headline: "297/372 within-condition Fisher tests sig at FDR<0.05 (BWM 140/186, control 157/186; 117 cells sig in both, 23 BWM-only, 40 control-only); min p_adj 1.273e-313, no exact-zero underflow, at whole-transcriptome N. In-frame stop TGA carries the largest swings at all three sites (e.g. E:TGA BWM log2_OR=+0.950 vs control -1.334) but is low-count/unstable. Largest stable sense divergence A:CCC (BWM -0.195 vs control +0.533); site-A Pro codons show control enriched at d10 while BWM stays flat (A:CCA -0.080/+0.341, A:CCT -0.077/+0.324). Largest shared-direction cells: A:GGG both enriched (+0.292/+0.345), E:GGC both enriched (+0.321/+0.101)."
user_directives:
  - "(per-CSV triage) 'Confirm test type for the family?' -> 'Fisher's exact, BH per (condition, site)' (applies to all 6 files)."
  - "(per-CSV triage) 'Per-CSV caveats beyond the 4 locked family caveats?' -> confirmed control-vs-BWM-divergent-direction, near-universal-sig-large-N, rare-codon-low-count, stop-codon-instability."
  - "(per-CSV triage) 'How firmly should this family read?' -> 'Firm' (significant cells read as established; still rank by |Effect change|, not p)."
  - "(per-CSV triage) Top-hits table source -> user authorised running scripts/within_condition_sig_split.py (codon files with --rare-bwm-threshold 50 --rare-control-threshold 50) to generate the three-section paired tables transcribed below."
---

# Interpretation -- codon_timepoint_fisher_within_condition_d10_vs_d5

> Source: `results/global_occupancy/analysis_corrected/codon_timepoint_fisher_within_condition_d10_vs_d5.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), within-condition day_10-vs-day_5 contrast per (condition, site, codon); BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (per-CSV triage) "Confirm test type for the family?" -> "Fisher's exact, BH per (condition, site)" (confirmed for all 6 family files).
- (per-CSV triage) "Per-CSV caveats beyond the 4 locked family caveats?" -> confirmed `control-vs-BWM-divergent-direction`, `near-universal-sig-large-N`, `rare-codon-low-count`, `stop-codon-instability` (the last two for codon files only).
- (per-CSV triage) "How firmly should this family read?" -> "Firm". Significant cells read as established; the A.2.x large-N discipline still applies -- rank by `|Effect change|`, not p.
- (per-CSV triage) Top-hits table source -> user authorised running `scripts/within_condition_sig_split.py` (a display-only reshape of the existing `odds_ratio`/`p_adj` columns -- no statistics re-run; codon files use `--rare-bwm-threshold 50 --rare-control-threshold 50`) to generate the three-section paired tables below.

## Headline
At codon-level within-condition Fisher d10 vs d5, 297/372 tests clear FDR<0.05 (BWM 140/186, control 157/186): 117 site x codon cells significant in both conditions, 23 BWM-only, 40 control-only. Minimum adjusted p is 1.273e-313 with no exact-zero underflow (unlike the other codon files in this family) at whole-transcriptome pooled N (totals 1.3M-2.4M); the informative axis is `Effect change`, not p.

The in-frame stop codon TGA carries the largest swings in the file (A:TGA BWM `log2_OR`=+0.536 / control -1.702; P:TGA BWM +0.633 / control -1.284; E:TGA BWM +0.950 / control -1.334) but rests on small control counts (flagged `low-count`, unstable -- see caveats). Among stable sense codons the largest divergence is A:CCC (BWM -0.195 / control +0.533); the site-A Pro codons consistently show control enriched at d10 vs d5 while BWM stays flat-to-down (A:CCA BWM -0.080 / control +0.341, A:CCT BWM -0.077 / control +0.324, A:CCG BWM -0.013 / control +0.451). Reported at equal billing (A.2.2), the largest shared-direction cells (both arms same sign) are A:GGG both enriched (+0.292 / +0.345), P:GGG both enriched (+0.347 / +0.241), E:GGC both enriched (+0.321 / +0.101).

## Top hits

`log2_OR` is the within-condition Fisher effect for the **day_10 vs day_5** contrast: positive = enriched at day_10 relative to day_5, negative = depleted. Each row pairs the BWM and control value for one (site, codon) cell; `Effect change` = BWM `log2_OR` - control `log2_OR` (large magnitude = the two conditions' day_10-vs-day_5 trajectories diverge). Rows are grouped by `site` in A -> P -> E order, then sorted by `Effect change` descending. Cells significant (FDR<0.05) in neither condition are omitted. Generated by `scripts/within_condition_sig_split.py`.

Every cell shown is FDR-significant in at least one arm at whole-transcriptome N (most both-section cells sit at p_adj far below 1e-10), so per A.2.6/A.2.4 the large-N anti-conservatism alternative explanation applies symmetrically to every such row and ranking is by `|Effect change|`, not p. The `Flag` column carries `low-count`; in this family it fires only on the in-frame stop codon TGA (see the `rare-codon-low-count` / `stop-codon-instability` caveats).

### Significant in both conditions (n = 117 site x codon cells)

<details>
<summary>Full 117-cell both-conditions table (A / P / E, sorted by Effect change desc)</summary>

| Site | Codon | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | TGA | +0.536 | -1.702 | +2.238 | low-count (C) |
| A | ATT | +0.067 | -0.236 | +0.303 |  |
| A | AGT | +0.108 | -0.163 | +0.270 |  |
| A | AGA | +0.071 | -0.171 | +0.243 |  |
| A | GAA | +0.132 | -0.101 | +0.233 |  |
| A | TGT | +0.097 | -0.126 | +0.224 |  |
| A | ATG | +0.021 | -0.134 | +0.155 |  |
| A | AAG | -0.054 | -0.208 | +0.154 |  |
| A | ATA | -0.095 | -0.243 | +0.147 |  |
| A | TCA | +0.071 | -0.049 | +0.120 |  |
| A | AGC | +0.057 | -0.063 | +0.120 |  |
| A | GAG | +0.150 | +0.032 | +0.118 |  |
| A | GTT | +0.023 | -0.092 | +0.115 |  |
| A | ATC | -0.025 | -0.131 | +0.106 |  |
| A | CAA | -0.118 | -0.214 | +0.097 |  |
| A | GAT | -0.040 | -0.118 | +0.078 |  |
| A | ACA | -0.065 | -0.102 | +0.036 |  |
| A | ACT | -0.053 | -0.086 | +0.032 |  |
| A | TAT | -0.130 | -0.162 | +0.032 |  |
| A | GGA | +0.167 | +0.154 | +0.012 |  |
| A | CAT | -0.045 | -0.031 | -0.014 |  |
| A | TCG | +0.125 | +0.173 | -0.047 |  |
| A | GGG | +0.292 | +0.345 | -0.053 |  |
| A | GAC | -0.048 | +0.033 | -0.081 |  |
| A | ACC | -0.091 | +0.037 | -0.128 |  |
| A | TTG | -0.035 | +0.122 | -0.157 |  |
| A | CAC | -0.098 | +0.075 | -0.173 |  |
| A | GCT | -0.024 | +0.236 | -0.260 |  |
| A | CTT | -0.154 | +0.124 | -0.278 |  |
| A | TAC | -0.087 | +0.198 | -0.285 |  |
| A | GCG | +0.081 | +0.369 | -0.289 |  |
| A | TCC | -0.068 | +0.237 | -0.305 |  |
| A | CTC | -0.123 | +0.191 | -0.314 |  |
| A | GGC | +0.110 | +0.433 | -0.322 |  |
| A | CGG | +0.088 | +0.411 | -0.323 |  |
| A | GCC | -0.047 | +0.309 | -0.355 |  |
| A | TGG | -0.048 | +0.342 | -0.390 |  |
| A | CCT | -0.077 | +0.324 | -0.401 |  |
| A | CCA | -0.080 | +0.341 | -0.421 |  |
| A | CCC | -0.195 | +0.533 | -0.727 |  |
| P | TGA | +0.633 | -1.284 | +1.917 | low-count (C) |
| P | TTA | +0.056 | -0.409 | +0.464 |  |
| P | GTT | -0.042 | -0.266 | +0.224 |  |
| P | GTC | +0.061 | -0.162 | +0.223 |  |
| P | TCG | +0.112 | -0.085 | +0.197 |  |
| P | GCG | +0.227 | +0.054 | +0.172 |  |
| P | GTA | +0.071 | -0.094 | +0.165 |  |
| P | TTG | -0.034 | -0.185 | +0.150 |  |
| P | ATT | -0.176 | -0.299 | +0.122 |  |
| P | GGC | +0.224 | +0.114 | +0.110 |  |
| P | GGG | +0.347 | +0.241 | +0.106 |  |
| P | GAA | +0.083 | -0.019 | +0.102 |  |
| P | GAG | +0.204 | +0.103 | +0.101 |  |
| P | GGT | +0.167 | +0.069 | +0.098 |  |
| P | GAT | +0.137 | +0.054 | +0.083 |  |
| P | GCA | +0.187 | +0.116 | +0.071 |  |
| P | ATC | -0.143 | -0.203 | +0.060 |  |
| P | TTC | -0.046 | -0.101 | +0.055 |  |
| P | CCT | +0.189 | +0.139 | +0.050 |  |
| P | TCT | -0.088 | -0.138 | +0.050 |  |
| P | CTT | -0.169 | -0.204 | +0.035 |  |
| P | CGG | +0.304 | +0.292 | +0.012 |  |
| P | AAT | -0.073 | -0.084 | +0.011 |  |
| P | CGA | +0.193 | +0.197 | -0.003 |  |
| P | CTC | -0.053 | -0.034 | -0.019 |  |
| P | GAC | +0.073 | +0.098 | -0.025 |  |
| P | ATG | -0.051 | -0.026 | -0.026 |  |
| P | CCG | +0.248 | +0.307 | -0.058 |  |
| P | GGA | +0.125 | +0.202 | -0.077 |  |
| P | CCC | +0.206 | +0.289 | -0.083 |  |
| P | TAT | -0.121 | -0.036 | -0.085 |  |
| P | CGT | -0.027 | +0.074 | -0.102 |  |
| P | AGG | +0.231 | +0.356 | -0.126 |  |
| P | CAG | +0.061 | +0.226 | -0.165 |  |
| P | CAA | -0.091 | +0.079 | -0.170 |  |
| P | TAC | -0.148 | +0.055 | -0.203 |  |
| P | ACC | -0.083 | +0.120 | -0.203 |  |
| P | AGA | -0.099 | +0.119 | -0.218 |  |
| P | CCA | -0.051 | +0.167 | -0.218 |  |
| P | AAG | -0.168 | +0.072 | -0.240 |  |
| P | CAC | -0.150 | +0.140 | -0.290 |  |
| E | TGA | +0.950 | -1.334 | +2.284 | low-count (C) |
| E | GGT | +0.248 | -0.108 | +0.356 |  |
| E | AGT | +0.207 | -0.145 | +0.353 |  |
| E | TGG | +0.141 | -0.152 | +0.293 |  |
| E | AGG | +0.122 | -0.150 | +0.272 |  |
| E | ACG | +0.150 | -0.099 | +0.249 |  |
| E | CAG | +0.091 | -0.143 | +0.234 |  |
| E | ATG | +0.051 | -0.183 | +0.234 |  |
| E | GGC | +0.321 | +0.101 | +0.220 |  |
| E | TGT | +0.060 | -0.145 | +0.205 |  |
| E | TCG | +0.124 | -0.069 | +0.193 |  |
| E | GTG | +0.067 | -0.119 | +0.185 |  |
| E | AAG | -0.065 | -0.250 | +0.185 |  |
| E | CTG | +0.090 | -0.056 | +0.146 |  |
| E | AGA | -0.038 | -0.160 | +0.122 |  |
| E | CAT | +0.047 | -0.064 | +0.111 |  |
| E | GAT | +0.030 | -0.072 | +0.102 |  |
| E | CTA | +0.066 | +0.063 | +0.003 |  |
| E | AAA | -0.021 | -0.021 | +0.000 |  |
| E | ATT | -0.081 | -0.080 | -0.002 |  |
| E | CCG | +0.133 | +0.140 | -0.007 |  |
| E | GCA | +0.078 | +0.118 | -0.040 |  |
| E | TAT | -0.123 | -0.040 | -0.083 |  |
| E | GAC | +0.032 | +0.127 | -0.095 |  |
| E | CTT | -0.049 | +0.052 | -0.101 |  |
| E | CGC | -0.038 | +0.103 | -0.141 |  |
| E | CTC | +0.081 | +0.254 | -0.173 |  |
| E | CCT | +0.105 | +0.301 | -0.196 |  |
| E | CAC | -0.067 | +0.133 | -0.200 |  |
| E | ATC | -0.077 | +0.137 | -0.214 |  |
| E | ACT | -0.066 | +0.158 | -0.224 |  |
| E | TTC | -0.085 | +0.141 | -0.227 |  |
| E | TAC | -0.165 | +0.125 | -0.290 |  |
| E | TCC | -0.058 | +0.243 | -0.301 |  |
| E | CCA | -0.142 | +0.258 | -0.400 |  |
| E | ACC | -0.129 | +0.275 | -0.404 |  |

</details>

### Significant in BWM only (n = 23 site x codon cells)

| Site | Codon | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | AGG | +0.206 | +0.034 | +0.172 |  |
| A | GGT | +0.073 | -0.013 | +0.086 |  |
| A | TTA | -0.040 | -0.032 | -0.008 |  |
| A | GTA | -0.061 | -0.001 | -0.060 |  |
| P | ACG | +0.158 | +0.005 | +0.153 |  |
| P | GTG | +0.124 | -0.020 | +0.144 |  |
| P | TGT | +0.068 | +0.008 | +0.060 |  |
| P | CTA | -0.058 | -0.055 | -0.003 |  |
| P | ACT | -0.052 | +0.013 | -0.065 |  |
| P | TCC | -0.052 | +0.034 | -0.086 |  |
| P | AAC | -0.117 | -0.007 | -0.109 |  |
| P | CAT | -0.094 | +0.023 | -0.118 |  |
| E | GGG | +0.306 | -0.016 | +0.322 |  |
| E | CGG | +0.253 | -0.046 | +0.299 |  |
| E | GCG | +0.218 | +0.029 | +0.189 |  |
| E | CGA | +0.177 | -0.007 | +0.183 |  |
| E | AGC | +0.071 | -0.035 | +0.106 |  |
| E | GGA | +0.075 | -0.014 | +0.089 |  |
| E | GAA | +0.032 | +0.002 | +0.029 |  |
| E | TTT | -0.041 | -0.026 | -0.015 |  |
| E | CAA | -0.029 | -0.006 | -0.023 |  |
| E | GTT | -0.048 | -0.007 | -0.041 |  |
| E | AAC | -0.096 | -0.011 | -0.085 |  |

### Significant in control only (n = 40 site x codon cells)

| Site | Codon | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | AAT | +0.007 | -0.364 | +0.371 |  |
| A | AAC | -0.011 | -0.325 | +0.314 |  |
| A | AAA | -0.007 | -0.319 | +0.312 |  |
| A | TTT | +0.024 | -0.033 | +0.057 |  |
| A | TTC | +0.001 | +0.078 | -0.077 |  |
| A | CGA | +0.003 | +0.121 | -0.119 |  |
| A | CGT | -0.009 | +0.166 | -0.174 |  |
| A | GTG | -0.011 | +0.180 | -0.191 |  |
| A | CTA | -0.027 | +0.194 | -0.221 |  |
| A | TGC | +0.035 | +0.256 | -0.222 |  |
| A | GCA | +0.016 | +0.242 | -0.225 |  |
| A | CTG | +0.033 | +0.343 | -0.310 |  |
| A | CGC | -0.029 | +0.334 | -0.363 |  |
| A | CCG | -0.013 | +0.451 | -0.464 |  |
| P | TTT | +0.012 | -0.251 | +0.263 |  |
| P | ATA | +0.026 | -0.183 | +0.209 |  |
| P | TCA | +0.012 | -0.080 | +0.092 |  |
| P | CTG | +0.011 | -0.045 | +0.056 |  |
| P | AAA | +0.002 | -0.024 | +0.026 |  |
| P | AGC | -0.037 | -0.053 | +0.016 |  |
| P | AGT | +0.029 | +0.040 | -0.011 |  |
| P | GCT | +0.008 | +0.025 | -0.017 |  |
| P | TGG | +0.022 | +0.100 | -0.078 |  |
| P | GCC | +0.017 | +0.126 | -0.109 |  |
| P | TGC | +0.004 | +0.128 | -0.125 |  |
| P | CGC | -0.028 | +0.186 | -0.214 |  |
| E | TTG | +0.015 | -0.197 | +0.212 |  |
| E | AAT | +0.014 | -0.196 | +0.210 |  |
| E | GAG | -0.004 | -0.122 | +0.117 |  |
| E | TTA | +0.017 | -0.087 | +0.104 |  |
| E | ATA | +0.026 | -0.077 | +0.103 |  |
| E | CGT | +0.025 | -0.077 | +0.102 |  |
| E | TCA | +0.023 | +0.055 | -0.032 |  |
| E | TGC | +0.030 | +0.066 | -0.036 |  |
| E | ACA | +0.010 | +0.051 | -0.041 |  |
| E | TCT | +0.000 | +0.077 | -0.076 |  |
| E | GCT | +0.001 | +0.187 | -0.186 |  |
| E | GTC | -0.018 | +0.215 | -0.234 |  |
| E | CCC | +0.059 | +0.354 | -0.295 |  |
| E | GCC | +0.002 | +0.378 | -0.377 |  |

## Numbers at a glance
- `n_tests`: 372 (186 per condition; 3 sites x ~62 codons each)
- `n_significant` (adjusted-p < 0.05): 297 (BWM 140/186, control 157/186)
- `n_significant` (adjusted-p < 0.10): 313
- `min adjusted-p`: 1.273e-313 (no exact-zero underflow). Smallest non-zero p_adj is 1.273225e-313 (E,control,AAG).
- `p_floor`: n/a -- Fisher with pooled N in the millions has no meaningful floor; the dominant concern is `large-N-Fisher-anticonservative` (family-wide).
- Per (condition, site) sig at FDR<0.05: BWM A 44/62, P 49/62, E 47/62; control A 54/62, P 53/62, E 50/62.
- Section split: 117 cells sig in both, 23 BWM-only, 40 control-only.

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 of (codon_count, total - codon_count) at day_10 vs day_5 within each condition; user confirmed. `global_codon_occ_stats.py` (Analysis 3b) pools the replicate counts per (condition, timepoint) before the test and applies BH-FDR within each (condition, site) family of ~62 codons (corrected by `merge_global_occupancy_analysis.py`); each of the 6 (condition, site) sub-families is corrected independently, not across the full 372-row file. Effect is reported as `log2_OR` (log2 of the CSV `odds_ratio`): >0 = enriched at day_10 relative to day_5 within that condition, <0 = depleted. The test does not compare BWM against control directly (that is the `control-vs-BWM-divergent-direction` reading) and does not test enrichment vs the transcriptomic background (the `within_condition_binomial` family). About 62 of 64 codons appear per (condition, site); the in-frame stop TGA is present, TAA/TAG are not retained in the significant set.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* -- the 2 biological replicates per (condition, timepoint) are summed into the 2x2 before Fisher; per-replicate variance is not in the test statistic, so p-values are anti-conservative. (Inherited -- see `_INDEX.md`.)
- **large-N-Fisher-anticonservative** *(family-wide)* -- pooled totals are whole-transcriptome footprint counts (day_10 ~1.56M BWM / ~1.26M control; day_5 ~2.01M BWM / ~2.41M control); Fisher returns vanishing p for tiny relative deviations. `log2_OR` is the primary effect column; p magnitude is not a ranking axis. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* -- BH is applied within each of the 6 (condition, site) families of ~62 codons; `p_adj` means corrected within this sub-family, not across the 372-row file. (Inherited.)
- **within-condition-clean** *(family-wide)* -- no condition or timepoint pooling across the contrast; structurally cleaner than the between-condition / between-timepoint Wilcoxon families. (Inherited.)
- **control-vs-BWM-divergent-direction** *(per-CSV)* -- flagged for Chumeng's reconciliation. Sense codons that split direction between conditions: A:CCC (BWM `log2_OR`=-0.195 / control +0.533), A:CCA (BWM -0.080 / control +0.341), A:CCT (BWM -0.077 / control +0.324), A:CCG (BWM -0.013 / control +0.451), A:GCC (BWM -0.047 / control +0.309). The design-target reading of the within-condition contrast.
- **near-universal-sig-large-N** *(per-CSV)* -- 297/372 tests clear FDR<0.05, so section membership is weakly discriminating; the ranking axis is `|Effect change|`.
- **rare-codon-low-count** *(per-CSV)* -- the `low-count` flag (BWM<50 / control<50) fires only on the in-frame stop TGA cells (A:TGA, P:TGA and E:TGA (all low-count control)); all sense codons clear the threshold. Marks the same cells as `stop-codon-instability`.
- **stop-codon-instability** *(per-CSV)* -- TGA carries the largest |log2_OR| swings in the file (A:TGA (BWM log2_OR=+0.536 vs control -1.702), P:TGA (BWM +0.633 vs control -1.284), E:TGA (BWM +0.950 vs control -1.334)) on small counts in at least one arm; magnitude is unstable and must not anchor the read.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` -- sister CSVs: `aa_timepoint_fisher_within_condition_d10_vs_d5` (the AA aggregate of this contrast), plus the other two codon contrasts (additivity / monotonicity: does d10_vs_d0 ~= d5_vs_d0 + d10_vs_d5 per cell?). See `## Joint-reading suggestions` in [`_INDEX.md`](_INDEX.md).
- Falsifier (synonymous split): for each AA-level divergence in the sister AA file, does the codon file show the signal carried by one synonym (codon-usage shift) or spread across synonyms (amino-acid-level)? e.g. do the site-A Pro-codon divergences (CCA/CCT/CCG, control enriched / BWM flat) localise consistently across the Pro family?
- Falsifier (stop TGA): TGA carries the largest swings here but is low-count/unstable. Does it reach a comparable extreme in any large-N file where its counts are higher (`per_timepoint_fisher_codon`), or only in these within-condition contrasts? If only here, treat it as sampling noise on a rare feature.
- Falsifier (shared-direction baselines): do the large shared-direction codon cells reappear as stable baselines across all 6 groups in `codon_within_condition_binomial` (then they are baseline composition both conditions carry, not a day_10-vs-day_5 effect), or as group-variable cells?
