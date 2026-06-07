---
input_csv: results/global_occupancy/analysis_corrected/codon_timepoint_fisher_within_condition_d5_vs_d0.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), within-condition day_5-vs-day_0 contrast per (condition, site, codon); BH-FDR within each (condition, site) family of ~62 codons
test_type_source: user-confirmed
n_tests: 372
n_significant_fdr05: 329
n_significant_fdr10: 335
min_p_adj: 0.0
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Sense codons where BWM and control split direction in the day_5-vs-day_0 contrast (design-target reading), reported alongside shared-direction cells per A.2.2. See the file's Caveats section for the listed cells. User-confirmed."}
  - {label: "near-universal-sig-large-N", proposed_by: dylan, status: confirmed, why: "329/372 tests clear FDR<0.05 at whole-transcriptome pooled N; both/BWM-only/control-only section membership is weakly discriminating, so rank by |Effect change|, not by section or p. User-confirmed."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "The within_condition_sig_split low-count flag (BWM<50 / control<50) fires only on the in-frame stop codon TGA: P:TGA and E:TGA (low-count BWM); A:TGA clears the threshold and is not flagged. All sense codons clear the threshold, so this caveat and stop-codon-instability mark the same TGA cells. User-confirmed."}
  - {label: "stop-codon-instability", proposed_by: dylan, status: confirmed, why: "The in-frame stop TGA carries the largest |log2_OR| swings in the file (E:TGA (BWM log2_OR=+1.195 vs control +0.701), P:TGA (BWM +1.111 vs control +0.909), A:TGA (BWM +0.503 vs control +0.466)) but rests on small counts in at least one arm; its magnitude is unstable and must not anchor the read. User-confirmed."}
caveats_considered: []
headline: "329/372 within-condition Fisher tests sig at FDR<0.05 (BWM 154/186, control 175/186; 144 cells sig in both, 10 BWM-only, 31 control-only) -- the most coordinated codon file; min p_adj underflows to 0.0 (20 cells) at whole-transcriptome N. In-frame stop TGA is enriched at d5 in both conditions (E:TGA +1.195/+0.701, P:TGA +1.111/+0.909, low-count BWM). Largest sense divergences: A:AGG (BWM -0.202 vs control +0.364), E:AGG (-0.137 vs +0.394), E:TCC (+0.212 vs -0.296). Largest shared-direction cells: A:GGC both depleted (-0.469/-0.244), A:CCA both depleted (-0.169/-0.490), A:GAT both enriched (+0.347/+0.170)."
user_directives:
  - "(per-CSV triage) 'Confirm test type for the family?' -> 'Fisher's exact, BH per (condition, site)' (applies to all 6 files)."
  - "(per-CSV triage) 'Per-CSV caveats beyond the 4 locked family caveats?' -> confirmed control-vs-BWM-divergent-direction, near-universal-sig-large-N, rare-codon-low-count, stop-codon-instability."
  - "(per-CSV triage) 'How firmly should this family read?' -> 'Firm' (significant cells read as established; still rank by |Effect change|, not p)."
  - "(per-CSV triage) Top-hits table source -> user authorised running scripts/within_condition_sig_split.py (codon files with --rare-bwm-threshold 50 --rare-control-threshold 50) to generate the three-section paired tables transcribed below."
---

# Interpretation -- codon_timepoint_fisher_within_condition_d5_vs_d0

> Source: `results/global_occupancy/analysis_corrected/codon_timepoint_fisher_within_condition_d5_vs_d0.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), within-condition day_5-vs-day_0 contrast per (condition, site, codon); BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (per-CSV triage) "Confirm test type for the family?" -> "Fisher's exact, BH per (condition, site)" (confirmed for all 6 family files).
- (per-CSV triage) "Per-CSV caveats beyond the 4 locked family caveats?" -> confirmed `control-vs-BWM-divergent-direction`, `near-universal-sig-large-N`, `rare-codon-low-count`, `stop-codon-instability` (the last two for codon files only).
- (per-CSV triage) "How firmly should this family read?" -> "Firm". Significant cells read as established; the A.2.x large-N discipline still applies -- rank by `|Effect change|`, not p.
- (per-CSV triage) Top-hits table source -> user authorised running `scripts/within_condition_sig_split.py` (a display-only reshape of the existing `odds_ratio`/`p_adj` columns -- no statistics re-run; codon files use `--rare-bwm-threshold 50 --rare-control-threshold 50`) to generate the three-section paired tables below.

## Headline
At codon-level within-condition Fisher d5 vs d0, 329/372 tests clear FDR<0.05 (BWM 154/186, control 175/186) -- the most of any codon file in the family: 144 site x codon cells significant in both conditions, 10 BWM-only, 31 control-only. Minimum adjusted p underflows to 0.0 (20 cells) at whole-transcriptome pooled N (totals 2.0M-3.4M); the informative axis is `Effect change`, not p.

The in-frame stop codon TGA is enriched at d5 relative to d0 in both conditions (E:TGA BWM `log2_OR`=+1.195 / control +0.701; P:TGA BWM +1.111 / control +0.909; A:TGA BWM +0.503 / control +0.466), with the P/E cells flagged `low-count` in BWM (unstable -- see caveats). Like its AA sister, this contrast is more coordinated than d10_vs_d0 / d10_vs_d5: large shared-direction cells include A:GGC both depleted (BWM -0.469 / control -0.244), A:CCA both depleted (-0.169 / -0.490), P:GGA both depleted (-0.399 / -0.311), A:GAT both enriched (+0.347 / +0.170), A:AAT both enriched (+0.298 / +0.329), A:GTT both enriched (+0.278 / +0.042). Reported at equal billing (A.2.2), the largest BWM-vs-control divergences are A:AGG (BWM -0.202 / control +0.364), E:AGG (BWM -0.137 / control +0.394), E:TCC (BWM +0.212 / control -0.296), A:ACC (BWM +0.144 / control -0.287).

## Top hits

`log2_OR` is the within-condition Fisher effect for the **day_5 vs day_0** contrast: positive = enriched at day_5 relative to day_0, negative = depleted. Each row pairs the BWM and control value for one (site, codon) cell; `Effect change` = BWM `log2_OR` - control `log2_OR` (large magnitude = the two conditions' day_5-vs-day_0 trajectories diverge). Rows are grouped by `site` in A -> P -> E order, then sorted by `Effect change` descending. Cells significant (FDR<0.05) in neither condition are omitted. Generated by `scripts/within_condition_sig_split.py`.

Every cell shown is FDR-significant in at least one arm at whole-transcriptome N (most both-section cells sit at p_adj far below 1e-10), so per A.2.6/A.2.4 the large-N anti-conservatism alternative explanation applies symmetrically to every such row and ranking is by `|Effect change|`, not p. The `Flag` column carries `low-count`; in this family it fires only on the in-frame stop codon TGA (see the `rare-codon-low-count` / `stop-codon-instability` caveats).

### Significant in both conditions (n = 144 site x codon cells)

<details>
<summary>Full 144-cell both-conditions table (A / P / E, sorted by Effect change desc)</summary>

| Site | Codon | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | ACC | +0.144 | -0.287 | +0.431 |  |
| A | GTC | +0.186 | -0.149 | +0.336 |  |
| A | CCA | -0.169 | -0.490 | +0.320 |  |
| A | TCC | -0.030 | -0.331 | +0.302 |  |
| A | GCT | +0.053 | -0.213 | +0.266 |  |
| A | CGT | +0.064 | -0.194 | +0.259 |  |
| A | CTT | +0.025 | -0.218 | +0.242 |  |
| A | GTT | +0.278 | +0.042 | +0.236 |  |
| A | GAC | +0.108 | -0.123 | +0.231 |  |
| A | GAT | +0.347 | +0.170 | +0.177 |  |
| A | ATC | +0.173 | +0.041 | +0.132 |  |
| A | CAC | -0.118 | -0.250 | +0.132 |  |
| A | AAC | +0.182 | +0.090 | +0.092 |  |
| A | GAG | +0.028 | -0.016 | +0.044 |  |
| A | TGA | +0.503 | +0.466 | +0.036 |  |
| A | ACT | +0.096 | +0.071 | +0.025 |  |
| A | GGA | -0.228 | -0.243 | +0.015 |  |
| A | ATT | +0.228 | +0.242 | -0.014 |  |
| A | CAA | -0.043 | -0.023 | -0.020 |  |
| A | AAT | +0.298 | +0.329 | -0.031 |  |
| A | TTT | +0.130 | +0.171 | -0.041 |  |
| A | CCT | -0.096 | -0.053 | -0.044 |  |
| A | ATG | +0.186 | +0.230 | -0.044 |  |
| A | AAG | -0.071 | -0.020 | -0.051 |  |
| A | CAT | +0.048 | +0.106 | -0.058 |  |
| A | CCC | -0.348 | -0.269 | -0.079 |  |
| A | TCA | +0.079 | +0.159 | -0.080 |  |
| A | GGG | -0.186 | -0.079 | -0.107 |  |
| A | TCG | -0.039 | +0.070 | -0.109 |  |
| A | TAC | -0.360 | -0.241 | -0.118 |  |
| A | GGT | -0.232 | -0.112 | -0.120 |  |
| A | TAT | +0.096 | +0.250 | -0.155 |  |
| A | AGA | -0.050 | +0.107 | -0.157 |  |
| A | GTG | -0.292 | -0.133 | -0.159 |  |
| A | TGC | -0.315 | -0.125 | -0.190 |  |
| A | TGT | +0.134 | +0.328 | -0.194 |  |
| A | TTG | -0.251 | -0.043 | -0.209 |  |
| A | CTG | -0.148 | +0.061 | -0.209 |  |
| A | AGT | +0.124 | +0.335 | -0.212 |  |
| A | CTA | -0.153 | +0.059 | -0.212 |  |
| A | GCA | -0.290 | -0.077 | -0.213 |  |
| A | GGC | -0.469 | -0.244 | -0.225 |  |
| A | CGG | -0.120 | +0.154 | -0.274 |  |
| A | CGA | -0.056 | +0.220 | -0.276 |  |
| A | TGG | -0.538 | -0.255 | -0.284 |  |
| A | ATA | +0.135 | +0.427 | -0.292 |  |
| A | AGG | -0.202 | +0.364 | -0.566 |  |
| P | ACC | +0.027 | -0.338 | +0.365 |  |
| P | CAC | +0.037 | -0.302 | +0.339 |  |
| P | AAG | -0.064 | -0.368 | +0.305 |  |
| P | TCC | +0.114 | -0.188 | +0.302 |  |
| P | CGC | +0.057 | -0.227 | +0.284 |  |
| P | CTC | +0.162 | -0.119 | +0.281 |  |
| P | CCA | -0.144 | -0.401 | +0.257 |  |
| P | GCC | -0.075 | -0.305 | +0.230 |  |
| P | TAC | +0.048 | -0.179 | +0.227 |  |
| P | TGA | +1.111 | +0.909 | +0.202 | low-count (BWM) |
| P | ATG | +0.154 | -0.041 | +0.195 |  |
| P | AGA | -0.037 | -0.198 | +0.161 |  |
| P | CAG | +0.030 | -0.130 | +0.160 |  |
| P | TTG | +0.174 | +0.075 | +0.099 |  |
| P | CAT | +0.116 | +0.024 | +0.092 |  |
| P | CTT | +0.132 | +0.054 | +0.078 |  |
| P | CGT | -0.035 | -0.107 | +0.073 |  |
| P | CTG | +0.194 | +0.143 | +0.051 |  |
| P | ATC | +0.022 | -0.020 | +0.042 |  |
| P | CCG | +0.086 | +0.061 | +0.025 |  |
| P | ACT | +0.044 | +0.045 | -0.001 |  |
| P | TTA | +0.332 | +0.335 | -0.003 |  |
| P | AAA | +0.044 | +0.047 | -0.004 |  |
| P | AGC | +0.067 | +0.074 | -0.007 |  |
| P | ATA | +0.252 | +0.260 | -0.009 |  |
| P | TAT | +0.086 | +0.102 | -0.017 |  |
| P | ACA | +0.054 | +0.083 | -0.029 |  |
| P | TCT | +0.065 | +0.106 | -0.041 |  |
| P | GAC | -0.115 | -0.071 | -0.044 |  |
| P | ACG | +0.183 | +0.238 | -0.056 |  |
| P | TCG | +0.182 | +0.240 | -0.058 |  |
| P | TCA | +0.090 | +0.147 | -0.058 |  |
| P | GAA | +0.040 | +0.121 | -0.082 |  |
| P | GGA | -0.399 | -0.311 | -0.088 |  |
| P | CGG | +0.098 | +0.196 | -0.098 |  |
| P | CCT | +0.087 | +0.218 | -0.131 |  |
| P | AAT | +0.159 | +0.290 | -0.131 |  |
| P | ATT | +0.087 | +0.223 | -0.137 |  |
| P | AGT | +0.034 | +0.186 | -0.152 |  |
| P | TGT | +0.129 | +0.288 | -0.159 |  |
| P | GTA | -0.040 | +0.153 | -0.194 |  |
| P | GCG | +0.059 | +0.259 | -0.200 |  |
| P | GGC | -0.161 | +0.045 | -0.205 |  |
| P | GCA | -0.204 | +0.064 | -0.268 |  |
| P | GAT | -0.035 | +0.243 | -0.278 |  |
| P | TTT | +0.087 | +0.425 | -0.338 |  |
| P | CGA | -0.230 | +0.213 | -0.443 |  |
| E | TCC | +0.212 | -0.296 | +0.508 |  |
| E | TGA | +1.195 | +0.701 | +0.494 | low-count (BWM) |
| E | GCC | -0.072 | -0.509 | +0.438 |  |
| E | TAC | +0.108 | -0.272 | +0.380 |  |
| E | GTC | +0.062 | -0.286 | +0.348 |  |
| E | CAC | +0.126 | -0.208 | +0.334 |  |
| E | CGC | +0.061 | -0.192 | +0.254 |  |
| E | TCT | +0.202 | -0.030 | +0.232 |  |
| E | CTC | +0.042 | -0.182 | +0.224 |  |
| E | TTC | +0.055 | -0.166 | +0.220 |  |
| E | TGC | +0.075 | -0.081 | +0.156 |  |
| E | GCT | -0.046 | -0.201 | +0.155 |  |
| E | TCA | +0.218 | +0.089 | +0.128 |  |
| E | CAA | +0.074 | -0.031 | +0.105 |  |
| E | AAC | +0.025 | -0.079 | +0.104 |  |
| E | CCA | -0.396 | -0.500 | +0.104 |  |
| E | GAC | -0.024 | -0.127 | +0.103 |  |
| E | ACA | +0.179 | +0.085 | +0.095 |  |
| E | TAT | +0.084 | +0.042 | +0.043 |  |
| E | CAT | +0.135 | +0.099 | +0.036 |  |
| E | GTT | -0.047 | -0.035 | -0.011 |  |
| E | GGA | -0.213 | -0.195 | -0.018 |  |
| E | GAG | +0.050 | +0.080 | -0.030 |  |
| E | TCG | +0.290 | +0.323 | -0.033 |  |
| E | TTA | +0.143 | +0.179 | -0.037 |  |
| E | CTA | +0.129 | +0.175 | -0.047 |  |
| E | TTT | +0.075 | +0.134 | -0.058 |  |
| E | GAA | +0.058 | +0.122 | -0.064 |  |
| E | CGT | -0.089 | -0.022 | -0.067 |  |
| E | TTG | +0.123 | +0.202 | -0.079 |  |
| E | AGC | +0.065 | +0.153 | -0.088 |  |
| E | GAT | +0.066 | +0.155 | -0.089 |  |
| E | ATA | +0.165 | +0.316 | -0.151 |  |
| E | ATG | +0.083 | +0.235 | -0.152 |  |
| E | GTA | -0.077 | +0.086 | -0.163 |  |
| E | AGA | -0.125 | +0.043 | -0.169 |  |
| E | CAG | +0.042 | +0.220 | -0.178 |  |
| E | GGC | -0.091 | +0.087 | -0.178 |  |
| E | AAG | -0.121 | +0.078 | -0.199 |  |
| E | ACG | +0.196 | +0.399 | -0.203 |  |
| E | TGG | -0.058 | +0.151 | -0.209 |  |
| E | TGT | +0.053 | +0.267 | -0.215 |  |
| E | AAT | +0.090 | +0.310 | -0.220 |  |
| E | CCG | -0.080 | +0.161 | -0.240 |  |
| E | CTG | +0.063 | +0.346 | -0.283 |  |
| E | CGA | +0.095 | +0.380 | -0.285 |  |
| E | GTG | -0.096 | +0.219 | -0.315 |  |
| E | GGT | -0.218 | +0.131 | -0.349 |  |
| E | AGT | +0.040 | +0.397 | -0.356 |  |
| E | AGG | -0.137 | +0.394 | -0.531 |  |

</details>

### Significant in BWM only (n = 10 site x codon cells)

| Site | Codon | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | TTC | +0.079 | -0.008 | +0.087 |  |
| A | GTA | -0.107 | -0.014 | -0.093 |  |
| A | GCG | -0.214 | -0.023 | -0.190 |  |
| P | GAG | +0.073 | -0.011 | +0.084 |  |
| P | GTC | -0.032 | -0.008 | -0.025 |  |
| P | GCT | -0.054 | +0.011 | -0.064 |  |
| P | GGG | -0.201 | -0.023 | -0.178 |  |
| P | GGT | -0.340 | -0.006 | -0.333 |  |
| E | GCA | -0.057 | +0.020 | -0.077 |  |
| E | CCT | -0.233 | -0.019 | -0.213 |  |

### Significant in control only (n = 31 site x codon cells)

| Site | Codon | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | GCC | -0.008 | -0.489 | +0.480 |  |
| A | CGC | -0.004 | -0.333 | +0.329 |  |
| A | CTC | -0.021 | -0.328 | +0.307 |  |
| A | CCG | -0.025 | -0.089 | +0.064 |  |
| A | CAG | +0.012 | -0.045 | +0.057 |  |
| A | TCT | -0.000 | +0.037 | -0.038 |  |
| A | AGC | +0.031 | +0.115 | -0.084 |  |
| A | TTA | -0.014 | +0.195 | -0.209 |  |
| A | ACA | +0.004 | +0.222 | -0.219 |  |
| A | ACG | -0.009 | +0.235 | -0.245 |  |
| A | GAA | -0.012 | +0.265 | -0.277 |  |
| A | AAA | +0.005 | +0.338 | -0.333 |  |
| P | TGG | +0.013 | -0.207 | +0.219 |  |
| P | CAA | +0.000 | -0.192 | +0.192 |  |
| P | CCC | +0.061 | -0.098 | +0.159 |  |
| P | TGC | +0.005 | -0.094 | +0.099 |  |
| P | AAC | -0.017 | -0.065 | +0.049 |  |
| P | TTC | +0.009 | +0.086 | -0.076 |  |
| P | GTG | +0.013 | +0.115 | -0.102 |  |
| P | CTA | +0.017 | +0.226 | -0.209 |  |
| P | GTT | -0.008 | +0.216 | -0.223 |  |
| E | ACC | +0.019 | -0.460 | +0.479 |  |
| E | ATC | +0.004 | -0.221 | +0.225 |  |
| E | ACT | -0.015 | -0.128 | +0.113 |  |
| E | CCC | -0.041 | -0.140 | +0.098 |  |
| E | CTT | +0.015 | -0.038 | +0.054 |  |
| E | ATT | -0.015 | +0.067 | -0.082 |  |
| E | AAA | -0.011 | +0.124 | -0.135 |  |
| E | GCG | +0.021 | +0.277 | -0.256 |  |
| E | GGG | -0.049 | +0.234 | -0.283 |  |
| E | CGG | +0.047 | +0.542 | -0.495 |  |

## Numbers at a glance
- `n_tests`: 372 (186 per condition; 3 sites x ~62 codons each)
- `n_significant` (adjusted-p < 0.05): 329 (BWM 154/186, control 175/186)
- `n_significant` (adjusted-p < 0.10): 335
- `min adjusted-p`: 0.0 (underflow; 20 cells at exactly 0.0). Smallest non-zero p_adj is 2.762370e-296 (A,BWM,TAC).
- `p_floor`: n/a -- Fisher with pooled N in the millions has no meaningful floor; the dominant concern is `large-N-Fisher-anticonservative` (family-wide).
- Per (condition, site) sig at FDR<0.05: BWM A 50/62, P 52/62, E 52/62; control A 59/62, P 56/62, E 60/62.
- Section split: 144 cells sig in both, 10 BWM-only, 31 control-only.

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 of (codon_count, total - codon_count) at day_5 vs day_0 within each condition; user confirmed. `global_codon_occ_stats.py` (Analysis 3b) pools the replicate counts per (condition, timepoint) before the test and applies BH-FDR within each (condition, site) family of ~62 codons (corrected by `merge_global_occupancy_analysis.py`); each of the 6 (condition, site) sub-families is corrected independently, not across the full 372-row file. Effect is reported as `log2_OR` (log2 of the CSV `odds_ratio`): >0 = enriched at day_5 relative to day_0 within that condition, <0 = depleted. The test does not compare BWM against control directly (that is the `control-vs-BWM-divergent-direction` reading) and does not test enrichment vs the transcriptomic background (the `within_condition_binomial` family). About 62 of 64 codons appear per (condition, site); the in-frame stop TGA is present, TAA/TAG are not retained in the significant set.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* -- the 2 biological replicates per (condition, timepoint) are summed into the 2x2 before Fisher; per-replicate variance is not in the test statistic, so p-values are anti-conservative. (Inherited -- see `_INDEX.md`.)
- **large-N-Fisher-anticonservative** *(family-wide)* -- pooled totals are whole-transcriptome footprint counts (day_5 ~2.01M BWM / ~2.41M control; day_0 ~2.11M BWM / ~3.37M control); Fisher returns vanishing p for tiny relative deviations. `log2_OR` is the primary effect column; p magnitude is not a ranking axis. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* -- BH is applied within each of the 6 (condition, site) families of ~62 codons; `p_adj` means corrected within this sub-family, not across the 372-row file. (Inherited.)
- **within-condition-clean** *(family-wide)* -- no condition or timepoint pooling across the contrast; structurally cleaner than the between-condition / between-timepoint Wilcoxon families. (Inherited.)
- **control-vs-BWM-divergent-direction** *(per-CSV)* -- flagged for Chumeng's reconciliation. Sense codons that split direction between conditions: A:AGG (BWM `log2_OR`=-0.202 / control +0.364), E:AGG (BWM -0.137 / control +0.394), E:TCC (BWM +0.212 / control -0.296), A:ACC (BWM +0.144 / control -0.287), E:CGG (BWM +0.047 / control +0.542), A:GCC (BWM -0.008 / control -0.489). The design-target reading of the within-condition contrast.
- **near-universal-sig-large-N** *(per-CSV)* -- 329/372 tests clear FDR<0.05, so section membership is weakly discriminating; the ranking axis is `|Effect change|`.
- **rare-codon-low-count** *(per-CSV)* -- the `low-count` flag (BWM<50 / control<50) fires only on the in-frame stop TGA cells (P:TGA and E:TGA (low-count BWM); A:TGA clears the threshold and is not flagged); all sense codons clear the threshold. Marks the same cells as `stop-codon-instability`.
- **stop-codon-instability** *(per-CSV)* -- TGA carries the largest |log2_OR| swings in the file (E:TGA (BWM log2_OR=+1.195 vs control +0.701), P:TGA (BWM +1.111 vs control +0.909), A:TGA (BWM +0.503 vs control +0.466)) on small counts in at least one arm; magnitude is unstable and must not anchor the read.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` -- sister CSVs: `aa_timepoint_fisher_within_condition_d5_vs_d0` (the AA aggregate of this contrast), plus the other two codon contrasts (additivity / monotonicity: does d10_vs_d0 ~= d5_vs_d0 + d10_vs_d5 per cell?). See `## Joint-reading suggestions` in [`_INDEX.md`](_INDEX.md).
- Falsifier (synonymous split): for each AA-level divergence in the sister AA file, does the codon file show the signal carried by one synonym (codon-usage shift) or spread across synonyms (amino-acid-level)? e.g. do the Arg-codon divergences (A:AGG, E:AGG) localise to AGG specifically or spread across the Arg family?
- Falsifier (stop TGA): TGA carries the largest swings here but is low-count/unstable. Does it reach a comparable extreme in any large-N file where its counts are higher (`per_timepoint_fisher_codon`), or only in these within-condition contrasts? If only here, treat it as sampling noise on a rare feature.
- Falsifier (shared-direction baselines): do the large shared-direction codon cells reappear as stable baselines across all 6 groups in `codon_within_condition_binomial` (then they are baseline composition both conditions carry, not a day_5-vs-day_0 effect), or as group-variable cells?
