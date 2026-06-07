---
input_csv: results/global_occupancy/analysis_corrected/codon_timepoint_fisher_within_condition_d10_vs_d0.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), within-condition day_10-vs-day_0 contrast per (condition, site, codon); BH-FDR within each (condition, site) family of ~62 codons
test_type_source: user-confirmed
n_tests: 372
n_significant_fdr05: 330
n_significant_fdr10: 337
min_p_adj: 0.0
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Sense codons where BWM and control split direction in the day_10-vs-day_0 contrast (design-target reading), reported alongside shared-direction cells per A.2.2. See the file's Caveats section for the listed cells. User-confirmed."}
  - {label: "near-universal-sig-large-N", proposed_by: dylan, status: confirmed, why: "330/372 tests clear FDR<0.05 at whole-transcriptome pooled N; both/BWM-only/control-only section membership is weakly discriminating, so rank by |Effect change|, not by section or p. User-confirmed."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "The within_condition_sig_split low-count flag (BWM<50 / control<50) fires only on the in-frame stop codon TGA: A:TGA (low-count control), P:TGA and E:TGA (low-count BWM and control). All sense codons clear the threshold, so this caveat and stop-codon-instability mark the same TGA cells. User-confirmed."}
  - {label: "stop-codon-instability", proposed_by: dylan, status: confirmed, why: "The in-frame stop TGA carries the largest |log2_OR| swings in the file (A:TGA (BWM log2_OR=+1.039 vs control -1.235), P:TGA (BWM +1.744 vs control -0.375), E:TGA (BWM +2.145 vs control -0.633)) but rests on small counts in at least one arm; its magnitude is unstable and must not anchor the read. User-confirmed."}
caveats_considered: []
headline: "330/372 within-condition Fisher tests sig at FDR<0.05 (BWM 162/186, control 168/186; 148 cells sig in both, 14 BWM-only, 20 control-only); min p_adj underflows to 0.0 (7 cells) at whole-transcriptome N. In-frame stop TGA carries the largest swings (E:TGA BWM log2_OR=+2.145, A:TGA +1.039 vs control -1.235) but is low-count/unstable; largest stable sense divergences at site A: A:CCC (BWM -0.542 vs control +0.264), A:TGG/Trp (-0.586 vs +0.088), A:GGC (-0.359 vs +0.188). Largest shared-direction sense cells: E:CCA both depleted (-0.538 / -0.242), P:CGG both enriched (+0.402 / +0.488)."
user_directives:
  - "(per-CSV triage) 'Confirm test type for the family?' -> 'Fisher's exact, BH per (condition, site)' (applies to all 6 files)."
  - "(per-CSV triage) 'Per-CSV caveats beyond the 4 locked family caveats?' -> confirmed control-vs-BWM-divergent-direction, near-universal-sig-large-N, rare-codon-low-count, stop-codon-instability."
  - "(per-CSV triage) 'How firmly should this family read?' -> 'Firm' (significant cells read as established; still rank by |Effect change|, not p)."
  - "(per-CSV triage) Top-hits table source -> user authorised running scripts/within_condition_sig_split.py (codon files with --rare-bwm-threshold 50 --rare-control-threshold 50) to generate the three-section paired tables transcribed below."
---

# Interpretation -- codon_timepoint_fisher_within_condition_d10_vs_d0

> Source: `results/global_occupancy/analysis_corrected/codon_timepoint_fisher_within_condition_d10_vs_d0.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), within-condition day_10-vs-day_0 contrast per (condition, site, codon); BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (per-CSV triage) "Confirm test type for the family?" -> "Fisher's exact, BH per (condition, site)" (confirmed for all 6 family files).
- (per-CSV triage) "Per-CSV caveats beyond the 4 locked family caveats?" -> confirmed `control-vs-BWM-divergent-direction`, `near-universal-sig-large-N`, `rare-codon-low-count`, `stop-codon-instability` (the last two for codon files only).
- (per-CSV triage) "How firmly should this family read?" -> "Firm". Significant cells read as established; the A.2.x large-N discipline still applies -- rank by `|Effect change|`, not p.
- (per-CSV triage) Top-hits table source -> user authorised running `scripts/within_condition_sig_split.py` (a display-only reshape of the existing `odds_ratio`/`p_adj` columns -- no statistics re-run; codon files use `--rare-bwm-threshold 50 --rare-control-threshold 50`) to generate the three-section paired tables below.

## Headline
At codon-level within-condition Fisher d10 vs d0, 330/372 tests clear FDR<0.05 (BWM 162/186, control 168/186): 148 site x codon cells significant in both conditions, 14 BWM-only, 20 control-only. Minimum adjusted p underflows to 0.0 (7 cells) at whole-transcriptome pooled N (totals 1.3M-3.4M); the informative axis is `Effect change`, not p.

The in-frame stop codon TGA carries the largest swings in the file (A:TGA BWM `log2_OR`=+1.039 vs control -1.235; P:TGA BWM +1.744; E:TGA BWM +2.145) but rests on small counts in at least one arm and is flagged `low-count` / unstable (see caveats) -- it should not anchor the read. Among stable sense codons, the largest BWM-vs-control divergences concentrate at site A: A:CCC (BWM -0.542 / control +0.264), A:TGG (the sole Trp codon; BWM -0.586 / control +0.088, mirroring the AA file's A:W split), A:GGC (BWM -0.359 / control +0.188), A:CGG (BWM -0.031 / control +0.566), A:CTG (BWM -0.115 / control +0.404). Reported at equal billing (A.2.2), the largest shared-direction sense cells (both arms same sign) are E:CCA both depleted (BWM -0.538 / control -0.242), A:CCA both depleted (-0.249 / -0.148), P:CGG both enriched (+0.402 / +0.488), P:CCG both enriched (+0.334 / +0.368), A:GAT both enriched (+0.306 / +0.051).

## Top hits

`log2_OR` is the within-condition Fisher effect for the **day_10 vs day_0** contrast: positive = enriched at day_10 relative to day_0, negative = depleted. Each row pairs the BWM and control value for one (site, codon) cell; `Effect change` = BWM `log2_OR` - control `log2_OR` (large magnitude = the two conditions' day_10-vs-day_0 trajectories diverge). Rows are grouped by `site` in A -> P -> E order, then sorted by `Effect change` descending. Cells significant (FDR<0.05) in neither condition are omitted. Generated by `scripts/within_condition_sig_split.py`.

Every cell shown is FDR-significant in at least one arm at whole-transcriptome N (most both-section cells sit at p_adj far below 1e-10), so per A.2.6/A.2.4 the large-N anti-conservatism alternative explanation applies symmetrically to every such row and ranking is by `|Effect change|`, not p. The `Flag` column carries `low-count`; in this family it fires only on the in-frame stop codon TGA (see the `rare-codon-low-count` / `stop-codon-instability` caveats).

### Significant in both conditions (n = 148 site x codon cells)

<details>
<summary>Full 148-cell both-conditions table (A / P / E, sorted by Effect change desc)</summary>

| Site | Codon | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | TGA | +1.039 | -1.235 | +2.274 | low-count (C) |
| A | AAC | +0.171 | -0.235 | +0.406 |  |
| A | GTT | +0.301 | -0.050 | +0.351 |  |
| A | AAT | +0.305 | -0.035 | +0.340 |  |
| A | GTC | +0.182 | -0.138 | +0.319 |  |
| A | ACC | +0.053 | -0.250 | +0.303 |  |
| A | GAT | +0.306 | +0.051 | +0.255 |  |
| A | ATC | +0.149 | -0.090 | +0.238 |  |
| A | GAG | +0.178 | +0.016 | +0.161 |  |
| A | GAC | +0.060 | -0.090 | +0.150 |  |
| A | GCC | -0.055 | -0.180 | +0.125 |  |
| A | ATG | +0.207 | +0.096 | +0.111 |  |
| A | AAG | -0.125 | -0.228 | +0.103 |  |
| A | CGT | +0.056 | -0.029 | +0.084 |  |
| A | CAA | -0.160 | -0.237 | +0.077 |  |
| A | AGT | +0.231 | +0.172 | +0.059 |  |
| A | TCA | +0.150 | +0.110 | +0.040 |  |
| A | AGC | +0.088 | +0.051 | +0.036 |  |
| A | TGT | +0.231 | +0.202 | +0.030 |  |
| A | GGA | -0.062 | -0.089 | +0.027 |  |
| A | TTT | +0.154 | +0.139 | +0.015 |  |
| A | TTC | +0.080 | +0.070 | +0.010 |  |
| A | GCT | +0.028 | +0.023 | +0.006 |  |
| A | TCC | -0.098 | -0.095 | -0.003 |  |
| A | CTC | -0.144 | -0.137 | -0.007 |  |
| A | GGT | -0.159 | -0.125 | -0.034 |  |
| A | CTT | -0.129 | -0.094 | -0.036 |  |
| A | CAC | -0.216 | -0.175 | -0.041 |  |
| A | GAA | +0.120 | +0.164 | -0.044 |  |
| A | CCA | -0.249 | -0.148 | -0.101 |  |
| A | TAT | -0.034 | +0.088 | -0.122 |  |
| A | ATA | +0.040 | +0.185 | -0.145 |  |
| A | TCG | +0.087 | +0.243 | -0.156 |  |
| A | GGG | +0.106 | +0.266 | -0.160 |  |
| A | ACA | -0.062 | +0.121 | -0.182 |  |
| A | TTA | -0.054 | +0.163 | -0.217 |  |
| A | GTG | -0.303 | +0.047 | -0.350 |  |
| A | TTG | -0.287 | +0.079 | -0.366 |  |
| A | CGA | -0.054 | +0.341 | -0.395 |  |
| A | CCG | -0.038 | +0.362 | -0.400 |  |
| A | TAC | -0.447 | -0.043 | -0.404 |  |
| A | TGC | -0.280 | +0.131 | -0.412 |  |
| A | CTA | -0.180 | +0.253 | -0.432 |  |
| A | GCA | -0.273 | +0.165 | -0.438 |  |
| A | CCT | -0.173 | +0.271 | -0.444 |  |
| A | GCG | -0.133 | +0.346 | -0.479 |  |
| A | CTG | -0.115 | +0.404 | -0.519 |  |
| A | GGC | -0.359 | +0.188 | -0.547 |  |
| A | TGG | -0.586 | +0.088 | -0.674 |  |
| A | CCC | -0.542 | +0.264 | -0.806 |  |
| P | TTA | +0.388 | -0.074 | +0.462 |  |
| P | CTC | +0.109 | -0.152 | +0.261 |  |
| P | TTG | +0.140 | -0.109 | +0.249 |  |
| P | TCC | +0.062 | -0.154 | +0.216 |  |
| P | ATA | +0.278 | +0.077 | +0.201 |  |
| P | GTC | +0.029 | -0.169 | +0.198 |  |
| P | GAG | +0.277 | +0.092 | +0.185 |  |
| P | ATG | +0.103 | -0.066 | +0.170 |  |
| P | ACC | -0.056 | -0.218 | +0.161 |  |
| P | TCG | +0.294 | +0.154 | +0.140 |  |
| P | GCC | -0.058 | -0.179 | +0.121 |  |
| P | CTT | -0.037 | -0.150 | +0.113 |  |
| P | CTG | +0.206 | +0.099 | +0.107 |  |
| P | ATC | -0.121 | -0.222 | +0.101 |  |
| P | ACG | +0.341 | +0.243 | +0.098 |  |
| P | CCC | +0.267 | +0.191 | +0.076 |  |
| P | AAG | -0.232 | -0.296 | +0.065 |  |
| P | CAC | -0.113 | -0.162 | +0.049 |  |
| P | GTG | +0.137 | +0.095 | +0.043 |  |
| P | CCA | -0.195 | -0.234 | +0.039 |  |
| P | TCA | +0.102 | +0.068 | +0.034 |  |
| P | TAC | -0.100 | -0.124 | +0.024 |  |
| P | AAA | +0.046 | +0.023 | +0.022 |  |
| P | CAA | -0.091 | -0.113 | +0.022 |  |
| P | GAA | +0.123 | +0.102 | +0.020 |  |
| P | GTT | -0.049 | -0.050 | +0.001 |  |
| P | CAG | +0.092 | +0.096 | -0.004 |  |
| P | ACA | +0.075 | +0.087 | -0.013 |  |
| P | ATT | -0.090 | -0.075 | -0.014 |  |
| P | GCG | +0.286 | +0.314 | -0.028 |  |
| P | CGT | -0.062 | -0.033 | -0.029 |  |
| P | CCG | +0.334 | +0.368 | -0.033 |  |
| P | AGA | -0.136 | -0.079 | -0.057 |  |
| P | AAC | -0.133 | -0.073 | -0.061 |  |
| P | GAC | -0.042 | +0.027 | -0.069 |  |
| P | GGG | +0.146 | +0.218 | -0.071 |  |
| P | TTT | +0.099 | +0.174 | -0.075 |  |
| P | GCT | -0.046 | +0.035 | -0.081 |  |
| P | CCT | +0.276 | +0.357 | -0.081 |  |
| P | CGG | +0.402 | +0.488 | -0.086 |  |
| P | GGC | +0.063 | +0.159 | -0.096 |  |
| P | TGT | +0.197 | +0.296 | -0.099 |  |
| P | TAT | -0.035 | +0.066 | -0.101 |  |
| P | AGG | +0.252 | +0.372 | -0.120 |  |
| P | AAT | +0.085 | +0.206 | -0.120 |  |
| P | AGT | +0.063 | +0.226 | -0.164 |  |
| P | GGA | -0.274 | -0.109 | -0.165 |  |
| P | GAT | +0.103 | +0.297 | -0.194 |  |
| P | GGT | -0.173 | +0.063 | -0.235 |  |
| P | CGA | -0.037 | +0.410 | -0.446 |  |
| E | TCC | +0.153 | -0.053 | +0.207 |  |
| E | TCG | +0.414 | +0.254 | +0.160 |  |
| E | TCT | +0.202 | +0.047 | +0.156 |  |
| E | CAT | +0.182 | +0.035 | +0.147 |  |
| E | CAC | +0.059 | -0.075 | +0.134 |  |
| E | GTC | +0.043 | -0.070 | +0.114 |  |
| E | TCA | +0.241 | +0.145 | +0.096 |  |
| E | TAC | -0.058 | -0.148 | +0.090 |  |
| E | GAG | +0.045 | -0.041 | +0.087 |  |
| E | CAA | +0.045 | -0.038 | +0.082 |  |
| E | ATG | +0.134 | +0.053 | +0.082 |  |
| E | ACC | -0.110 | -0.186 | +0.076 |  |
| E | GGA | -0.137 | -0.208 | +0.071 |  |
| E | TTA | +0.160 | +0.092 | +0.068 |  |
| E | GCC | -0.070 | -0.131 | +0.061 |  |
| E | CAG | +0.133 | +0.077 | +0.056 |  |
| E | ACA | +0.189 | +0.136 | +0.053 |  |
| E | CTC | +0.123 | +0.072 | +0.051 |  |
| E | ACG | +0.346 | +0.299 | +0.046 |  |
| E | GGC | +0.230 | +0.188 | +0.043 |  |
| E | GGG | +0.258 | +0.218 | +0.040 |  |
| E | CGT | -0.065 | -0.100 | +0.035 |  |
| E | AAC | -0.070 | -0.090 | +0.020 |  |
| E | AGC | +0.136 | +0.118 | +0.018 |  |
| E | GAT | +0.096 | +0.083 | +0.013 |  |
| E | ATC | -0.073 | -0.083 | +0.011 |  |
| E | AGT | +0.248 | +0.252 | -0.004 |  |
| E | TTC | -0.031 | -0.024 | -0.007 |  |
| E | AAT | +0.104 | +0.114 | -0.009 |  |
| E | TGT | +0.112 | +0.122 | -0.010 |  |
| E | AAG | -0.186 | -0.171 | -0.014 |  |
| E | GAA | +0.090 | +0.125 | -0.034 |  |
| E | CTA | +0.195 | +0.238 | -0.044 |  |
| E | AGA | -0.163 | -0.116 | -0.047 |  |
| E | ATA | +0.191 | +0.239 | -0.048 |  |
| E | GTT | -0.094 | -0.042 | -0.052 |  |
| E | GCG | +0.239 | +0.305 | -0.066 |  |
| E | TTT | +0.034 | +0.107 | -0.073 |  |
| E | CGA | +0.271 | +0.373 | -0.102 |  |
| E | ACT | -0.081 | +0.030 | -0.111 |  |
| E | GTG | -0.029 | +0.100 | -0.130 |  |
| E | AAA | -0.032 | +0.102 | -0.134 |  |
| E | CTG | +0.153 | +0.291 | -0.138 |  |
| E | GTA | -0.082 | +0.080 | -0.163 |  |
| E | CGG | +0.300 | +0.496 | -0.196 |  |
| E | CCG | +0.054 | +0.301 | -0.247 |  |
| E | CCA | -0.538 | -0.242 | -0.296 |  |
| E | CCT | -0.128 | +0.281 | -0.409 |  |

</details>

### Significant in BWM only (n = 14 site x codon cells)

| Site | Codon | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | ATT | +0.295 | +0.005 | +0.290 |  |
| A | ACT | +0.043 | -0.015 | +0.057 |  |
| A | CGC | -0.033 | +0.000 | -0.034 |  |
| A | GTA | -0.168 | -0.015 | -0.153 |  |
| P | TGA | +1.744 | -0.375 | +2.119 | low-count (BWM, C) |
| P | TTC | -0.037 | -0.015 | -0.021 |  |
| E | TGA | +2.145 | -0.633 | +2.777 | low-count (BWM, C) |
| E | TTG | +0.138 | +0.006 | +0.132 |  |
| E | TGC | +0.106 | -0.015 | +0.120 |  |
| E | TGG | +0.083 | -0.001 | +0.084 |  |
| E | GCT | -0.046 | -0.014 | -0.032 |  |
| E | TAT | -0.039 | +0.001 | -0.040 |  |
| E | CTT | -0.034 | +0.013 | -0.047 |  |
| E | ATT | -0.096 | -0.013 | -0.083 |  |

### Significant in control only (n = 20 site x codon cells)

| Site | Codon | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | AGA | +0.022 | -0.064 | +0.086 |  |
| A | CAG | +0.014 | -0.058 | +0.073 |  |
| A | AAA | -0.002 | +0.019 | -0.021 |  |
| A | TCT | -0.023 | +0.042 | -0.065 |  |
| A | CAT | +0.003 | +0.075 | -0.072 |  |
| A | ACG | +0.030 | +0.251 | -0.222 |  |
| A | AGG | +0.004 | +0.398 | -0.394 |  |
| A | CGG | -0.031 | +0.566 | -0.597 |  |
| P | TGG | +0.035 | -0.107 | +0.141 |  |
| P | CGC | +0.029 | -0.042 | +0.071 |  |
| P | TCT | -0.023 | -0.032 | +0.009 |  |
| P | CAT | +0.022 | +0.047 | -0.026 |  |
| P | GTA | +0.031 | +0.059 | -0.028 |  |
| P | ACT | -0.008 | +0.058 | -0.066 |  |
| P | GCA | -0.017 | +0.180 | -0.197 |  |
| P | CTA | -0.041 | +0.171 | -0.212 |  |
| E | CGC | +0.023 | -0.090 | +0.113 |  |
| E | GCA | +0.021 | +0.138 | -0.117 |  |
| E | CCC | +0.018 | +0.215 | -0.197 |  |
| E | AGG | -0.015 | +0.244 | -0.258 |  |

## Numbers at a glance
- `n_tests`: 372 (186 per condition; 3 sites x ~62 codons each)
- `n_significant` (adjusted-p < 0.05): 330 (BWM 162/186, control 168/186)
- `n_significant` (adjusted-p < 0.10): 337
- `min adjusted-p`: 0.0 (underflow; 7 cells at exactly 0.0). Smallest non-zero p_adj is 3.434934e-251 (P,BWM,GAG).
- `p_floor`: n/a -- Fisher with pooled N in the millions has no meaningful floor; the dominant concern is `large-N-Fisher-anticonservative` (family-wide).
- Per (condition, site) sig at FDR<0.05: BWM A 54/62, P 52/62, E 56/62; control A 58/62, P 58/62, E 52/62.
- Section split: 148 cells sig in both, 14 BWM-only, 20 control-only.

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 of (codon_count, total - codon_count) at day_10 vs day_0 within each condition; user confirmed. `global_codon_occ_stats.py` (Analysis 3b) pools the replicate counts per (condition, timepoint) before the test and applies BH-FDR within each (condition, site) family of ~62 codons (corrected by `merge_global_occupancy_analysis.py`); each of the 6 (condition, site) sub-families is corrected independently, not across the full 372-row file. Effect is reported as `log2_OR` (log2 of the CSV `odds_ratio`): >0 = enriched at day_10 relative to day_0 within that condition, <0 = depleted. The test does not compare BWM against control directly (that is the `control-vs-BWM-divergent-direction` reading) and does not test enrichment vs the transcriptomic background (the `within_condition_binomial` family). About 62 of 64 codons appear per (condition, site); the in-frame stop TGA is present, TAA/TAG are not retained in the significant set.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* -- the 2 biological replicates per (condition, timepoint) are summed into the 2x2 before Fisher; per-replicate variance is not in the test statistic, so p-values are anti-conservative. (Inherited -- see `_INDEX.md`.)
- **large-N-Fisher-anticonservative** *(family-wide)* -- pooled totals are whole-transcriptome footprint counts (day_10 ~1.56M BWM / ~1.26M control; day_0 ~2.11M BWM / ~3.37M control); Fisher returns vanishing p for tiny relative deviations. `log2_OR` is the primary effect column; p magnitude is not a ranking axis. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* -- BH is applied within each of the 6 (condition, site) families of ~62 codons; `p_adj` means corrected within this sub-family, not across the 372-row file. (Inherited.)
- **within-condition-clean** *(family-wide)* -- no condition or timepoint pooling across the contrast; structurally cleaner than the between-condition / between-timepoint Wilcoxon families. (Inherited.)
- **control-vs-BWM-divergent-direction** *(per-CSV)* -- flagged for Chumeng's reconciliation. Sense codons that split direction between conditions: A:CCC (BWM `log2_OR`=-0.542 / control +0.264), A:TGG (Trp; BWM -0.586 / control +0.088), A:GGC (BWM -0.359 / control +0.188), A:CGG (BWM -0.031 / control +0.566), A:CTG (BWM -0.115 / control +0.404). The design-target reading of the within-condition contrast.
- **near-universal-sig-large-N** *(per-CSV)* -- 330/372 tests clear FDR<0.05, so section membership is weakly discriminating; the ranking axis is `|Effect change|`.
- **rare-codon-low-count** *(per-CSV)* -- the `low-count` flag (BWM<50 / control<50) fires only on the in-frame stop TGA cells (A:TGA (low-count control), P:TGA and E:TGA (low-count BWM and control)); all sense codons clear the threshold. Marks the same cells as `stop-codon-instability`.
- **stop-codon-instability** *(per-CSV)* -- TGA carries the largest |log2_OR| swings in the file (A:TGA (BWM log2_OR=+1.039 vs control -1.235), P:TGA (BWM +1.744 vs control -0.375), E:TGA (BWM +2.145 vs control -0.633)) on small counts in at least one arm; magnitude is unstable and must not anchor the read.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` -- sister CSVs: `aa_timepoint_fisher_within_condition_d10_vs_d0` (the AA aggregate of this contrast), plus the other two codon contrasts (additivity / monotonicity: does d10_vs_d0 ~= d5_vs_d0 + d10_vs_d5 per cell?). See `## Joint-reading suggestions` in [`_INDEX.md`](_INDEX.md).
- Falsifier (synonymous split): for each AA-level divergence in the sister AA file, does the codon file show the signal carried by one synonym (codon-usage shift) or spread across synonyms (amino-acid-level)? e.g. does the AA-level Trp split at site A localise to TGG here?
- Falsifier (stop TGA): TGA carries the largest swings here but is low-count/unstable. Does it reach a comparable extreme in any large-N file where its counts are higher (`per_timepoint_fisher_codon`), or only in these within-condition contrasts? If only here, treat it as sampling noise on a rare feature.
- Falsifier (shared-direction baselines): do the large shared-direction codon cells reappear as stable baselines across all 6 groups in `codon_within_condition_binomial` (then they are baseline composition both conditions carry, not a day_10-vs-day_0 effect), or as group-variable cells?
