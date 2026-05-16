---
input_csv: results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d5_vs_d0_codon.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U / Wilcoxon rank-sum (two-sided)
test_type_source: user-confirmed
n_tests: 183
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.4357142857142857
p_floor: 0.02857142857142857
pseudoreplicated: null
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "weighted_log2_enrichment-absent", proposed_by: dylan, status: confirmed, why: "Effect column is `log2_FC` of medians only."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "61 codons per site → BH wall ~3x more stringent than aa."}
  - {label: "low-count-rare-codon-instability", proposed_by: dylan, status: confirmed, why: "Top |effect| codons here include A-TTA (-1.341), P-GCG (-1.212), P-GGG (+1.001) — all median freq < 0.2%."}
headline: "Firm null at FDR<0.05 (0/183) for codon-level d5-vs-d0 MW with reps pooled across BWM and control per timepoint; min p_adj = 0.436 at site P — 4 floor rows (CTA -0.923, GAG +0.190, GCA -0.740, GCG -1.212). Five total floor rows file-wide (1 at A: ACT +0.227; 4 at P; 0 at E). Min raw p = 0.0286 (n=4 vs n=4 floor); 'no FDR hits' is structural per the locked floor caveat."
user_directives:
  - "(triage, batched across all 6 family members) Test type confirmation — `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → `Confirm`."
  - "(triage, batched across the 3 codon files) CSV-specific caveats → `weighted_log2_enrichment-absent`, `larger-bh-family`, `low-count-rare-codon-instability` confirmed."
---

# Interpretation — between_timepoint_wilcoxon_d5_vs_d0_codon

> Source: `results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d5_vs_d0_codon.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage, batched) Test type: `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → "Confirm".
- (triage, batched 3 codon files) CSV-specific caveats: `weighted_log2_enrichment-absent`, `larger-bh-family`, `low-count-rare-codon-instability` confirmed.

## Headline
Firm null at FDR<0.05 (0/183) for codon-level day_5 vs day_0 MW with BWM and control reps pooled within each timepoint (n=4 per side). Min p_adj = 0.436 at site P — 4 floor rows (CTA -0.923, GCG -1.212, GCA -0.740, GAG +0.190), giving BH wall 61*0.0286/4 = 0.436. File-wide: 5 floor rows (4 at P, 1 at A: ACT +0.227; 0 at E). Site E completely flat (min p_adj=0.926); site A min p_adj=0.775. Closest-to-significant codon-level signature: at site P, three depletions (CTA, GCG, GCA) and one enrichment (GAG) all reaching the floor between day_0 and day_5 reps. Treat as exploratory; the floor blocks formal FDR<0.05.

## Top hits

### P site (headline group — min p_adj = 0.436, 4 floor rows)

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GAG | +0.190 | 0.436 | floor |
| enriched | TCT | +0.388 | 0.775 | nominal-only |
| enriched | GGG | +1.001 | 0.775 | low-count |
| enriched | CAA | +0.250 | 0.871 |  |
| enriched | GTC | +0.335 | 0.871 |  |
| depleted | GCG | -1.212 | 0.436 | floor, low-count |
| depleted | CTA | -0.923 | 0.436 | floor, low-count |
| depleted | GCA | -0.740 | 0.436 | floor |
| depleted | GGT | -0.115 | 0.697 | nominal-only |
| depleted | AGG | -0.729 | 0.775 | low-count |

<details>
<summary>A site (1 floor row)</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | ACT | +0.227 | 0.775 | floor |
| enriched | AAC | +0.672 | 0.775 | nominal-only |
| enriched | ATC | +0.470 | 0.775 |  |
| enriched | TCT | +0.250 | 0.775 |  |
| enriched | CAA | +0.158 | 0.775 |  |
| depleted | TTA | -1.341 | 0.775 | nominal-only, low-count |
| depleted | GCA | -0.713 | 0.775 | nominal-only |
| depleted | GGC | -0.683 | 0.775 | nominal-only, low-count |
| depleted | CTG | -0.652 | 0.775 | nominal-only, low-count |
| depleted | TGG | -0.671 | 0.775 |  |

</details>

<details>
<summary>E site (no floor rows; flat)</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TCC | +0.442 | 0.926 | nominal-only |
| enriched | CAA | +0.433 | 0.926 |  |
| enriched | TCT | +0.385 | 0.926 |  |
| enriched | CAG | +0.312 | 0.926 |  |
| enriched | ACT | +0.199 | 0.926 |  |
| depleted | GCG | -0.610 | 0.926 | low-count |
| depleted | CCA | -0.525 | 0.926 |  |
| depleted | CTA | -0.398 | 0.926 | low-count |
| depleted | AGA | -0.270 | 0.926 |  |
| depleted | ACC | -0.143 | 0.926 |  |

</details>

## Numbers at a glance
- `n_tests`: 183
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.436 (P-site, BH wall for 4 floor rows in 61-codon family)
- `min raw-p`: 0.02857 (= MW exact floor); 5 rows at the floor — A: ACT; P: CTA, GAG, GCA, GCG
- `p_floor`: 0.02857 (n=4 vs n=4 two-sided exact)
- Per-site BH families:
  - A site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.775 (1 floor row)
  - E site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.926 (0 floor rows)
  - P site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.436 (4 floor rows)

## Methods
MW rank-sum two-sided on per-replicate frequencies, n=4 day_5 (BWM_day5_rep2/3 + control_day5_rep2/3) vs n=4 day_0, BH-FDR per site (each site = 61-codon family). Test answers "do day_5 and day_0 reps differ in per-rep codon frequency at this site?" with BWM and control reps pooled within each timepoint.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide).
- **condition-pooled-confound** (family-wide).
- **n=4-low-power** (family-wide).
- **p-floor-aware-headline** (family-wide).
- **weighted_log2_enrichment-absent** (per-CSV).
- **larger-bh-family** (per-CSV) — aa-level d5_vs_d0 file gets to p_adj=0.571 with one floor row (P-C); here at codon level, 4 floor rows still only p_adj=0.436.
- **low-count-rare-codon-instability** (per-CSV) — extreme |effect| at A-TTA (-1.341), P-GCG (-1.212), P-GGG (+1.001), P-CTA (-0.923) all sit on rare codons (median freq < 0.2%).

### Considered but not applicable
*(Dylan did not propose any further per-CSV caveats here.)*

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs: `between_timepoint_wilcoxon_d5_vs_d0_aa.csv` (aa, same contrast), and the d10_vs_d0 / d10_vs_d5 pairs at both resolutions.
- Open questions Chumeng should resolve at synthesis time:
  - **The aa-level d5_vs_d0 file's only floor row is P-site C (+0.249) — but C has codons TGT, TGC, neither of which is at the floor here.** P-TGC: log2=+0.068, p=0.486. P-TGT: log2=+0.427, p=0.686. So the aa-level P-C signal is actually distributed across both Cys codons but neither individually clears the codon BH (3x more stringent at codon level). aa-level enrichment is real-but-codon-balanced, not single-codon-driven.
  - **Site P codon hits here (CTA depleted, GCG depleted, GCA depleted, GAG enriched) are NOT visible at the aa level**: P-A is +0.021 (p=1.0), P-E is +0.268 (p=0.343), P-L (codons including CTA) is +0.004 (p=0.686). This is the inverse of the previous question: codon-level signals at site P that aa-level aggregation hides. Only one or two synonymous codons of A/L/E carry the day-5-vs-day-0 shift.
  - Cross-contrast tracking — d10_vs_d0 codon's only P-site floor row was GAG (+0.387); d5_vs_d0 P-GAG is +0.190 (floor). The numerical sequence is consistent with several patterns (monotonic accumulation, biphasic, two independent floor coincidences). Does P-GAG show coherent direction in `per_timepoint_fisher_codon` and `timepoint_fisher_within_condition_*_codon` files? If yes across both contrast designs → Chumeng decides whether to elevate to a consensus feature; if only one design → reading is contrast-design-dependent.
  - Site E completely flat at every contrast in this family (d10_vs_d0 codon: 1 floor row, p_adj=0.93; d10_vs_d5 codon: 1 floor, p_adj=0.87; d5_vs_d0 codon: 0 floor, p_adj=0.93). Suggests that whatever timepoint shifts exist concentrate at the A and P sites, not E.
