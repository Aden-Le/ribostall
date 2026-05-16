---
input_csv: results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d5_codon.csv
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
  - {label: "low-count-rare-codon-instability", proposed_by: dylan, status: confirmed, why: "Top |effect| codons here include TTT (-0.659 at P), CCT (+0.596 at E), CCA (+0.337 at E) — multiple have median freq < 0.5%."}
headline: "Firm null at FDR<0.05 (0/183) for codon-level d10-vs-d5 MW with reps pooled across BWM and control per timepoint; min p_adj = 0.436 at site A — the lowest in any codon file in this family — driven by 4 site-A floor rows (CAA -0.681, GGA +0.317, TCT -0.281, TTT -0.465). Total floor rows: 7 (4 at A, 1 at E, 2 at P). Min raw p = 0.0286 (n=4 vs n=4 floor); 'no FDR hits' is structural per the locked floor caveat."
user_directives:
  - "(triage, batched across all 6 family members) Test type confirmation — `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → `Confirm`."
  - "(triage, batched across the 3 codon files) CSV-specific caveats → `weighted_log2_enrichment-absent`, `larger-bh-family`, `low-count-rare-codon-instability` confirmed."
---

# Interpretation — between_timepoint_wilcoxon_d10_vs_d5_codon

> Source: `results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d5_codon.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage, batched) Test type: `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → "Confirm".
- (triage, batched 3 codon files) CSV-specific caveats: `weighted_log2_enrichment-absent`, `larger-bh-family`, `low-count-rare-codon-instability` confirmed.

## Headline
Firm null at FDR<0.05 (0/183) for codon-level day_10 vs day_5 MW with BWM and control reps pooled within each timepoint (n=4 per side). Min p_adj = 0.436 at site A — driven by 4 floor rows (CAA -0.681, TTT -0.465, GGA +0.317, TCT -0.281), giving BH wall 61*0.0286/4 = 0.436 (matches observed). Total floor rows file-wide: 7 (4 at A, 1 at E (ATT -0.304), 2 at P (CTT -0.189, GGA +0.366)). Site E (min p_adj 0.871) and site P (min p_adj 0.581) are flatter than A. Closest-to-significant codon-level signature: at A site, CAA / TCT / TTT depletion + GGA enrichment in day_10 vs day_5 — i.e. day_10 reps trend toward more A-site GGA (Glycine) and less A-site CAA (Glutamine), TCT/TTT (Ser/Phe). Treat as exploratory; the floor blocks formal FDR<0.05 here regardless of biology.

## Top hits

### A site (headline group — min p_adj = 0.436, 4 floor rows)

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GGA | +0.317 | 0.436 | floor |
| enriched | CGT | +0.367 | 0.498 | nominal-only |
| enriched | CGC | +0.250 | 0.697 |  |
| enriched | TGG | +0.175 | 0.697 |  |
| enriched | ACC | +0.197 | 0.871 |  |
| depleted | CAA | -0.681 | 0.436 | floor |
| depleted | TTT | -0.465 | 0.436 | floor |
| depleted | TCT | -0.281 | 0.436 | floor |
| depleted | AAA | -0.388 | 0.498 | nominal-only |
| depleted | ACT | -0.301 | 0.498 | nominal-only |

<details>
<summary>E site (1 floor row)</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GCC | +0.225 | 0.871 | nominal-only |
| enriched | ACC | +0.311 | 0.938 |  |
| enriched | CTC | +0.338 | 0.938 |  |
| enriched | CCA | +0.337 | 0.948 |  |
| enriched | CCT | +0.596 | 0.948 | low-count |
| depleted | ATT | -0.304 | 0.871 | floor |
| depleted | GTG | -0.480 | 0.871 | nominal-only |
| depleted | ACG | -0.465 | 0.871 | nominal-only |
| depleted | GCG | -0.593 | 0.938 | low-count |
| depleted | GAT | -0.195 | 0.938 |  |

</details>

<details>
<summary>P site (2 floor rows)</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GGA | +0.366 | 0.581 | floor |
| enriched | ACC | +0.281 | 0.581 | nominal-only |
| enriched | AGG | +0.371 | 0.757 |  |
| enriched | GAG | +0.197 | 0.813 |  |
| enriched | TGC | +0.151 | 0.813 |  |
| depleted | CTT | -0.189 | 0.581 | floor |
| depleted | CAA | -0.556 | 0.581 | nominal-only |
| depleted | ATT | -0.476 | 0.581 | nominal-only |
| depleted | TCT | -0.304 | 0.581 | nominal-only |
| depleted | TTT | -0.659 | 0.757 | low-count |

</details>

## Numbers at a glance
- `n_tests`: 183
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.436 (site A, BH wall for 4 floor rows in 61-codon family: 61*0.0286/4)
- `min raw-p`: 0.02857 (= MW exact floor); 7 rows at the floor — A: CAA, GGA, TCT, TTT; E: ATT; P: CTT, GGA
- `p_floor`: 0.02857 (n=4 vs n=4 two-sided exact)
- Per-site BH families:
  - A site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.436 (4 floor rows)
  - E site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.871 (1 floor row)
  - P site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.581 (2 floor rows)

## Methods
Same as the rest of the family: MW rank-sum two-sided on per-replicate frequencies, n=4 day_10 (BWM_day10_rep2/3 + control_day10_rep2/3) vs n=4 day_5, BH-FDR per site over 61-codon families. Test answers "do day_10 and day_5 reps differ in per-rep codon frequency at this site?" with BWM and control reps pooled within timepoint.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide).
- **condition-pooled-confound** (family-wide).
- **n=4-low-power** (family-wide).
- **p-floor-aware-headline** (family-wide).
- **weighted_log2_enrichment-absent** (per-CSV).
- **larger-bh-family** (per-CSV) — 61 codons per site means BH wall is ~3x further from FDR<0.05 vs aa-resolution. The aa-level d10_vs_d5 file gets to p_adj=0.381 with 1 floor + 2 near-floor rows; here at codon level, 4 floor rows still only get to p_adj=0.436.
- **low-count-rare-codon-instability** (per-CSV) — top |effect| at E (CCT +0.596 at median freq ~0.1%) and P (TTT -0.659 at median freq ~0.5%) sit in the regime where small absolute count moves dominate.

### Considered but not applicable
*(Dylan did not propose any further per-CSV caveats here.)*

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs: `between_timepoint_wilcoxon_d10_vs_d5_aa.csv` (aa, same contrast) and the d10_vs_d0 / d5_vs_d0 pairs.
- Open questions Chumeng should resolve at synthesis time:
  - aa-codon agreement: the aa file at d10_vs_d5 reports A-site G enrichment at the floor and Q depletion near-floor. Codon level: A-GGA (+0.317, floor) is the single GGA = Gly synonymous codon hitting the floor — so the A-G aa-level signal is concentrated on GGA, not the other Gly codons (GGT/GGC/GGG). For A-Q (-0.462 aa-level): A-CAA (-0.681) is at the floor, A-CAG (-0.041, p=1.0) is not — the Q depletion is entirely on CAA, the more frequent Q codon in C. elegans.
  - A-CGT (+0.367, near-floor) and A-CGC (+0.250) point the same direction as A-R aa-level (+0.239, near-floor) — Arg signal carried by both major synonyms.
  - A-TCT (-0.281, floor) at codon level vs A-S (-0.122, p=0.114) at aa level — TCT is one of 6 Ser codons; aa-level Ser depletion is partial-codon-driven.
  - A-TTT (-0.465, floor) at codon level vs A-F (+0.075, p=1.0) at aa level — TTT/TTC split for Phe. TTT depleted; is TTC enriched?  Cross-check.
  - Same biphasic question as the aa file: is d10_vs_d5 the strongest codon contrast (min p_adj 0.436) because of a transient day-5 dip in these particular codons that recovers by day_10?
