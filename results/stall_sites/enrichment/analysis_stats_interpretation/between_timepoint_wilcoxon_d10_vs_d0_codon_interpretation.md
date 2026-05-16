---
input_csv: results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d0_codon.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U / Wilcoxon rank-sum (two-sided)
test_type_source: user-confirmed
n_tests: 183
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.4979591836734693
p_floor: 0.02857142857142857
pseudoreplicated: null
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "weighted_log2_enrichment-absent", proposed_by: dylan, status: confirmed, why: "Effect column is `log2_FC` of medians only; no count-weighted variant."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "~61 codons per site → BH-FDR family of ~61 vs ~20 at aa; with the same raw-p floor, codon BH wall is ~3x more stringent."}
  - {label: "low-count-rare-codon-instability", proposed_by: dylan, status: confirmed, why: "Rare codons (TTA, CTA, AGG, CGG, GCG, ATA) have median freqs < 0.2%; small absolute changes give large log2_FC swings. Top-|effect| codons here (TTA -1.20, GCA -1.07, GTG -0.90 at A; CTA -1.24, GCG -1.03 at P) all sit in this regime."}
headline: "Firm null at FDR<0.05 (0/183) for codon-level d10-vs-d0 MW with BWM and control reps pooled per timepoint; min raw p = 0.0286 (n=4 vs n=4 floor), 3 floor rows total — E-site TCC (+0.589), P-site GAG (+0.387), P-site TGC (+0.219). Min p_adj = 0.498 at A site (no floor row, but a 14-codon block tied at raw p=0.114 produces the BH wall). 'No FDR hits' is structural per the locked floor caveat, not a biological negative."
user_directives:
  - "(triage, batched across all 6 family members) Test type confirmation — `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → `Confirm`."
  - "(triage, batched across the 3 codon files) CSV-specific caveats → `weighted_log2_enrichment-absent`, `larger-bh-family`, `low-count-rare-codon-instability` confirmed."
---

# Interpretation — between_timepoint_wilcoxon_d10_vs_d0_codon

> Source: `results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d0_codon.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage, batched) Test type: `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → "Confirm".
- (triage, batched 3 codon files) CSV-specific caveats: `weighted_log2_enrichment-absent`, `larger-bh-family`, `low-count-rare-codon-instability` confirmed.

## Headline
Firm null at FDR<0.05 (0/183) for codon-level day_10 vs day_0 MW with BWM and control reps pooled within each timepoint (n=4 per side). Three floor rows in the whole file: E-TCC (+0.589), P-GAG (+0.387), P-TGC (+0.219). Min p_adj = 0.498 at site A — driven by a 14-codon block tied at raw p = 0.114, producing the BH wall 0.114 * 61/14 ≈ 0.498. Site A has the densest near-floor signal but no floor rows; sites E and P each have 1-2 floor rows but min p_adj climbs to 0.93 / 0.69. Treat as exploratory leads only; the per-timepoint Fisher (`per_timepoint_fisher_codon`) at day_0 and day_10 is the test that can actually resolve a BWM-vs-control codon shift at either timepoint.

## Top hits

### A site (headline group — min p_adj = 0.498, no floor rows but densest near-floor block)

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | CGC | +0.640 | 0.498 | nominal-only |
| enriched | AAC | +0.595 | 0.498 | nominal-only |
| enriched | ATC | +0.518 | 0.498 | nominal-only |
| enriched | ACC | +0.295 | 0.498 | nominal-only |
| enriched | CGT | +0.276 | 0.498 | nominal-only |
| depleted | TTA | -1.201 | 0.498 | nominal-only, low-count |
| depleted | GCA | -1.065 | 0.498 | nominal-only, low-count |
| depleted | GTG | -0.901 | 0.498 | nominal-only, low-count |
| depleted | CTG | -0.800 | 0.498 | nominal-only, low-count |
| depleted | GGC | -0.639 | 0.498 | nominal-only, low-count |

<details>
<summary>E site (1 floor row)</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | TCC | +0.589 | 0.930 | floor |
| enriched | CTC | +0.490 | 0.930 |  |
| enriched | CAG | +0.267 | 0.930 |  |
| enriched | TGC | +0.220 | 0.930 |  |
| enriched | GCT | +0.113 | 0.930 |  |
| depleted | GCG | -1.204 | 0.930 | nominal-only, low-count |
| depleted | AGG | -0.767 | 0.930 | low-count |
| depleted | CGA | -0.678 | 0.930 |  |
| depleted | AAA | -0.468 | 0.930 |  |
| depleted | ATG | -0.384 | 0.930 |  |

</details>

<details>
<summary>P site (2 floor rows)</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | GAG | +0.387 | 0.688 | floor |
| enriched | TGC | +0.219 | 0.688 | floor |
| enriched | CCC | +0.298 | 0.688 | nominal-only |
| enriched | GCC | +0.221 | 0.688 | nominal-only |
| enriched | GGG | +1.147 | 0.688 | low-count |
| depleted | CTA | -1.242 | 0.688 | low-count |
| depleted | GCG | -1.031 | 0.688 | nominal-only, low-count |
| depleted | GCA | -0.698 | 0.688 |  |
| depleted | ATT | -0.578 | 0.688 |  |
| depleted | GGT | -0.146 | 0.688 |  |

</details>

## Numbers at a glance
- `n_tests`: 183
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.498 (A site, 14-codon block tied at raw p = 0.114)
- `min raw-p`: 0.02857 (= MW exact floor); 3 rows at the floor (E-TCC +0.589, P-GAG +0.387, P-TGC +0.219)
- `p_floor`: 0.02857 (n=4 vs n=4 two-sided exact)
- Per-site BH families:
  - A site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.498 (no floor rows)
  - E site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.930 (1 floor row)
  - P site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.688 (2 floor rows)

## Methods
Dylan proposed Mann-Whitney U / Wilcoxon rank-sum two-sided on per-replicate frequencies, n=4 day_10 (BWM_day10_rep2/3 + control_day10_rep2/3) vs n=4 day_0, BH-FDR per site (each E/P/A site = 61-codon family); user confirmed. Effect column is `log2_FC` of medians; test statistic is `U_stat`. The test answers "do day_10 and day_0 reps differ in per-rep codon frequency at this site?", with BWM and control reps pooled within each timepoint. Does *not* answer "BWM-vs-control at any single day" (that is `per_timepoint_fisher_codon`) and does *not* answer "BWM moves from day_0 to day_10 holding condition fixed" (that is `timepoint_fisher_within_condition_d10_vs_d0_codon`).

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — MW two-sided exact p-floor for n=4 vs n=4 = 0.02857. With 61 codons per site, BH wall for one floor hit is 61 * 0.0286 ≈ 1.74; for two floor hits at the same site, 0.85; for three, 0.57. Even all 61 codons tied at the floor would BH to 0.0286 — the only path to FDR<0.05.
- **condition-pooled-confound** (family-wide) — n=4 per timepoint = 2 BWM + 2 control reps; condition-by-time interactions cancel inside MW.
- **n=4-low-power** (family-wide) — null is weakly informative.
- **p-floor-aware-headline** (family-wide) — the headline above explicitly states the floor.
- **weighted_log2_enrichment-absent** (per-CSV) — `log2_FC` of medians only.
- **larger-bh-family** (per-CSV) — 61 vs 20 means the codon BH wall is ~3x further from FDR<0.05 for any given count of floor hits than the aa BH wall for the same contrast. Empirically: aa min p_adj = 0.571 (1 floor row), codon min p_adj = 0.498 (no floor rows but a 14-codon block).
- **low-count-rare-codon-instability** (per-CSV) — top |effect| ranks at site A (TTA -1.20, GCA -1.07, GTG -0.90, CTG -0.80) and site P (CTA -1.24, GCG -1.03, GGG +1.15) are dominated by codons with median freq < 0.5%. A few absolute counts moving in/out of the stall set produces large log2_FC; do not over-read.

### Considered but not applicable
*(Dylan did not propose any further per-CSV caveats here.)*

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs in this family that should be reconciled: `between_timepoint_wilcoxon_d10_vs_d0_aa.csv` (aa resolution, same contrast), `between_timepoint_wilcoxon_d10_vs_d5_aa/codon.csv`, `between_timepoint_wilcoxon_d5_vs_d0_aa/codon.csv`.
- Open questions Chumeng should resolve at synthesis time:
  - Codon-vs-aa for the day_10 vs day_0 contrast: P-aa C (+0.405, floor) — do its codons TGT and TGC behave alike here? TGC is at the floor (+0.219), TGT is at p=1.0 (+0.238). So the aa-level P-C signal is driven by the synonymous codon TGC, not TGT. Codon usage shift at P-site cysteine.
  - P-aa E (+0.373, near-floor) — synonyms GAA (raw p=1.0, +0.021) and GAG (floor, +0.387). The aa-level E signal is entirely on GAG, not GAA. Codon usage shift, not aa shift.
  - E-site TCC (+0.589, floor): is this the codon-level fingerprint of E-aa S (which was unremarkable at aa level)? S has 6 codons (TCT, TCC, TCA, TCG, AGT, AGC); only TCC is at the floor.
  - Site A's 14-codon "near-significant" block: if the same codons appear at floor in `per_timepoint_fisher_codon` (day_0 or day_10) with consistent direction → BWM-vs-control codon signal that MW couldn't resolve. If they don't → likely a between-day frequency drift independent of perturbation.
