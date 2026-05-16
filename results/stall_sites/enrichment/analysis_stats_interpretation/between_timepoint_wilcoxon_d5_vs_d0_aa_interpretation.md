---
input_csv: results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d5_vs_d0_aa.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U / Wilcoxon rank-sum (two-sided)
test_type_source: user-confirmed
n_tests: 60
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.5714285714285714
p_floor: 0.02857142857142857
pseudoreplicated: null
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "weighted_log2_enrichment-absent", proposed_by: dylan, status: confirmed, why: "Effect column is `log2_FC` of medians only."}
  - {label: "small-bh-family-discreteness", proposed_by: dylan, status: confirmed, why: "BH adjusts only 20 raw-p per site. Single floor row at P-C with one near-floor companion P-H gives BH wall = 0.5714, exactly observed."}
headline: "Firm null at FDR<0.05 (0/60) for AA-level d5-vs-d0 MW with reps pooled across BWM and control per timepoint; one floor row at P-site C (+0.249, p_adj=0.571) and one near-floor at P-site H (-0.144, p_adj=0.571). Site E is completely flat (min p_adj=1.0). Min raw p = 0.0286 (n=4 vs n=4 floor); 'no FDR hits' is structural per the locked floor caveat."
user_directives:
  - "(triage, batched across all 6 family members) Test type confirmation — `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → `Confirm`."
  - "(triage, batched across the 3 aa files) CSV-specific caveats → `weighted_log2_enrichment-absent`, `small-bh-family-discreteness` confirmed."
---

# Interpretation — between_timepoint_wilcoxon_d5_vs_d0_aa

> Source: `results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d5_vs_d0_aa.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage, batched) Test type: `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → "Confirm".
- (triage, batched 3 aa files) CSV-specific caveats: `weighted_log2_enrichment-absent`, `small-bh-family-discreteness` confirmed.

## Headline
Firm null at FDR<0.05 (0/60) for AA-level day_5 vs day_0 MW with BWM and control reps pooled within each timepoint (n=4 per side). One floor row across the file: P-site C (+0.249, raw p=0.0286, p_adj=0.571), with a P-site H near-floor companion (-0.144, raw p=0.0571, p_adj=0.571). Site E is completely flat (min p_adj=1.0); site A min p_adj=0.8. The day_5 vs day_0 contrast carries the *least* signal of the three pairwise contrasts in this family — every site has weaker signal here than in the corresponding d10_vs_d5 or d10_vs_d0 file. Treat as exploratory; the only feature with even a floor-level signal is P-site Cysteine, also flagged at d10_vs_d0_aa.

## Top hits

### P site (headline group — only site with a floor row; min p_adj=0.571)

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | C | +0.249 | 0.571 | floor |
| enriched | S | +0.152 | 0.883 |  |
| enriched | E | +0.268 | 0.883 |  |
| enriched | F | +0.086 | 0.883 |  |
| enriched | Q | +0.166 | 0.883 |  |
| depleted | H | -0.144 | 0.571 | nominal-only |
| depleted | K | -0.192 | 0.883 |  |
| depleted | G | -0.193 | 0.883 |  |
| depleted | W | -0.252 | 0.883 |  |
| depleted | M | -0.383 | 0.914 |  |

<details>
<summary>A site (no floor rows; min p_adj = 0.8)</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | I | +0.188 | 0.800 |  |
| enriched | D | +0.222 | 0.800 |  |
| enriched | N | +0.618 | 0.800 |  |
| enriched | E | +0.069 | 0.857 |  |
| enriched | Q | +0.056 | 0.857 |  |
| depleted | Y | -0.412 | 0.800 |  |
| depleted | W | -0.671 | 0.800 |  |
| depleted | L | -0.097 | 0.857 |  |
| depleted | A | -0.243 | 0.883 |  |
| depleted | P | -0.074 | 0.883 |  |

</details>

<details>
<summary>E site (completely flat; min p_adj = 1.0)</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | Q | +0.482 | 1.0 |  |
| enriched | S | +0.300 | 1.0 |  |
| enriched | E | +0.212 | 1.0 |  |
| enriched | C | +0.131 | 1.0 |  |
| enriched | N | +0.040 | 1.0 |  |
| depleted | P | -0.510 | 1.0 |  |
| depleted | K | -0.201 | 1.0 |  |
| depleted | M | -0.170 | 1.0 |  |
| depleted | W | -0.143 | 1.0 |  |
| depleted | A | -0.092 | 1.0 |  |

</details>

## Numbers at a glance
- `n_tests`: 60
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.5714 (P-site C floor + P-site H near-floor)
- `min raw-p`: 0.02857 (= MW exact floor); 1 row at the floor (P-C +0.249)
- `p_floor`: 0.02857 (n=4 vs n=4 two-sided exact)
- Per-site BH families:
  - A site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.800
  - E site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 1.000
  - P site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.571

## Methods
MW rank-sum two-sided on per-replicate frequencies, n=4 day_5 (BWM_day5_rep2/3 + control_day5_rep2/3) vs n=4 day_0, BH-FDR per site (each site = 20-AA family). Test answers "do day_5 and day_0 reps differ in per-rep AA frequency at this site?" with BWM and control reps pooled within each timepoint.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide).
- **condition-pooled-confound** (family-wide).
- **n=4-low-power** (family-wide).
- **p-floor-aware-headline** (family-wide).
- **weighted_log2_enrichment-absent** (per-CSV).
- **small-bh-family-discreteness** (per-CSV) — observed min p_adj=0.571 is the BH wall for one floor hit in a 20-test family, exactly as expected.

### Considered but not applicable
*(Dylan did not propose any further per-CSV caveats here.)*

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs: `between_timepoint_wilcoxon_d5_vs_d0_codon.csv` (codon resolution, same contrast), and the d10_vs_d0 / d10_vs_d5 pairs at both resolutions.
- Open questions Chumeng should resolve at synthesis time:
  - **P-site C is the only feature consistently at the MW floor in this family.** Does the d0→d5→d10 trajectory across the three contrasts support a single coherent pattern, or could the additivity be coincidence given n=4 floor effects? Per-cell numbers: d10_vs_d0 +0.405, this file (d5_vs_d0) +0.249, d10_vs_d5 (P-C raw p=0.0571) +0.156. The additive arithmetic across the 3 contrasts (+0.249 + +0.156 ≈ +0.405) is consistent with several patterns including monotonic accumulation; Chumeng should weigh against the alternatives at synthesis time.
  - d5_vs_d0 is the flattest contrast in the family. Possible reasons (Chumeng to weigh): smaller frequency change between consecutive early timepoints, or n=4 floor noise. Useful constraint on what `per_timepoint_fisher` should show.
  - E site flat (min p_adj=1.0): does this reproduce in `between_timepoint_wilcoxon_d5_vs_d0_codon` at the codon level, or does codon resolution expose hits aa aggregation hides?
