---
input_csv: results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d10_vs_d5.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_10 vs day_5
test_type_source: user-confirmed
synced_from_olive_qmd: 2026-06-07
n_tests: 60
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.5714
p_floor: 0.02857
pseudoreplicated: false
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
caveats_considered:
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: denied, why: "scipy's MW could in principle fall back to the asymptotic Z approximation under tied per-replicate occupancy, shifting raw p off the exact discrete bound; ruled out empirically — the 60 tests collapse to 9 distinct p-values all on the n=4-vs-4 exact two-sided grid (k/70), every U_stat integer (0-15), and a per-replicate branch audit found 0/60 rank ties, the exact branch selected for all 60, and forcing the asymptotic branch shifts raw p by at most 0.0305 (median 0.0152) with 0 raw-p<0.05 decision flips and 0 FDR hits either branch.", user_note: "Adopted at readback to match the aa sisters aa_wilcoxon_condition + aa_wilcoxon_timepoint_d10_vs_d0 (user-approved family-wide 2026-06-07)."}
headline: "0 of 60 day_10-vs-day_5 amino-acid MW tests significant at FDR<0.05 (min p_adj 0.5714); a single cell reaches raw p<0.05 — P-site I (-0.1469), sitting exactly at the n=4 MW floor (0.02857); coordinated floor-limited null, the absence of FDR hits is floor-blocking not a biological negative. Largest enriched cell P-site G (+0.1524) is just above the floor (raw p 0.0571); site E is the flattest site (min p_adj 1.0)."
user_directives:
  - "(triage, family-batched) 'Confirm test type? Columns median_day_10/median_day_5/log2_FC/U_stat/p_value/p_adj at n=4 vs 4; my read two-sample Wilcoxon rank-sum (Mann-Whitney U), day_10 vs day_5.' -> 'Confirm Mann-Whitney U (all 6)'"
  - "(triage, family-batched) 'How firmly should these read? 0 FDR hits; raw-p hits only at the n=4 floor.' -> 'Mixed' (firm coordinated-floor-blocked-null headline; floor-level cells reported as nominal-only exploratory leads, symmetric on enriched/depleted)"
  - "(triage, family-batched) 'Per-CSV flags beyond the 4 family caveats?' -> 'Inline stop annotation + formal stop-codon-instability caveat on codon files' (neither applies to this AA file: the AA path drops stop windows, so no stop codon is present here)"
  - "(triage, family-batched) 'Spotlight any site/feature/group?' -> 'No spotlight'"
  - "(readback) 'Reconciled shared content from the corrected .qmd on 2026-06-07' -> 'Adopted Olive 6 per-direction Top-hits sub-tables + raw p_value column (bare AA codes kept); applied the Stage-5 headline wording fix As a nominal-only exploratory lead -> As exploratory leads only (only P-site I is nominal-only; P-site G and site E sit above the floor), keeping the (Mixed framing) tag; aligned Numbers site order to A/P/E; adopted the asymptotic-with-ties entry into Considered but not applicable to match the aa sisters.'"
---

# Interpretation — aa_wilcoxon_timepoint_d10_vs_d5

> Source: `results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d10_vs_d5.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_10 vs day_5 (source: user-confirmed)

## User directives
- (triage, family-batched) "Confirm test type? Columns are median_day_10, median_day_5, log2_FC, U_stat, p_value, p_adj (day_10 vs day_5, n=4 vs 4 pooling 2 BWM + 2 control reps per timepoint)." → "Confirm Mann-Whitney U (all 6)"
- (triage, family-batched) "How firmly should these read? 0 FDR hits across the family; raw-p hits only at the n=4 MW floor (0.0286)." → "Mixed" (firm coordinated-floor-blocked-null headline; note floor-level cells as nominal-only exploratory leads; symmetric enriched/depleted per A.2.2)
- (triage, family-batched) "Per-CSV flags beyond the 4 locked family caveats?" → "Inline stop annotation + formal stop-codon-instability caveat on the codon files." Neither applies to this AA file — the amino-acid path drops stop windows, so this file has no stop codon (60 tests = 20 features × 3 sites).
- (triage, family-batched) "Spotlight any site/feature/group?" → "No spotlight"
- (readback) "Reconciled shared content from the corrected .qmd on 2026-06-07" → adopted Olive's 6 per-direction Top-hits sub-tables and raw `p_value` column (bare AA codes kept); applied the Stage-5 headline wording fix "As a nominal-only exploratory lead" → "As exploratory leads only" (only P-site I is nominal-only; P-site G and site E sit above the floor), keeping the "(Mixed framing)" tag; aligned the Numbers site order to A/P/E; adopted the asymptotic-with-ties entry under Considered but not applicable to match the aa sisters `aa_wilcoxon_condition` and `aa_wilcoxon_timepoint_d10_vs_d0`.

## Headline
0 of 60 day_10-vs-day_5 amino-acid tests are significant at FDR<0.05; minimum adjusted p is 0.5714. A single cell reaches raw p<0.05 — P-site I (`log2_FC` -0.1469) — and it sits exactly at the n=4-vs-n=4 two-sided Mann-Whitney floor (0.02857). The absence of FDR hits is floor-blocking, not a biological negative. This is a coordinated floor-limited null. As exploratory leads only (Mixed framing): P-site I (-0.1469, depleted in day_10) is the only nominally-significant cell; the largest-magnitude enriched cell, P-site G (+0.1524), sits just above the floor (raw p 0.0571); site E is the flattest site (every E cell has p_adj 1.0).

## Top hits

Tables are split per site (decoding-site order A / P / E) and per direction (enriched / depleted). Two p columns are shown: `p_value` is the raw Mann-Whitney p and `p_adj` is BH-corrected within each site's 20-amino-acid family; rows are ranked by raw `p_value` ascending with `|log2_FC|` descending as the tiebreaker (the family clears no FDR threshold, so selection is by raw p). The single cell flagged `nominal-only` (P-site I) sits exactly at the n=4 two-sided MW floor (raw p 0.02857): its raw significance is the maximum the test can produce at this N, not evidence of a strong effect.

### Site A — enriched

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| R | +0.0463 | 0.1143 | 0.8571 |  |
| G | +0.1299 | 0.3429 | 0.8571 |  |
| C | +0.0361 | 0.3429 | 0.8571 |  |
| S | +0.0152 | 0.4857 | 0.9796 |  |
| W | +0.1063 | 0.6857 | 0.9796 |  |

### Site A — depleted

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| K | -0.1202 | 0.2000 | 0.8571 |  |
| M | -0.0186 | 0.2000 | 0.8571 |  |
| Q | -0.1162 | 0.3429 | 0.8571 |  |
| N | -0.0671 | 0.3429 | 0.8571 |  |
| T | -0.0373 | 0.3429 | 0.8571 |  |

<details>
<summary>Site P and Site E</summary>

### Site P — enriched

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| G | +0.1524 | 0.0571 | 0.5714 |  |
| E | +0.0840 | 0.3429 | 0.9143 |  |
| D | +0.1213 | 0.4857 | 0.9143 |  |
| P | +0.0418 | 0.4857 | 0.9143 |  |
| C | +0.0349 | 0.4857 | 0.9143 |  |

### Site P — depleted

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| I | -0.1469 | 0.0286 | 0.5714 | nominal-only |
| L | -0.0740 | 0.3429 | 0.9143 |  |
| S | -0.0374 | 0.3429 | 0.9143 |  |
| M | -0.0649 | 0.4857 | 0.9143 |  |
| V | -0.0176 | 0.4857 | 0.9143 |  |

### Site E — enriched

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| T | +0.0325 | 0.3429 | 1.0000 |  |
| P | +0.0894 | 0.4857 | 1.0000 |  |
| S | +0.0671 | 0.4857 | 1.0000 |  |
| H | +0.0236 | 0.6857 | 1.0000 |  |
| A | +0.0185 | 0.6857 | 1.0000 |  |

### Site E — depleted

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| R | -0.0590 | 0.6857 | 1.0000 |  |
| E | -0.0388 | 0.8857 | 1.0000 |  |
| C | -0.0320 | 0.8857 | 1.0000 |  |
| V | -0.0267 | 0.8857 | 1.0000 |  |
| N | -0.0034 | 0.8857 | 1.0000 |  |

</details>

## Numbers at a glance
- `n_tests`: 60 (20 per site, A/P/E)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.5714 (site P)
- `min raw p`: 0.0286 (= the n=4 floor; 1 cell, P-site I)
- `p_floor`: 0.02857 (theoretical two-sided Mann-Whitney floor for n=4 vs n=4 = 2/C(8,4)). One cell reaches it; site E does not reach it at all (min raw p there is higher, all p_adj 1.0).
- Per site: site A min p_adj 0.8571; site P min p_adj 0.5714; site E min p_adj 1.0000.

## Methods
Columns present: `site`, `amino_acid`, `median_day_10`, `median_day_5`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing day_10 vs day_5 occupancy per (site, amino acid) at n=4 vs n=4; user confirmed. The `U_stat` column and the paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's 20-feature family, per `merge_global_occupancy_analysis.py`). The test compares the two timepoint distributions per cell; it does not separate the BWM and control reps pooled into each n=4 arm, and at this N a single FDR-significant cell needs ~12+ features tied at the floor at once.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — the n=4-vs-n=4 two-sided MW floor is exactly 0.02857; with a per-site BH family of 20, clearing FDR<0.05 needs p_adj = 0.02857 × 20/k < 0.05, i.e. k > 11.4 cells tied at the floor at once. Only one cell reaches the floor here, so FDR<0.05 is mathematically out of reach.
- **condition-pooled-confound** (family-wide) — each n=4 arm pools 2 BWM + 2 control reps; divergent BWM-vs-control day_10-vs-day_5 responses would be masked by pooling.
- **n=4-low-power** (family-wide) — n=4 vs n=4 MW has very low power; this null is weakly informative.
- **p-floor-aware-headline** (family-wide) — the headline states min raw p ≈ 0.0286 and why "no FDR hits" is uninformative here, not a biological negative.

### Considered but not applicable
- **asymptotic-with-ties** (per-CSV, considered) — concern that scipy's MW could fall back to the asymptotic Z approximation under tied per-replicate occupancy values, shifting raw p off the exact discrete bound. Ruled out empirically: the 60 (site, amino acid) tests collapse to 9 distinct p-values, every `U_stat` integer (range 0-15), and all 9 fall exactly on the n=4-vs-4 exact two-sided MW grid (k/70 for integer k) — the exact-branch signature with no ties. A per-replicate branch audit of the same n=4-vs-4 design (day_10 vs day_5) found 0/60 rank ties, `method='auto'` selected the exact branch for all 60, recomputed p matched the pipeline to ~8e-17, and forcing `method='asymptotic'` shifts raw p by at most 0.0305 (median 0.0152), flips zero raw-p<0.05 decisions, and leaves 0 FDR hits on either branch. (Audit values are derivation/audit-sourced, not raw-CSV columns.)

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs to reconcile: the codon resolution of this contrast (`codon_wilcoxon_timepoint_d10_vs_d5.csv`) and the other two contrasts; see the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- This is the least-separated AA contrast in the family (only 1 floor cell; site E reaches p_adj 1.0 everywhere). Falsifier: does the within-condition Fisher d10_vs_d5 contrast also show the flattest signal of the three day-pairs, or does whole-transcriptome N reveal structure the n=4 MW could not?
- The single floor cell P-site I (-0.1469) — does it recur as a depletion in `*_d5_vs_d0` (which also shows a P-site Ile-adjacent pattern) or in the within-condition Fisher? Falsifier framing: a one-off floor cell that does not recur is most parsimoniously a discreteness artefact, not a day_10-vs-day_5 signal.
