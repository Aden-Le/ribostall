---
input_csv: results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d5_vs_d0.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_5 vs day_0
test_type_source: user-confirmed
synced_from_olive_qmd: 2026-06-08
n_tests: 60
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.2857
p_floor: 0.02857
pseudoreplicated: false
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
caveats_considered:
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: denied, why: "scipy's MW could in principle fall back to the asymptotic Z approximation under tied per-replicate occupancy, shifting raw p off the exact discrete bound; ruled out empirically — the 60 tests collapse to 8 distinct p-values all on the n=4-vs-4 exact two-sided grid (k/70), every U_stat integer (0-16), and a per-replicate branch audit (day_5 vs day_0) found 0/60 rank ties, the exact branch selected for all 60, and forcing the asymptotic branch shifts raw p by at most 0.0305 (median 0.0106) with 0 raw-p<0.05 decision flips and 0 FDR hits either branch.", user_note: "Adopted at readback to match the aa sisters aa_wilcoxon_condition + aa_wilcoxon_timepoint_d10_vs_d0 + aa_wilcoxon_timepoint_d10_vs_d5 (user-approved family-wide 2026-06-07)."}
headline: "0 of 60 day_5-vs-day_0 amino-acid MW tests significant at FDR<0.05 (min p_adj 0.2857); 5 cells reach raw p<0.05, all sitting exactly at the n=4 MW floor (0.02857); coordinated floor-limited null, the absence of FDR hits is floor-blocking not a biological negative. Largest-magnitude exploratory leads: P-site G (-0.2029) depleted, A-site M (+0.1567) enriched; largest cell overall A-site W (-0.2413) sits above the floor (raw p 0.1143)."
user_directives:
  - "(triage, family-batched) 'Confirm test type? Columns median_day_5/median_day_0/log2_FC/U_stat/p_value/p_adj at n=4 vs 4; my read two-sample Wilcoxon rank-sum (Mann-Whitney U), day_5 vs day_0.' -> 'Confirm Mann-Whitney U (all 6)'"
  - "(triage, family-batched) 'How firmly should these read? 0 FDR hits; raw-p hits only at the n=4 floor.' -> 'Mixed' (firm coordinated-floor-blocked-null headline; floor-level cells reported as nominal-only exploratory leads, symmetric on enriched/depleted)"
  - "(triage, family-batched) 'Per-CSV flags beyond the 4 family caveats?' -> 'Inline stop annotation + formal stop-codon-instability caveat on codon files' (neither applies to this AA file: the AA path drops stop windows, so no stop codon is present here)"
  - "(triage, family-batched) 'Spotlight any site/feature/group?' -> 'No spotlight'"
  - "(readback) 'Reconciled shared content from the corrected .qmd on 2026-06-08' -> 'Adopted Olive 6 per-direction Top-hits sub-tables + raw p_value column (bare AA codes kept, Site P/E under <details>); applied the Stage-5 headline wording fix As nominal-only exploratory leads -> As exploratory leads only (the floor cells are nominal-only but the largest-overall A-site W sits above the floor), keeping the (Mixed framing) tag; aligned Numbers site order E/P/A -> A/P/E; adopted the asymptotic-with-ties entry into Considered but not applicable to match the aa sisters. Olive-only floor-arithmetic + Site-E reframes not imported (Dylan mw-floor-blocking already carries the per-site k > 11.4 math); Biological interpretation, composite figure, and plots not imported.'"
---

# Interpretation — aa_wilcoxon_timepoint_d5_vs_d0

> Source: `results/global_occupancy/analysis_corrected/aa_wilcoxon_timepoint_d5_vs_d0.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U (two-sample Wilcoxon rank-sum), day_5 vs day_0 (source: user-confirmed)

## User directives
- (triage, family-batched) "Confirm test type? Columns are median_day_5, median_day_0, log2_FC, U_stat, p_value, p_adj (day_5 vs day_0, n=4 vs 4 pooling 2 BWM + 2 control reps per timepoint)." → "Confirm Mann-Whitney U (all 6)"
- (triage, family-batched) "How firmly should these read? 0 FDR hits across the family; raw-p hits only at the n=4 MW floor (0.0286)." → "Mixed" (firm coordinated-floor-blocked-null headline; note floor-level cells as nominal-only exploratory leads; symmetric enriched/depleted per A.2.2)
- (triage, family-batched) "Per-CSV flags beyond the 4 locked family caveats?" → "Inline stop annotation + formal stop-codon-instability caveat on the codon files." Neither applies to this AA file — the amino-acid path drops stop windows, so this file has no stop codon (60 tests = 20 features × 3 sites).
- (triage, family-batched) "Spotlight any site/feature/group?" → "No spotlight"
- (readback) "Reconciled shared content from the corrected .qmd on 2026-06-08" → adopted Olive's 6 per-direction Top-hits sub-tables and raw `p_value` column (bare AA codes kept, Site P/E under `<details>`); applied the Stage-5 headline wording fix "As nominal-only exploratory leads" → "As exploratory leads only" (the floor cells are nominal-only but the largest-overall A-site W sits above the floor), keeping the "(Mixed framing)" tag; aligned the Numbers site order E/P/A → A/P/E; adopted the asymptotic-with-ties entry under Considered but not applicable to match the aa sisters `aa_wilcoxon_condition`, `aa_wilcoxon_timepoint_d10_vs_d0`, and `aa_wilcoxon_timepoint_d10_vs_d5`. The Olive-only floor-arithmetic and "Site E weakest signal" reframes were not imported (Dylan's `mw-floor-blocking` caveat already carries the per-site `k > 11.4` math); the Biological interpretation, composite figure, and plots were not imported.

## Headline
0 of 60 day_5-vs-day_0 amino-acid tests are significant at FDR<0.05; minimum adjusted p is 0.2857. Five cells reach raw p<0.05, and all five sit exactly at the n=4-vs-n=4 two-sided Mann-Whitney floor (0.02857). The absence of FDR hits is a structural floor-blocking outcome, not a biological negative. This is a coordinated floor-limited null. As exploratory leads only (Mixed framing): the largest-magnitude floor cells are P-site G (`log2_FC` -0.2029, depleted in day_5) and A-site M (+0.1567, enriched); the largest-magnitude cell overall, A-site W (-0.2413), sits above the floor (raw p 0.1143).

## Top hits

Tables are split per site (decoding-site order A / P / E) and per direction (enriched / depleted). Two p columns are shown: `p_value` is the raw Mann-Whitney p and `p_adj` is BH-corrected within each site's 20-amino-acid family; rows are ranked by raw `p_value` ascending with `|log2_FC|` descending as the tiebreaker (the family clears no FDR threshold, so selection is by raw p). Every cell flagged `nominal-only` (five in this file) sits exactly at the n=4 two-sided MW floor (raw p 0.02857): its raw significance is the maximum the test can produce at this N, not evidence of a strong effect.

### Site A — enriched

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| M | +0.1567 | 0.0286 | 0.2857 | nominal-only |
| S | +0.0597 | 0.1143 | 0.4571 |  |
| N | +0.2178 | 0.2000 | 0.6234 |  |
| D | +0.1073 | 0.3429 | 0.6234 |  |
| F | +0.1019 | 0.3429 | 0.6234 |  |

### Site A — depleted

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| P | -0.0866 | 0.0286 | 0.2857 | nominal-only |
| W | -0.2413 | 0.1143 | 0.4571 |  |
| G | -0.1770 | 0.1143 | 0.4571 |  |
| A | -0.0903 | 0.3429 | 0.6234 |  |
| L | -0.0425 | 0.3429 | 0.6234 |  |

<details>
<summary>Site P and Site E</summary>

### Site P — enriched

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| S | +0.0783 | 0.0286 | 0.2857 | nominal-only |
| C | +0.1311 | 0.1143 | 0.5714 |  |
| E | +0.0936 | 0.3429 | 0.8831 |  |
| N | +0.0905 | 0.3429 | 0.8831 |  |
| L | +0.0604 | 0.4857 | 0.8831 |  |

### Site P — depleted

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| G | -0.2029 | 0.0286 | 0.2857 | nominal-only |
| P | -0.0888 | 0.1143 | 0.5714 |  |
| K | -0.1539 | 0.3429 | 0.8831 |  |
| R | -0.0654 | 0.4857 | 0.8831 |  |
| A | -0.0437 | 0.8857 | 0.9841 |  |

### Site E — enriched

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| M | +0.0727 | 0.0286 | 0.5714 | nominal-only |
| S | +0.1506 | 0.1143 | 0.6667 |  |
| C | +0.1122 | 0.2000 | 0.6667 |  |
| H | +0.0450 | 0.2000 | 0.6667 |  |
| L | +0.0725 | 0.3429 | 0.8571 |  |

### Site E — depleted

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| P | -0.2195 | 0.2000 | 0.6667 |  |
| G | -0.0976 | 0.2000 | 0.6667 |  |
| K | -0.1014 | 0.3429 | 0.8571 |  |
| I | -0.0516 | 0.6857 | 1.0000 |  |
| Y | -0.0422 | 0.8857 | 1.0000 |  |

</details>

## Numbers at a glance
- `n_tests`: 60 (20 per site, A/P/E)
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.2857 (sites A and P)
- `min raw p`: 0.0286 (= the n=4 floor; 5 cells tie here)
- `p_floor`: 0.02857 (theoretical two-sided Mann-Whitney floor for n=4 vs n=4 = 2/C(8,4)). Five cells reach it, the most of the three AA contrasts, yet still 0 FDR hits.
- Per site: site A min p_adj 0.2857; site P min p_adj 0.2857; site E min p_adj 0.5714.

## Methods
Columns present: `site`, `amino_acid`, `median_day_5`, `median_day_0`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Dylan proposed a two-sample Wilcoxon rank-sum (Mann-Whitney U) comparing day_5 vs day_0 occupancy per (site, amino acid) at n=4 vs n=4; user confirmed. The `U_stat` column and the paired median columns are consistent with that test. P-correction column is `p_adj` (BH within each site's 20-feature family, per `merge_global_occupancy_analysis.py`). The test compares the two timepoint distributions per cell; it does not separate the BWM and control reps pooled into each n=4 arm, and at this N FDR<0.05 needs ~12+ features tied at the floor at once.

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — the n=4-vs-n=4 two-sided MW floor is exactly 0.02857; with a per-site BH family of 20, FDR<0.05 needs k > 11.4 cells tied at the floor. Five cells reach the floor here, still far short.
- **condition-pooled-confound** (family-wide) — each n=4 arm pools 2 BWM + 2 control reps; divergent BWM-vs-control day_5-vs-day_0 responses would be masked by pooling.
- **n=4-low-power** (family-wide) — n=4 vs n=4 MW has very low power; this null is weakly informative.
- **p-floor-aware-headline** (family-wide) — the headline states min raw p ≈ 0.0286 and why "no FDR hits" is uninformative here, not a biological negative.

### Considered but not applicable
- **asymptotic-with-ties** (per-CSV, considered) — concern that scipy's MW could fall back to the asymptotic Z approximation under tied per-replicate occupancy values, shifting raw p off the exact discrete bound. Ruled out empirically: the 60 (site, amino acid) tests collapse to 8 distinct p-values, every `U_stat` integer (range 0-16), and all 8 fall exactly on the n=4-vs-4 exact two-sided MW grid (k/70 for integer k) — the exact-branch signature with no ties. A per-replicate branch audit of the same n=4-vs-4 design (day_5 vs day_0) found 0/60 rank ties, `method='auto'` selected the exact branch for all 60, recomputed p matched the pipeline to ~8e-17, and forcing `method='asymptotic'` shifts raw p by at most 0.0305 (median 0.0106), flips zero raw-p<0.05 decisions, and leaves 0 FDR hits on either branch. (Audit values are derivation/audit-sourced, not raw-CSV columns.)

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs to reconcile: the codon resolution of this contrast (`codon_wilcoxon_timepoint_d5_vs_d0.csv`) and the other two contrasts; see the `## Joint-reading suggestions` block in [`_INDEX.md`](_INDEX.md).
- This contrast (and its codon sister) carries the most floor cells in the family, leaning toward depletions (P-site G, A-site P) and enrichments (A-site M, P-site S). Falsifier: do these same cells move with consistent direction and additive magnitude in `*_d10_vs_d0` (= d5_vs_d0 + d10_vs_d5), or is the apparent day_5-vs-day_0 structure non-additive / discreteness-driven?
- Falsifier on the within-condition Fisher: do P-site G depletion and A-site M enrichment reappear at FDR there (where N is large), or only here at the n=4 floor? Consistency across designs would let Chumeng decide whether to elevate; appearance only in the large-N Fisher would point to large-N amplification of a baseline difference.
