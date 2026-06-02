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
synced_from_olive_qmd: 2026-06-01
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "weighted_log2_enrichment-absent", proposed_by: dylan, status: confirmed, why: "Effect column is `log2_FC` of medians only; no count-weighted variant."}
  - {label: "small-bh-family-discreteness", proposed_by: dylan, status: confirmed, why: "BH adjusts only 20 raw-p per site. Single floor row at P-C with one near-floor companion P-H gives BH wall = 0.5714, exactly observed."}
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: ruled_out, why: "Empirically verified via scripts/_for_claude_mw_branch_audit.py --design between_timepoint --timepoints day5 day0 --level aa: 0/60 (site, AA) tests have rank ties in the pooled 8-element sample; scipy mannwhitneyu(method='auto') picked the exact branch for all 60 (recomputed p matches the pipeline CSV to ~8e-17); forcing method='asymptotic' shifts raw p by at most 0.0305 (median 0.0106), flips 0 raw-p<0.05 decisions, and leaves 0 hits at FDR<0.05 (asymptotic per-site min p_adj A/P/E = 0.776/0.606/1.000 vs as-shipped exact 0.800/0.571/1.000)."}
headline: "No statistically significant differences at FDR<0.05 (0/60) for AA-level d5-vs-d0 MW with reps pooled across BWM and control per timepoint; one floor row at P-site C (+0.249, raw p=0.0286, p_adj=0.571) and one near-floor at P-site H (-0.144, raw p=0.0571, p_adj=0.571). Site E is completely flat (min p_adj=1.0); site A min p_adj=0.8. The d5-vs-d0 contrast carries the least signal of the three pairwise contrasts in this family. Min raw p = 0.0286 (n=4 vs n=4 floor); 'no FDR hits' is structural per the locked floor caveat."
user_directives:
  - "(triage, batched across all 6 family members) Test type confirmation — `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → `Confirm`."
  - "(triage, batched across the 3 aa files) CSV-specific caveats → `weighted_log2_enrichment-absent`, `small-bh-family-discreteness` confirmed."
  - "(readback) Reconciled shared content from the corrected .qmd on 2026-06-01 → adopted the six per-(site, direction) Top-hits sub-tables in A/P/E order with the added raw `p_value` column, de-jargoned the Headline opener, and synced the `asymptotic-with-ties` ruled-out caveat; every number enumerated and verified against the .qmd/CSV, no values changed."
---

# Interpretation — between_timepoint_wilcoxon_d5_vs_d0_aa

> Source: `results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d5_vs_d0_aa.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage, batched) Test type: `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → "Confirm".
- (triage, batched 3 aa files) CSV-specific caveats: `weighted_log2_enrichment-absent`, `small-bh-family-discreteness` confirmed.
- (readback) Reconciled shared content from the corrected .qmd on 2026-06-01: adopted the six per-(site, direction) Top-hits sub-tables (A/P/E order) with the added raw `p_value` column, de-jargoned the Headline opener, and synced the `asymptotic-with-ties` ruled-out caveat. Every number enumerated and verified against the .qmd/CSV; no values changed.

## Headline
No statistically significant differences at FDR<0.05 (0/60) for AA-level day_5 vs day_0 MW with BWM and control reps pooled within each timepoint (n=4 per side). One floor row across the file: P-site C (+0.249, raw p=0.0286, p_adj=0.571), with a P-site H near-floor companion (-0.144, raw p=0.0571, p_adj=0.571). Site E is completely flat (min p_adj=1.0); site A min p_adj=0.8. The day_5 vs day_0 contrast carries the *least* signal of the three pairwise contrasts in this family — every site has weaker signal here than in the corresponding d10_vs_d5 or d10_vs_d0 file. Treat as exploratory; the only feature with even a floor-level signal is P-site Cysteine, also flagged at d10_vs_d0_aa.

## Top hits

Effect column is `log2_FC` (day_5/day_0 median ratio); `p_value` is the raw Mann-Whitney p; `p_adj` is BH-corrected per A/P/E site (20-AA family). Each site is split into one sub-table per sign of effect (positive `log2_FC` = day_5-enriched, negative = day_5-depleted); within each, up to 5 rows ranked by raw `p_value` ascending, `|log2_FC|` descending as the tiebreaker.

### A site — enriched (day_5 > day_0)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| I | +0.188 | 0.1143 | 0.800 |  |
| N | +0.618 | 0.2000 | 0.800 |  |
| D | +0.222 | 0.2000 | 0.800 |  |
| E | +0.069 | 0.3429 | 0.857 |  |
| Q | +0.056 | 0.3429 | 0.857 |  |

### A site — depleted (day_5 < day_0)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| Y | -0.412 | 0.1143 | 0.800 |  |
| W | -0.671 | 0.2000 | 0.800 |  |
| L | -0.097 | 0.3429 | 0.857 |  |
| A | -0.243 | 0.4857 | 0.883 |  |
| P | -0.074 | 0.4857 | 0.883 |  |

### P site — enriched (day_5 > day_0)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| C | +0.249 | 0.0286 | 0.571 | floor |
| S | +0.152 | 0.2000 | 0.883 |  |
| E | +0.268 | 0.3429 | 0.883 |  |
| F | +0.086 | 0.3429 | 0.883 |  |
| Q | +0.166 | 0.4857 | 0.883 |  |

### P site — depleted (day_5 < day_0)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| H | -0.144 | 0.0571 | 0.571 |  |
| K | -0.192 | 0.2000 | 0.883 |  |
| G | -0.193 | 0.3429 | 0.883 |  |
| W | -0.252 | 0.4857 | 0.883 |  |
| M | -0.383 | 0.6857 | 0.914 |  |

### E site — enriched (day_5 > day_0)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| Q | +0.482 | 0.1143 | 1.000 |  |
| S | +0.300 | 0.2000 | 1.000 |  |
| E | +0.212 | 0.3429 | 1.000 |  |
| N | +0.040 | 0.3429 | 1.000 |  |
| C | +0.131 | 0.4857 | 1.000 |  |

### E site — depleted (day_5 < day_0)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| P | -0.510 | 0.2000 | 1.000 |  |
| W | -0.143 | 0.3429 | 1.000 |  |
| M | -0.170 | 0.4857 | 1.000 |  |
| A | -0.092 | 0.4857 | 1.000 |  |
| K | -0.201 | 0.8857 | 1.000 |  |

## Numbers at a glance
- `n_tests`: 60
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.5714 (P-site C floor + P-site H near-floor)
- `min raw-p`: 0.02857 (= MW exact floor); 1 row at the floor (P-C +0.249)
- `p_floor`: 0.02857 (n=4 vs n=4 two-sided exact)
- Per-site BH families:
  - A site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.800
  - P site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.571
  - E site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 1.000

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
- **asymptotic-with-ties** (per-CSV, ruled out) — concern that `scipy.stats.mannwhitneyu` might fall back to the asymptotic Z approximation under tied per-rep frequencies. Empirically verified via `scripts/_for_claude_mw_branch_audit.py --design between_timepoint --timepoints day5 day0 --level aa`: 0/60 (site, AA) tests have rank ties in the pooled 8-element sample; scipy `method='auto'` picked the exact branch for all 60 (recomputed p matches the pipeline CSV to ~8e-17); forcing `method='asymptotic'` shifts raw p-values by at most 0.0305 (median 0.0106), flips 0 raw-p<0.05 decisions, and leaves 0 hits at FDR<0.05 (asymptotic per-site min p_adj A/P/E = 0.776/0.606/1.000 vs as-shipped exact 0.800/0.571/1.000). The branch choice does not affect any FDR-level conclusion in this file.

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs: `between_timepoint_wilcoxon_d5_vs_d0_codon.csv` (codon resolution, same contrast), and the d10_vs_d0 / d10_vs_d5 pairs at both resolutions.
- Open questions Chumeng should resolve at synthesis time:
  - **P-site C is the only feature consistently at the MW floor in this family.** Does the d0→d5→d10 trajectory across the three contrasts support a single coherent pattern, or could the additivity be coincidence given n=4 floor effects? Per-cell numbers: d10_vs_d0 +0.405, this file (d5_vs_d0) +0.249, d10_vs_d5 (P-C raw p=0.0571) +0.156. The additive arithmetic across the 3 contrasts (+0.249 + +0.156 ≈ +0.405) is consistent with several patterns including monotonic accumulation; Chumeng should weigh against the alternatives at synthesis time.
  - d5_vs_d0 is the flattest contrast in the family. Possible reasons (Chumeng to weigh): smaller frequency change between consecutive early timepoints, or n=4 floor noise. Useful constraint on what `per_timepoint_fisher` should show.
  - E site flat (min p_adj=1.0): does this reproduce in `between_timepoint_wilcoxon_d5_vs_d0_codon` at the codon level, or does codon resolution expose hits aa aggregation hides?
