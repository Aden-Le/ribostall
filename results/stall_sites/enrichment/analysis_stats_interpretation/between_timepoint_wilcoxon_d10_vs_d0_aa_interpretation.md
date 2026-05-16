---
input_csv: results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d0_aa.csv
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
  - {label: "weighted_log2_enrichment-absent", proposed_by: dylan, status: confirmed, why: "Effect column is `log2_FC` of medians only; no count-weighted enrichment column. Rare AAs (W, C ~0.5-1.5% median freq) ranked equivalently to common ones (K, E ~10%)."}
  - {label: "small-bh-family-discreteness", proposed_by: dylan, status: confirmed, why: "BH adjusts only 20 raw-p per site. With raw-p clamped at 0.0286, only one floor hit per site can reach p_adj = 20*0.0286 / 1 = 0.571; here exactly one P-site row hits the floor, producing the empirically observed p_adj = 0.5714."}
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: ruled_out, why: "Empirical audit (scripts/_for_claude_mw_branch_audit.py --design between_timepoint --timepoints day10 day0 --level aa): 0/60 (site, AA) tests have rank ties in the pooled 8-element sample; scipy.stats.mannwhitneyu(method='auto') picks the exact branch for all 60; recomputed p matches pipeline CSV to ~1e-16; forcing method='asymptotic' flips 0 raw-p<0.05 decisions and leaves the per-site BH conclusion unchanged (0 hits at FDR<0.05 either way)."}
headline: "No statistically significant differences at FDR<0.05 (0/60) for AA-level d10-vs-d0 MW with BWM and control reps pooled per timepoint; the file's only sub-0.05 raw-p row is P-site C (+0.405, raw p=0.0286, p_adj=0.571, flagged `nominal-only, floor`), with three more cells tied at the near-floor raw p=0.0571 (P-site E +0.373, A-site Q -0.406, A-site L -0.368, all p_adj=0.571). Min raw p across the file = 0.0286 (the n=4 vs n=4 floor), so 'no FDR hits' here is structural, not biological."
user_directives:
  - "(triage, batched across all 6 family members) Test type confirmation — `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → `Confirm`."
  - "(triage, batched across the 3 aa files) CSV-specific caveats → `weighted_log2_enrichment-absent`, `small-bh-family-discreteness` confirmed."
  - "(invocation context) `Read @shell_scripts/run_enrichment_stats.sh to help triage csv specific as extra context` — confirms the upstream pipeline calls `stall_sites_non_consensus_stats.py` with the EXP_GROUPS string defining the timepoint pooling."
---

# Interpretation — between_timepoint_wilcoxon_d10_vs_d0_aa

> Source: `results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d0_aa.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage, batched across all 6 family members) Test type: `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → "Confirm".
- (triage, batched across 3 aa files) CSV-specific caveats: `weighted_log2_enrichment-absent`, `small-bh-family-discreteness` confirmed.
- (invocation context) Shell script `run_enrichment_stats.sh` confirms upstream pipeline and the EXP_GROUPS string driving the timepoint pooling.

## Headline
No statistically significant differences at FDR<0.05 (0/60) for AA-level day_10 vs day_0 MW with BWM and control reps pooled within each timepoint (n=4 per side). Min raw p = 0.0286 = the exact MW two-sided floor for n=4 vs n=4 — so "no FDR hits" is structural per the locked `mw-floor-blocking` caveat, not a biological negative. One row at the floor and also the file's only sub-0.05 raw-p row (P-site C, +0.405, p_adj=0.571, flagged `nominal-only, floor`), plus three near-floor cells tied at raw p = 0.0571 (P-site E +0.373, A-site Q -0.406, A-site L -0.368, all p_adj=0.571). Sites P and A both reach the same per-site BH wall (min p_adj = 0.571); E-site is flat (min p_adj = 0.984). Treat as exploratory leads only; the only test in this family that can resolve a real day_10 vs day_0 effect is the per-timepoint Fisher.

## Top hits

### P site (headline group)

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | C | +0.405 | 0.571 | nominal-only, floor |
| enriched | E | +0.373 | 0.571 |  |
| enriched | G | +0.124 | 0.571 |  |
| enriched | D | +0.144 | 0.932 |  |
| enriched | V | +0.110 | 0.932 |  |
| depleted | K | -0.271 | 0.571 |  |
| depleted | W | -0.620 | 0.667 |  |
| depleted | I | -0.159 | 0.667 |  |
| depleted | M | -0.511 | 0.857 |  |
| depleted | H | -0.247 | 0.857 |  |

<details>
<summary>A site</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | G | +0.234 | 0.800 |  |
| enriched | I | +0.233 | 0.800 |  |
| enriched | R | +0.235 | 0.810 |  |
| enriched | D | +0.171 | 0.810 |  |
| enriched | F | +0.109 | 0.810 |  |
| depleted | Q | -0.406 | 0.571 |  |
| depleted | L | -0.368 | 0.571 |  |
| depleted | Y | -0.401 | 0.800 |  |
| depleted | W | -0.496 | 0.810 |  |
| depleted | S | -0.165 | 0.810 |  |

</details>

<details>
<summary>E site</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | C | +0.241 | 0.984 |  |
| enriched | S | +0.179 | 0.984 |  |
| enriched | L | +0.116 | 0.984 |  |
| enriched | T | +0.100 | 0.984 |  |
| enriched | Q | +0.369 | 0.984 |  |
| depleted | M | -0.384 | 0.984 |  |
| depleted | P | -0.253 | 0.984 |  |
| depleted | K | -0.128 | 0.984 |  |
| depleted | I | -0.047 | 0.984 |  |
| depleted | D | -0.108 | 0.984 |  |

</details>

## Numbers at a glance
- `n_tests`: 60
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.5714 (P-site C floor; ties with P-K, A-L, A-Q at the same p_adj wall)
- `min raw-p`: 0.02857 (= MW exact floor for n=4 vs n=4); 1 row at the floor (P-site C, +0.405)
- `p_floor`: 0.02857 (n=4 vs n=4 two-sided exact)
- Per-site BH families:
  - A site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.571
  - E site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.984
  - P site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.571

## Methods
Dylan proposed Mann-Whitney U / Wilcoxon rank-sum two-sided on per-replicate frequencies, n=4 day_10 (BWM_day10_rep2/3 + control_day10_rep2/3) vs n=4 day_0 (BWM_day0_rep2/3 + control_day0_rep2/3), BH-FDR per site (each E/P/A site = 20-AA family); user confirmed. Effect column is `log2_FC` of medians; test statistic is `U_stat`. The test answers "do day_10 and day_0 reps differ in per-rep AA frequency at this site?" — pooling BWM and control reps within timepoint makes condition-by-time interactions invisible. Does *not* answer "does BWM behave differently across time" (that is `timepoint_fisher_within_condition_d10_vs_d0`).

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — MW two-sided exact p-floor for n=4 vs n=4 = 2/C(8,4) = 0.02857. Per-site BH on 20 AA tests turns one floor hit into p_adj = 0.571 (observed exactly here at P-site C); two floor hits at the same site → p_adj = 0.286; six floor hits → p_adj = 0.095. FDR<0.05 mathematically requires ~every test in a site's family to tie at the floor, which is impossible.
- **condition-pooled-confound** (family-wide) — n=4 per timepoint = 2 BWM + 2 control. A BWM-specific time response and a control-specific time response that point in opposite directions cancel inside this MW; sister Fisher (`timepoint_fisher_within_condition_d10_vs_d0`) is the test that holds condition fixed.
- **n=4-low-power** (family-wide) — null is weakly informative.
- **p-floor-aware-headline** (family-wide) — headline above explicitly flags the floor as the reason for "no FDR hits".
- **weighted_log2_enrichment-absent** (per-CSV) — `log2_FC` of medians only; no weighted variant. P-site W (-0.620) is the largest |effect| in the file but median freq < 1%; treat top |effect| ranks with care.
- **small-bh-family-discreteness** (per-CSV) — the observed min p_adj of 0.571 is the BH wall for one floor hit in a 20-test family (20 * 0.0286 ≈ 0.571); confirms the structural ceiling empirically.

### Considered but not applicable
- **asymptotic-with-ties** — Concern: at smaller per-arm n (4 vs 4 here), scipy's `mannwhitneyu` could fall back to the asymptotic Z approximation under tied per-rep frequencies, which would shift p-values away from the exact-floor bound. **Ruled out empirically** via `scripts/_for_claude_mw_branch_audit.py --design between_timepoint --timepoints day10 day0 --level aa`: 0/60 (site, AA) tests have any rank ties in the pooled 8-element sample; `mannwhitneyu(method='auto')` picked the exact branch for all 60; recomputed p matches pipeline CSV to ~1e-16; forcing `method='asymptotic'` flips 0 raw-p<0.05 decisions and leaves per-site BH conclusion unchanged (0 hits at FDR<0.05 either way).
- *(Dylan did not propose further per-CSV caveats here; the locked family caveats already cover the structural failure mode and the column-shape note + BH-discreteness note are sufficient.)*

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs in this family that should be reconciled: `between_timepoint_wilcoxon_d10_vs_d0_codon.csv` (codon resolution, same contrast), `between_timepoint_wilcoxon_d10_vs_d5_aa/codon.csv`, `between_timepoint_wilcoxon_d5_vs_d0_aa/codon.csv`.
- Open questions Chumeng should resolve at synthesis time:
  - The single floor row P-site C (+0.405) and the near-floor row P-site E (+0.373): does either reappear in `per_timepoint_fisher_aa.csv` at day_10 (BWM-vs-control held fixed) with a consistent direction? If yes → real day-10 specific signal that MW couldn't formalize because of the n=4 floor.
  - P site is the only site with closest-to-significant rows (min p_adj = 0.571 vs 0.984 at E). Does P-site signal also dominate `timepoint_fisher_within_condition_d10_vs_d0_aa.csv` (which holds condition fixed)?
  - Codon agreement: do the codons synonymous with P-site C (TGT, TGC), P-site E (GAA, GAG), P-site K (AAA, AAG) at the same site in `between_timepoint_wilcoxon_d10_vs_d0_codon.csv` show the same direction?
