---
input_csv: results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d5_aa.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U / Wilcoxon rank-sum (two-sided)
test_type_source: user-confirmed
n_tests: 60
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.38095238095238093
p_floor: 0.02857142857142857
pseudoreplicated: null
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "weighted_log2_enrichment-absent", proposed_by: dylan, status: confirmed, why: "Effect column is `log2_FC` of medians only; no count-weighted variant."}
  - {label: "small-bh-family-discreteness", proposed_by: dylan, status: confirmed, why: "BH adjusts only 20 raw-p per site. Three rows at site A clear raw p ≤ 0.0571 — under BH this gives p_adj = max(20*0.0286/1, 20*0.0571/2, 20*0.0571/3) = max(0.571, 0.571, 0.381) → 0.381, exactly the observed min p_adj. Two near-floor + one floor at the same site is the only configuration in this family that breaks below the 0.571 wall."}
headline: "Firm null at FDR<0.05 (0/60) for AA-level d10-vs-d5 MW with reps pooled across BWM and control per timepoint; min p_adj = 0.381 at site A — the lowest in the entire family — driven by 3 rows ≤ raw p 0.057: G (+0.293, floor), R (+0.239, near-floor), Q (-0.462, near-floor). Site P also has one floor row (G +0.317). Min raw p = 0.0286 (n=4 vs n=4 floor); 'no FDR hits' is structural per the locked floor caveat."
user_directives:
  - "(triage, batched across all 6 family members) Test type confirmation — `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → `Confirm`."
  - "(triage, batched across the 3 aa files) CSV-specific caveats → `weighted_log2_enrichment-absent`, `small-bh-family-discreteness` confirmed."
---

# Interpretation — between_timepoint_wilcoxon_d10_vs_d5_aa

> Source: `results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d5_aa.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage, batched) Test type: `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → "Confirm".
- (triage, batched 3 aa files) CSV-specific caveats: `weighted_log2_enrichment-absent`, `small-bh-family-discreteness` confirmed.

## Headline
Firm null at FDR<0.05 (0/60) for AA-level day_10 vs day_5 MW with BWM and control reps pooled within each timepoint (n=4 per side). Min p_adj = 0.381 at site A — the lowest p_adj across the entire `between_timepoint_wilcoxon` family — driven by 3 site-A rows ≤ raw p 0.057: G enriched (+0.293, raw p=0.0286 floor), R enriched (+0.239, raw p=0.0571), Q depleted (-0.462, raw p=0.0571). One additional floor row at site P (G +0.317, raw p=0.0286, p_adj=0.571). Sites E (min p_adj=0.97) is flat. Closest-to-significant signature: A-site G enrichment + Q depletion in day_10 vs day_5 — i.e. day_10 reps trend toward more A-site Glycine and less A-site Glutamine than day_5 reps. Treat as exploratory; the floor caveat blocks formal FDR<0.05 here regardless of biology.

## Top hits

### A site (headline group — min p_adj = 0.381, lowest in the family)

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | G | +0.293 | 0.381 | floor |
| enriched | R | +0.239 | 0.381 | nominal-only |
| enriched | W | +0.175 | 0.457 |  |
| enriched | C | +0.045 | 0.857 |  |
| enriched | Y | +0.011 | 0.980 |  |
| depleted | Q | -0.462 | 0.381 | nominal-only |
| depleted | S | -0.122 | 0.457 |  |
| depleted | L | -0.271 | 0.857 |  |
| depleted | T | -0.063 | 0.857 |  |
| depleted | D | -0.051 | 0.980 |  |

<details>
<summary>E site (no floor rows; flat)</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | P | +0.258 | 0.971 |  |
| enriched | G | +0.094 | 0.971 |  |
| enriched | A | +0.069 | 0.971 |  |
| enriched | T | +0.055 | 0.971 |  |
| enriched | C | +0.109 | 0.984 |  |
| depleted | M | -0.214 | 0.971 |  |
| depleted | E | -0.182 | 0.971 |  |
| depleted | Y | -0.088 | 0.971 |  |
| depleted | S | -0.122 | 0.971 |  |
| depleted | Q | -0.113 | 0.971 |  |

</details>

<details>
<summary>P site (1 floor row)</summary>

| direction | feature | effect (`log2_FC`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | G | +0.317 | 0.571 | floor |
| enriched | C | +0.156 | 0.571 | nominal-only |
| enriched | E | +0.105 | 0.762 |  |
| enriched | D | +0.112 | 0.810 |  |
| enriched | R | +0.085 | 0.810 |  |
| depleted | I | -0.203 | 0.571 |  |
| depleted | S | -0.136 | 0.571 |  |
| depleted | Q | -0.378 | 0.762 |  |
| depleted | W | -0.369 | 0.762 |  |
| depleted | F | -0.079 | 0.762 |  |

</details>

## Numbers at a glance
- `n_tests`: 60
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.381 (site A: G enriched, R enriched, Q depleted, all tied at the BH wall)
- `min raw-p`: 0.02857 (= MW exact floor); 2 rows at the floor — A-G (+0.293), P-G (+0.317)
- `p_floor`: 0.02857 (n=4 vs n=4 two-sided exact)
- Per-site BH families:
  - A site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.381
  - E site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.971
  - P site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.571

## Methods
Dylan proposed Mann-Whitney U / Wilcoxon rank-sum two-sided on per-replicate frequencies, n=4 day_10 vs n=4 day_5 (BWM_day10/5_rep2/3 + control_day10/5_rep2/3), BH-FDR per site (each site = 20-AA family); user confirmed. Effect column is `log2_FC` of medians; test statistic is `U_stat`. Test answers "do day_10 and day_5 reps differ in per-rep AA frequency at this site?" — pooling BWM and control within timepoint. Does *not* answer condition-specific time response (`timepoint_fisher_within_condition_d10_vs_d5_aa`).

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — exact floor 0.0286; per-site BH wall for one floor hit on a 20-AA family is 0.571.
- **condition-pooled-confound** (family-wide) — n=4 per timepoint = 2 BWM + 2 control. The G enrichment at A and P at day_10 (vs day_5) cannot be attributed to BWM or control here.
- **n=4-low-power** (family-wide) — null is weakly informative.
- **p-floor-aware-headline** (family-wide).
- **weighted_log2_enrichment-absent** (per-CSV).
- **small-bh-family-discreteness** (per-CSV) — the observed p_adj=0.381 is a discrete-BH artefact: 3 site-A rows at raw p ≤ 0.0571 produce BH wall 20*0.0571/3 = 0.381. This is the *only* file in the `between_timepoint_wilcoxon` family that breaks below the 0.571 ceiling at any resolution.

### Considered but not applicable
*(Dylan did not propose any further per-CSV caveats here.)*

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs in this family that should be reconciled: `between_timepoint_wilcoxon_d10_vs_d5_codon.csv` (codon resolution, same contrast), and the d10_vs_d0 / d5_vs_d0 pairs at both resolutions.
- Open questions Chumeng should resolve at synthesis time:
  - The site-A G+/Q-/R+ closest-to-significant block at d10-vs-d5: do the same three AAs come up with consistent direction in `per_timepoint_fisher_aa.csv` at day_10 vs day_5 (BWM-vs-control held fixed)? If yes → real day-10 vs day-5 BWM/control divergence the MW couldn't formalize.
  - Site P G enrichment at the floor here AND at d10_vs_d0_aa (A-G also at floor at d10_vs_d0): is G accumulating at site A and P from day_5 → day_10 monotonically, or is it a d5-specific dip?
  - Why is d10_vs_d5 the strongest contrast in this family (min p_adj 0.381) while d10_vs_d0 (the longest interval) is weaker (0.571)? If a real perturbation effect peaked at day_5 then partially reversed by day_10, day_10 vs day_5 would have larger transient signal than day_10 vs day_0. Cross-check `timepoint_fisher_within_condition_d10_vs_d5` vs `_d10_vs_d0` for biphasic patterns.
