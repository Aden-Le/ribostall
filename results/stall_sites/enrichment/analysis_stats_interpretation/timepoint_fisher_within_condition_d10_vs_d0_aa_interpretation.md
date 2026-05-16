---
input_csv: results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_aa.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), BH-FDR per (condition, site)
test_type_source: user-confirmed
n_tests: 120
n_significant_fdr05: 35
n_significant_fdr10: 39
min_p_adj: 9.89e-08
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "rare-aa-low-count", proposed_by: dylan, status: confirmed, why: "Rare residues (W, M, C) carry day_X_count < 100 in BWM and < 200 in control across some (condition, site) cells, e.g. BWM,P,W with day_10=36, day_0=38. Small absolute counts produce unstable odds ratios that should not anchor magnitude claims."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "Two-sided OR collapses direction; tied or near-zero counts can produce 0/inf-adjacent ORs. None observed at zero in this file, but the metric ranks symmetrically about 1.0 — log2(OR) is the safer axis for ranking."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Three site×aa cells show BWM and control going opposite ways or one being flat while the other moves: site-A N (BWM↑1.78 sig vs control 0.86 near-flat), site-E K (BWM↓0.79 sig vs control↑1.13 sig), site-P G (BWM 1.01 ns vs control↑1.28 sig). Divergences are the design-target of within-condition contrasts; shared-direction cells are reported alongside (see headline)."}
caveats_considered: []
headline: "Strong signal across both conditions in AA-level within-condition Fisher d10 vs d0 (35/120 hits at FDR<0.05; 12/60 BWM, 23/60 control). Largest-magnitude divergence cells: A:N (BWM↑1.78 vs control 0.86 near-flat), E:K (BWM↓0.79 vs control↑1.13, both sig opposite), P:G (BWM 1.01 ns vs control↑1.28 sig). Largest-magnitude shared-direction cells: P:E both ↑ (BWM 1.32 / control 1.16, both sig); P:K both ↓ (BWM 0.76 sig / control depleted per the broad site-P pattern)."
user_directives:
  - "(triage) test type → user confirmed Fisher's exact, BH-FDR per (condition, site)"
  - "(triage) CSV-specific caveats → user confirmed all three Dylan-proposed: rare-aa-low-count, OR-direction-asymmetry, control-vs-BWM-divergent-direction"
  - "(triage) framing firmness → Firm"
  - "(triage) layout → Both conditions in headline; Top hits split BWM vs control"
  - "(invocation context) `read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging` — confirms the Fisher test design and BH-FDR per (condition,site) family scoping reflected in the test_type field."
---

# Interpretation — timepoint_fisher_within_condition_d10_vs_d0_aa

> Source: `results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_aa.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (triage) "Confirm test type for d10_vs_d0_aa (and by extension all 6 files in this family)?" → "Fisher's exact, BH-FDR per (condition, site) (recommended)"
- (triage) "Any additional CSV-specific caveats for d10_vs_d0_aa beyond the locked family caveats?" → user confirmed all three Dylan-proposed: `rare-aa-low-count`, `OR-direction-asymmetry`, `control-vs-BWM-divergent-direction`
- (triage) "Framing firmness for this file?" → "Firm (recommended)"
- (triage) "How should headline / Top hits be split for this file?" → "Both conditions in headline; Top hits split BWM vs control (recommended)"
- (invocation context) "read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging" → Dylan read it; confirms the Fisher design (within-condition, per (condition, site) BH-FDR) recorded above.

## Headline
Strong signal at AA-level within-condition Fisher d10 vs d0: 35/120 hits at FDR<0.05 (12/60 BWM, 23/60 control). BWM is dominated by site-A direction shifts (N enriched OR=1.78; L, Y, A depleted) plus site-E K↓/Q↑/C↑ and site-P K↓/E↑. Control is dominated by broad site-P shifts (G, C, R, E enriched; M, W, Q, K, I, L depleted) and site-A G/R↑ vs Q/W/L↓. **Three largest-magnitude BWM-vs-control direction-divergent cells**: site-A N (BWM↑1.78 vs control flat 0.86, p_adj=0.053), site-E K (BWM↓0.79 vs control↑1.13, both FDR<0.01), and site-P G (BWM flat 1.01 vs control↑1.28, p_adj=5.7e-8). **Largest-magnitude shared-direction cells**: site-P E both enriched (BWM↑1.32 FDR<1e-3 / control↑1.16 FDR<0.01), and site-P K both depleted (BWM↓0.76 FDR<5e-5 / control,P,K within the broad site-P depletion signature M/W/Q/K/I/L↓).

## Top hits

### BWM (n_sig FDR<0.05 = 12 / 60)

| direction | feature | effect (`odds_ratio`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | E:C (Cys) | 1.788 | 0.0267 | rare-aa-low-count |
| ↑ | A:N (Asn) | 1.777 | 9.17e-07 |  |
| ↑ | E:Q (Gln) | 1.426 | 0.00145 |  |
| ↑ | P:E (Glu) | 1.322 | 2.44e-04 |  |
| ↑ | A:F (Phe) | 1.300 | 0.0264 |  |
| ↓ | A:L (Leu) | 0.733 | 1.36e-04 |  |
| ↓ | A:Y (Tyr) | 0.731 | 0.00258 |  |
| ↓ | P:K (Lys) | 0.757 | 5.51e-05 |  |
| ↓ | A:A (Ala) | 0.783 | 0.0178 |  |
| ↓ | E:K (Lys) | 0.790 | 4.52e-05 |  |

### control (n_sig FDR<0.05 = 23 / 60)

| direction | feature | effect (`odds_ratio`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | A:G (Gly) | 1.288 | 9.89e-08 |  |
| ↑ | P:G (Gly) | 1.285 | 5.67e-08 |  |
| ↑ | P:C (Cys) | 1.266 | 0.0493 |  |
| ↑ | A:R (Arg) | 1.261 | 2.13e-05 |  |
| ↑ | P:R (Arg) | 1.168 | 0.00668 |  |
| ↓ | P:M (Met) | 0.624 | 1.08e-05 | rare-aa-low-count |
| ↓ | P:W (Trp) | 0.616 | 0.0137 | rare-aa-low-count |
| ↓ | A:W (Trp) | 0.658 | 0.00216 | rare-aa-low-count |
| ↓ | A:Q (Gln) | 0.657 | 9.89e-08 |  |
| ↓ | E:W (Trp) | 0.682 | 0.0424 | rare-aa-low-count |

## Numbers at a glance
- `n_tests`: 120 (60 per condition)
- `n_significant` (adjusted-p < 0.05): 35 (BWM 12, control 23)
- `n_significant` (adjusted-p < 0.10): 39 (BWM 13, control 26)
- `min adjusted-p`: 9.89e-08 (tied: control,A,G and control,A,Q)
- `p_floor`: n/a — Fisher with pooled N in the thousands has no meaningful floor; the dominant statistical-design concern is `large-N-Fisher-anticonservative` (family-wide), not floor.
- Per (condition, site):
  - BWM,A: 6 sig at FDR<0.05 (N, L, Y, A, I, F); min p_adj = 9.17e-07
  - BWM,E: 4 sig (K, Q, S, C); min p_adj = 4.52e-05
  - BWM,P: 2 sig (K, E); min p_adj = 5.51e-05
  - control,A: 9 sig (G, Q, L, R, W, K, I, S, Y); min p_adj = 9.89e-08
  - control,E: 4 sig (Y, K, M, W); min p_adj = 0.00358
  - control,P: 10 sig (G, M, Q, K, E, I, L, R, W, D); min p_adj = 5.67e-08

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2×2 of (feature_count, total − feature_count) at each timepoint within one condition; user confirmed. The script (`stall_sites_non_consensus_stats.py`) computes BH-FDR within each (condition, site) family of 20 tests for AA-level (so each of the 6 sub-families is corrected independently, not against the full 120). The test does **not** answer whether the same feature is changing in BWM and control in the same direction — that comparison is the user's job (see "control-vs-BWM-divergent-direction" caveat) and is the largest-magnitude reading from this file's cells. Counts (`day_X_count`) and totals (`day_X_total`) are summed across replicates before the test, hence the family-level pseudoreplication caveat.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* — the 2 biological replicates per (condition, timepoint) are summed into the 2×2 before Fisher; p-values are anti-conservative. (Inherited from family — see `_INDEX.md`.)
- **large-N-Fisher-anticonservative** *(family-wide)* — pooled totals here are 6091, 6945, 8788, 27732. Even tiny relative deviations yield p_adj < 1e-7 (e.g. control,A,G with OR=1.29 still hits p_adj=9.89e-08). `odds_ratio` (log-axis) is the dominant effect column; p magnitude alone over-states confidence. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* — the 6 (condition, site) sub-families of 20 tests each are corrected independently; `p_adj` means "corrected within this sub-family", not across the whole 120-row file. (Inherited.)
- **within-condition-clean** *(family-wide)* — unlike between-condition or between-timepoint Wilcoxon, no condition or timepoint pooling; structurally cleaner. (Inherited.)
- **rare-aa-low-count** *(per-CSV)* — 6 of the 10 control top-hit rows (P:M, P:W, A:W, P:Q, E:W, E:M) have day_10_count < 200, and BWM,E:C has day_10_count=69, day_0_count=34. Rank by OR but flag magnitude as unstable for these rows.
- **OR-direction-asymmetry** *(per-CSV)* — no zero-cell rows in this file, so no infinite/zero ORs; the caveat is preserved as a ranking-axis discipline note (use log2(OR), not OR, when ordering effects).
- **control-vs-BWM-divergent-direction** *(per-CSV)* — flagged for Chumeng's reconciliation: BWM,A:N is +1.78 (FDR<1e-6) while control,A:N is 0.86 (p_adj=0.053, marginal in the opposite direction); BWM,E:K is 0.79 while control,E:K is 1.13 (both FDR<0.01, opposite directions); BWM,P:G is 1.01 (ns) while control,P:G is 1.28 (FDR<1e-7). Listing for Chumeng to reconcile.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs to reconcile this file against: `timepoint_fisher_within_condition_d10_vs_d0_codon` (codon-resolution refinement of the same contrast), and the d10_vs_d5 + d5_vs_d0 contrasts at the same aa resolution (for monotonicity / biphasic patterns).
- Open questions Chumeng should resolve at synthesis time:
  - **Are the three BWM-vs-control direction divergences (A:N, E:K, P:G) reproduced at the codon level?** If A:N divergence localizes to one of AAT/AAC, that is a codon-usage-shift signature; if both, it is amino-acid-level.
  - **Is the BWM site-A enrichment of N + depletion of L/Y/A monotonic across d0→d5→d10**, or does it appear only at d10? Compare to BWM rows in d10_vs_d5 and d5_vs_d0 aa files.
  - **Does the strong control site-P signature (G/C/R/E ↑, M/W/Q/K/I/L ↓) reflect housekeeping translational drift across the 10-day timecourse**, or a stress response in the unperturbed line? Cross-check `between_condition_wilcoxon_aa` (which collapses time and showed 0 hits — if this signal exists primarily within control across time, the pooled MW would have washed it out by averaging d0/d5/d10).
  - **Per-cell observations on P-site Cys/Glu in this file** (for Chumeng to weigh against the between_timepoint_wilcoxon family): BWM,P:E↑ (1.32, FDR<1e-3) and BWM,E:C↑ (1.79, FDR<0.05); control,P:C↑ (1.27, FDR<0.05) and control,P:E↑ (1.16, FDR<0.01). Does the P-site E↑ shift in both conditions reproduce in `per_timepoint_fisher_aa` and in the within-condition binomial baselines? Cys appears at site-P in control but at site-E in BWM — does this P↔E adjacent-site offset reappear in the codon companion or in `per_timepoint_fisher`?
