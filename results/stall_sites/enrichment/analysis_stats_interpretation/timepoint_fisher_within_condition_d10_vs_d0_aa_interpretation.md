---
input_csv: results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_aa.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), BH-FDR per (condition, site)
test_type_source: user-confirmed
n_tests: 120
n_significant_fdr05: 36
n_significant_fdr10: 40
min_p_adj: 9.89e-08
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "rare-aa-low-count", proposed_by: dylan, status: confirmed, why: "Rare residues (W, M, C) carry day_X_count < 100 in BWM and < 200 in control across some (condition, site) cells, e.g. BWM,E,C with day_10=69, day_0=34. Small absolute counts produce unstable `log2_OR` estimates that should not anchor magnitude claims."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "Two-sided OR collapses direction; tied or near-zero counts can produce 0/inf-adjacent ORs. None observed at zero in this file; `log2_OR` is now the reported effect column (already log-transformed), so positive/negative bands are symmetric by construction — rank directly on it."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Three site×aa cells show BWM and control going opposite ways or one being flat while the other moves: site-A N (BWM↑log2_OR=+0.83 sig vs control log2_OR=-0.22 near-flat), site-E K (BWM↓log2_OR=-0.34 sig vs control↑log2_OR=+0.18 sig), site-P G (BWM log2_OR=+0.01 ns vs control↑log2_OR=+0.36 sig). Divergences are the design-target of within-condition contrasts; shared-direction cells are reported alongside (see headline)."}
caveats_considered: []
headline: "Strong signal across both conditions in AA-level within-condition Fisher d10 vs d0 (36/120 hits at FDR<0.05; 12/60 BWM, 24/60 control). Largest-magnitude divergence cells: A:N (BWM↑log2_OR=+0.83 vs control log2_OR=-0.22 near-flat), E:K (BWM↓log2_OR=-0.34 vs control↑log2_OR=+0.18, both sig opposite), P:G (BWM log2_OR=+0.01 ns vs control↑log2_OR=+0.36 sig). Largest-magnitude shared-direction cells: P:E both ↑ (BWM log2_OR=+0.40 / control log2_OR=+0.22, both sig); P:K both ↓ (BWM↓log2_OR=-0.40 FDR<1e-4 / control↓log2_OR=-0.23 FDR<0.01)."
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
Strong signal at AA-level within-condition Fisher d10 vs d0: 36/120 hits at FDR<0.05 (12/60 BWM, 24/60 control). BWM is dominated by site-A direction shifts (N enriched `log2_OR`=+0.83; L, Y, A depleted) plus site-E K↓/Q↑/S↑/C↑ and site-P K↓/E↑. Control is dominated by broad site-P shifts (G, E, R, D, C enriched; M, Q, K, I, L, W depleted) and site-A G/R/K/I↑ vs Q/L/W/S/Y↓. **Three largest-magnitude BWM-vs-control direction-divergent cells**: site-A N (BWM enriched `log2_OR`=+0.83 vs control flat `log2_OR`=-0.22, p_adj=0.053), site-E K (BWM depleted `log2_OR`=-0.34 vs control enriched `log2_OR`=+0.18, both FDR<0.01), and site-P G (BWM flat `log2_OR`=+0.01 vs control enriched `log2_OR`=+0.36, p_adj=5.7e-8). **Largest-magnitude shared-direction cells**: site-P E both enriched (BWM `log2_OR`=+0.40 FDR<1e-3 / control `log2_OR`=+0.22 FDR<0.01), and site-P K both depleted (BWM `log2_OR`=-0.40 FDR<1e-4 / control `log2_OR`=-0.23 FDR<0.01).

## Top hits

Selection rule (family-level, updated 2026-05-22): all rows with `p_adj` < 0.05, no row cap; rows grouped by `site` in A -> P -> E order then sorted by `p_adj` ascending with `|log2_OR|` descending as tiebreaker.

### BWM (n_sig FDR<0.05 = 12 / 60)

| direction | feature | effect (`log2_OR`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | A:N (Asn) | 0.830 | 9.17e-07 |  |
| ↑ | A:I (Ile) | 0.303 | 0.0195 |  |
| ↑ | A:F (Phe) | 0.379 | 0.0264 |  |
| ↑ | P:E (Glu) | 0.403 | 2.44e-04 |  |
| ↑ | E:Q (Gln) | 0.512 | 0.00145 |  |
| ↑ | E:S (Ser) | 0.377 | 0.0256 |  |
| ↑ | E:C (Cys) | 0.838 | 0.0267 | rare-aa-low-count |
| ↓ | A:L (Leu) | -0.447 | 1.36e-04 |  |
| ↓ | A:Y (Tyr) | -0.451 | 0.00258 |  |
| ↓ | A:A (Ala) | -0.353 | 0.0178 |  |
| ↓ | P:K (Lys) | -0.402 | 5.51e-05 |  |
| ↓ | E:K (Lys) | -0.339 | 4.52e-05 |  |

### control (n_sig FDR<0.05 = 24 / 60)

| direction | feature | effect (`log2_OR`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | A:G (Gly) | 0.365 | 9.89e-08 |  |
| ↑ | A:R (Arg) | 0.335 | 2.13e-05 |  |
| ↑ | A:K (Lys) | 0.193 | 0.00581 |  |
| ↑ | A:I (Ile) | 0.199 | 0.0185 |  |
| ↑ | P:G (Gly) | 0.362 | 5.67e-08 |  |
| ↑ | P:E (Glu) | 0.218 | 0.00270 |  |
| ↑ | P:R (Arg) | 0.224 | 0.00668 |  |
| ↑ | P:D (Asp) | 0.159 | 0.0205 |  |
| ↑ | P:C (Cys) | 0.341 | 0.0493 | rare-aa-low-count |
| ↑ | E:K (Lys) | 0.176 | 0.00845 |  |
| ↓ | A:Q (Gln) | -0.606 | 9.89e-08 |  |
| ↓ | A:L (Leu) | -0.325 | 2.13e-05 |  |
| ↓ | A:W (Trp) | -0.604 | 0.00216 | rare-aa-low-count |
| ↓ | A:S (Ser) | -0.232 | 0.0185 |  |
| ↓ | A:Y (Tyr) | -0.209 | 0.0373 |  |
| ↓ | P:M (Met) | -0.681 | 1.08e-05 | rare-aa-low-count |
| ↓ | P:Q (Gln) | -0.438 | 5.74e-04 |  |
| ↓ | P:K (Lys) | -0.230 | 0.00206 |  |
| ↓ | P:I (Ile) | -0.245 | 0.00525 |  |
| ↓ | P:L (Leu) | -0.228 | 0.00668 |  |
| ↓ | P:W (Trp) | -0.699 | 0.0137 | rare-aa-low-count |
| ↓ | E:Y (Tyr) | -0.450 | 0.00358 | rare-aa-low-count |
| ↓ | E:M (Met) | -0.395 | 0.0128 | rare-aa-low-count |
| ↓ | E:W (Trp) | -0.552 | 0.0424 | rare-aa-low-count |

## Numbers at a glance
- `n_tests`: 120 (60 per condition)
- `n_significant` (adjusted-p < 0.05): 36 (BWM 12, control 24)
- `n_significant` (adjusted-p < 0.10): 40 (BWM 13, control 27)
- `min adjusted-p`: 9.89e-08 (tied: control,A,G and control,A,Q)
- `p_floor`: n/a — Fisher with pooled N in the thousands has no meaningful floor; the dominant statistical-design concern is `large-N-Fisher-anticonservative` (family-wide), not floor.
- Per (condition, site):
  - BWM,A: 6 sig at FDR<0.05 (N, L, Y, A, I, F); min p_adj = 9.17e-07
  - BWM,P: 2 sig (K, E); min p_adj = 5.51e-05
  - BWM,E: 4 sig (K, Q, S, C); min p_adj = 4.52e-05
  - control,A: 9 sig (G, Q, L, R, W, K, I, S, Y); min p_adj = 9.89e-08
  - control,P: 11 sig (G, M, Q, K, E, I, L, R, W, D, C); min p_adj = 5.67e-08
  - control,E: 4 sig (Y, K, M, W); min p_adj = 0.00358

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2×2 of (feature_count, total − feature_count) at each timepoint within one condition; user confirmed. The script (`stall_sites_non_consensus_stats.py`) computes BH-FDR within each (condition, site) family of 20 tests for AA-level (so each of the 6 sub-families is corrected independently, not against the full 120). The test does **not** answer whether the same feature is changing in BWM and control in the same direction — that comparison is the user's job (see "control-vs-BWM-divergent-direction" caveat) and is the largest-magnitude reading from this file's cells. Counts (`day_X_count`) and totals (`day_X_total`) are summed across replicates before the test, hence the family-level pseudoreplication caveat.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* — the 2 biological replicates per (condition, timepoint) are summed into the 2×2 before Fisher; p-values are anti-conservative. (Inherited from family — see `_INDEX.md`.)
- **large-N-Fisher-anticonservative** *(family-wide)* — pooled totals here are 6091, 6945, 8788, 27732. Even tiny relative deviations yield p_adj < 1e-7 (e.g. control,A,G with `log2_OR`=+0.365 still hits p_adj=9.89e-08). `log2_OR` is the dominant effect column; p magnitude alone over-states confidence. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* — the 6 (condition, site) sub-families of 20 tests each are corrected independently; `p_adj` means "corrected within this sub-family", not across the whole 120-row file. (Inherited.)
- **within-condition-clean** *(family-wide)* — unlike between-condition or between-timepoint Wilcoxon, no condition or timepoint pooling; structurally cleaner. (Inherited.)
- **rare-aa-low-count** *(per-CSV)* — 8 of the 36 Top-hits rows fall below the per-condition threshold (BWM < 100 / control < 200): BWM,E:C (day_10=69 / day_0=34); control,A:W (80 / 382); control,P:M (121 / 607); control,P:C (130 / 325); control,P:W (37 / 189); control,E:Y (175 / 749); control,E:M (160 / 660); control,E:W (57 / 263). Rank by `log2_OR` but flag magnitude as unstable for these rows. Note: control,P:Q (day_10=204) sits 4 above the strict < 200 control threshold and is therefore NOT flagged here — borderline case, override with `--rare-low-control-threshold 250` if you want to include it.
- **OR-direction-asymmetry** *(per-CSV)* — no zero-cell rows in this file, so no infinite/zero ORs; the caveat is preserved as a ranking-axis discipline note — `log2_OR` is the reported effect column (already log-transformed), so positive/negative bands are symmetric by construction; rank directly on it.
- **control-vs-BWM-divergent-direction** *(per-CSV)* — flagged for Chumeng's reconciliation: BWM,A:N is `log2_OR`=+0.83 (FDR<1e-6) while control,A:N is `log2_OR`=-0.22 (p_adj=0.053, marginal in the opposite direction); BWM,E:K is `log2_OR`=-0.34 while control,E:K is `log2_OR`=+0.18 (both FDR<0.01, opposite directions); BWM,P:G is `log2_OR`=+0.01 (ns) while control,P:G is `log2_OR`=+0.36 (FDR<1e-7). Listing for Chumeng to reconcile.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs to reconcile this file against: `timepoint_fisher_within_condition_d10_vs_d0_codon` (codon-resolution refinement of the same contrast), and the d10_vs_d5 + d5_vs_d0 contrasts at the same aa resolution (for monotonicity / biphasic patterns).
- Open questions Chumeng should resolve at synthesis time:
  - **Are the three BWM-vs-control direction divergences (A:N, E:K, P:G) reproduced at the codon level?** If A:N divergence localizes to one of AAT/AAC, that is a codon-usage-shift signature; if both, it is amino-acid-level.
  - **Is the BWM site-A enrichment of N + depletion of L/Y/A monotonic across d0→d5→d10**, or does it appear only at d10? Compare to BWM rows in d10_vs_d5 and d5_vs_d0 aa files.
  - **Does the strong control site-P signature (G/C/R/E ↑, M/W/Q/K/I/L ↓) reflect housekeeping translational drift across the 10-day timecourse**, or a stress response in the unperturbed line? Cross-check `between_condition_wilcoxon_aa` (which collapses time and showed 0 hits — if this signal exists primarily within control across time, the pooled MW would have washed it out by averaging d0/d5/d10).
  - **Per-cell observations on P-site Cys/Glu in this file** (for Chumeng to weigh against the between_timepoint_wilcoxon family): BWM,P:E↑ (`log2_OR`=+0.40, FDR<1e-3) and BWM,E:C↑ (`log2_OR`=+0.84, FDR<0.05); control,P:C↑ (`log2_OR`=+0.34, FDR<0.05) and control,P:E↑ (`log2_OR`=+0.22, FDR<0.01). Does the P-site E↑ shift in both conditions reproduce in `per_timepoint_fisher_aa` and in the within-condition binomial baselines? Cys appears at site-P in control but at site-E in BWM — does this P↔E adjacent-site offset reappear in the codon companion or in `per_timepoint_fisher`?
