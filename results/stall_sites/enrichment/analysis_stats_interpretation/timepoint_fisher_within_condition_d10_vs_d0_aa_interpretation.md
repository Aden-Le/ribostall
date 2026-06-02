---
input_csv: results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_aa.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), BH-FDR per (condition, site)
test_type_source: user-confirmed
n_tests: 120
n_significant_fdr05: 36
n_significant_fdr10: 40
min_p_adj: 5.67e-08
p_floor: null
pseudoreplicated: true
synced_from_olive_qmd: 2026-06-01
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
  - "(triage) layout → Both conditions in headline; Top hits split into three significance sections (both / BWM-only / control-only), each pairing BWM and control `log2_OR` per (site, amino acid) cell with an `Effect change (BWM − control)` column (revised 2026-05-28; originally a BWM-vs-control two-table split)"
  - "(invocation context) `read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging` — confirms the Fisher test design and BH-FDR per (condition,site) family scoping reflected in the test_type field."
  - "(readback) Reconciled shared content from the corrected .qmd on 2026-06-01 -> number-audit only (table structure already mirrored the .qmd's three-section layout, no table change). Corrections: front-matter min_p_adj 9.89e-08 -> 5.67e-08 (true file minimum at control,P,G; the prior value was the within-(A)-family tied p_adj) and the Numbers-at-a-glance min adjusted-p line gained the .qmd's '5.67e-08 (control,P,G, the file minimum)' clause. Cosmetic Dylan back-sync: Top-hits column header 'Effect change (BWM - control)' -> 'Effect change' across all three sections and the intro, matching the family standard (flags were already on the low-count (BWM, C)/(C) form per the 2026-05-29 sync)."
---

# Interpretation — timepoint_fisher_within_condition_d10_vs_d0_aa

> Source: `results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_aa.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (triage) "Confirm test type for d10_vs_d0_aa (and by extension all 6 files in this family)?" → "Fisher's exact, BH-FDR per (condition, site) (recommended)"
- (triage) "Any additional CSV-specific caveats for d10_vs_d0_aa beyond the locked family caveats?" → user confirmed all three Dylan-proposed: `rare-aa-low-count`, `OR-direction-asymmetry`, `control-vs-BWM-divergent-direction`
- (triage) "Framing firmness for this file?" → "Firm (recommended)"
- (triage) "How should headline / Top hits be split for this file?" → "Both conditions in headline; Top hits split into three significance sections — both / BWM-only / control-only — pairing BWM and control `log2_OR` per (site, amino acid) cell with an `Effect change (BWM − control)` column" (layout revised 2026-05-28; originally a BWM-vs-control two-table split)
- (invocation context) "read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging" → Dylan read it; confirms the Fisher design (within-condition, per (condition, site) BH-FDR) recorded above.
- (2026-05-29 flag-token sync) Top-hits `Flag` column migrated from `rare-aa-low-count (BWM, control)` / `rare-aa-low-count (control)` to the family-standard table glyph `low-count (BWM, C)` / `low-count (C)`, matching the rendered Olive `.qmd` and the sister `.md` files (e.g. d5_vs_d0_aa). Glyph-only change: the 8 flagged rows and all numbers are unchanged; the conceptual `rare-aa-low-count` confirmed-caveat label is kept (distinct from the table glyph).
- (2026-06-01 Stage-7 readback) Reconciled from the corrected Olive `.qmd` (number-audit only — the three-section table already mirrored the `.qmd`, no table reshape). Every number enumerated and verified against the `.qmd` (shared content) or the raw CSV (Dylan-only column values). One numeric correction: `min_p_adj` 9.89e-08 -> 5.67e-08 (the front-matter and the Numbers-at-a-glance line had carried the within-(A)-family tied p_adj at control,A,G/Q as the file minimum, but control,P,G p_adj=5.67e-08 is smaller; the `.qmd` already stated the file minimum). Cosmetic back-sync (the open `### Needs to be corrected` item): the Top-hits column header `Effect change (BWM − control)` was renamed to the family-standard `Effect change` across all three sections and the intro. All 30 Top-hits cells re-derived from the CSV `odds_ratio` column (log2-transformed) and confirmed unchanged; section placement re-verified against `p_adj < 0.05`.

## Headline
Strong signal at AA-level within-condition Fisher d10 vs d0: 36/120 hits at FDR<0.05 (12/60 BWM, 24/60 control). BWM is dominated by site-A direction shifts (N enriched `log2_OR`=+0.83; L, Y, A depleted) plus site-E K↓/Q↑/S↑/C↑ and site-P K↓/E↑. Control is dominated by broad site-P shifts (G, E, R, D, C enriched; M, Q, K, I, L, W depleted) and site-A G/R/K/I↑ vs Q/L/W/S/Y↓. **Three largest-magnitude BWM-vs-control direction-divergent cells**: site-A N (BWM enriched `log2_OR`=+0.83 vs control flat `log2_OR`=-0.22, p_adj=0.053), site-E K (BWM depleted `log2_OR`=-0.34 vs control enriched `log2_OR`=+0.18, both FDR<0.01), and site-P G (BWM flat `log2_OR`=+0.01 vs control enriched `log2_OR`=+0.36, p_adj=5.7e-8). **Largest-magnitude shared-direction cells**: site-P E both enriched (BWM `log2_OR`=+0.40 FDR<1e-3 / control `log2_OR`=+0.22 FDR<0.01), and site-P K both depleted (BWM `log2_OR`=-0.40 FDR<1e-4 / control `log2_OR`=-0.23 FDR<0.01).

## Top hits

Selection rule (updated 2026-05-28): every (site, amino acid) cell significant at FDR<0.05 in at least one condition, split into three sections by which condition(s) reach significance — both, BWM only, control only. Each row pairs the BWM and control `log2_OR` (the within-condition day_10-vs-day_0 Fisher effect: positive = enriched at day_10 relative to day_0, negative = depleted) and reports `Effect change` = BWM `log2_OR` − control `log2_OR`, a single divergence measure (large magnitude = the two conditions' day_10-vs-day_0 trajectories diverge). Rows are grouped by `site` in A → P → E order, then sorted by `Effect change` descending (most BWM-positive divergence first, most control-positive last). Cells significant in neither condition are omitted. Generated by `scripts/within_condition_sig_split.py`.

### Significant in both conditions (n = 6 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | I (Ile) | +0.303 | +0.199 | +0.104 |  |
| A | L (Leu) | -0.447 | -0.325 | -0.122 |  |
| A | Y (Tyr) | -0.451 | -0.209 | -0.242 |  |
| P | E (Glu) | +0.403 | +0.218 | +0.185 |  |
| P | K (Lys) | -0.402 | -0.230 | -0.172 |  |
| E | K (Lys) | -0.339 | +0.176 | -0.515 |  |

### Significant in BWM only (n = 6 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | N (Asn) | +0.830 | -0.222 | +1.051 |  |
| A | F (Phe) | +0.379 | +0.064 | +0.315 |  |
| A | A (Ala) | -0.353 | -0.020 | -0.333 |  |
| E | C (Cys) | +0.838 | +0.011 | +0.827 | low-count (BWM, C) |
| E | Q (Gln) | +0.512 | -0.075 | +0.587 |  |
| E | S (Ser) | +0.377 | +0.019 | +0.358 |  |

### Significant in control only (n = 18 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | Q (Gln) | -0.069 | -0.606 | +0.537 |  |
| A | W (Trp) | -0.385 | -0.604 | +0.220 | low-count (BWM, C) |
| A | S (Ser) | -0.030 | -0.232 | +0.202 |  |
| A | R (Arg) | +0.099 | +0.335 | -0.236 |  |
| A | K (Lys) | -0.119 | +0.193 | -0.312 |  |
| A | G (Gly) | +0.025 | +0.365 | -0.339 |  |
| P | Q (Gln) | +0.195 | -0.438 | +0.633 |  |
| P | W (Trp) | -0.269 | -0.699 | +0.430 | low-count (BWM, C) |
| P | M (Met) | -0.328 | -0.681 | +0.352 | low-count (C) |
| P | I (Ile) | -0.069 | -0.245 | +0.177 |  |
| P | L (Leu) | -0.063 | -0.228 | +0.165 |  |
| P | C (Cys) | +0.431 | +0.341 | +0.090 | low-count (BWM, C) |
| P | D (Asp) | +0.112 | +0.159 | -0.047 |  |
| P | G (Gly) | +0.011 | +0.362 | -0.350 |  |
| P | R (Arg) | -0.190 | +0.224 | -0.415 |  |
| E | W (Trp) | +0.278 | -0.552 | +0.830 | low-count (BWM, C) |
| E | Y (Tyr) | +0.136 | -0.450 | +0.586 | low-count (C) |
| E | M (Met) | -0.103 | -0.395 | +0.291 | low-count (C) |

## Numbers at a glance
- `n_tests`: 120 (60 per condition)
- `n_significant` (adjusted-p < 0.05): 36 (BWM 12, control 24)
- `n_significant` (adjusted-p < 0.10): 40 (BWM 13, control 27)
- `min adjusted-p`: 9.89e-08 (tied: control,A,G and control,A,Q within their family); 5.67e-08 (control,P,G, the file minimum)
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
- **rare-aa-low-count** *(per-CSV)* — 8 of the significant Top-hits cells carry a rare-aa-low-count flag because at least one arm falls below its per-condition threshold (BWM < 100 / control < 200): BWM,E:C (day_10=69 / day_0=34); control,A:W (80 / 382); control,P:M (121 / 607); control,P:C (130 / 325); control,P:W (37 / 189); control,E:Y (175 / 749); control,E:M (160 / 660); control,E:W (57 / 263). In the Top-hits tables the `Flag` column names which arm(s) tripped it (e.g. `low-count (BWM, C)` when both arms are low). Rank by `log2_OR` but treat magnitude as unstable for these cells. Note: control,P:Q (day_10=204) sits 4 above the strict < 200 control threshold and is therefore NOT flagged here — borderline case, override with `--rare-control-threshold 250` if you want to include it.
- **OR-direction-asymmetry** *(per-CSV)* — no zero-cell rows in this file, so no infinite/zero ORs; the caveat is preserved as a ranking-axis discipline note — `log2_OR` is the reported effect column (already log-transformed), so positive/negative bands are symmetric by construction; rank directly on it.
- **control-vs-BWM-divergent-direction** *(per-CSV)* — flagged for Chumeng's reconciliation: BWM,A:N is `log2_OR`=+0.83 (FDR<1e-6) while control,A:N is `log2_OR`=-0.22 (p_adj=0.053, marginal in the opposite direction); BWM,E:K is `log2_OR`=-0.34 while control,E:K is `log2_OR`=+0.18 (both FDR<0.01, opposite directions); BWM,P:G is `log2_OR`=+0.01 (ns) while control,P:G is `log2_OR`=+0.36 (FDR<1e-7). Listing for Chumeng to reconcile.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs to reconcile this file against: `timepoint_fisher_within_condition_d10_vs_d0_codon` (codon-resolution refinement of the same contrast), and the d10_vs_d5 + d5_vs_d0 contrasts at the same aa resolution (for monotonicity / biphasic patterns).
- Open questions Chumeng should resolve at synthesis time:
  - **Are the three BWM-vs-control direction divergences (A:N, E:K, P:G) reproduced at the codon level?** If A:N divergence localizes to one of AAT/AAC, that is a codon-usage-shift signature; if both, it is amino-acid-level.
  - **Is the BWM site-A enrichment of N + depletion of L/Y/A monotonic across d0→d5→d10**, or does it appear only at d10? Compare to BWM rows in d10_vs_d5 and d5_vs_d0 aa files.
  - **Does the strong control site-P signature (G/C/R/E ↑, M/W/Q/K/I/L ↓) reflect housekeeping translational drift across the 10-day timecourse**, or a stress response in the unperturbed line? Cross-check `between_condition_wilcoxon_aa` (which collapses time and showed 0 hits — if this signal exists primarily within control across time, the pooled MW would have washed it out by averaging d0/d5/d10).
  - **Per-cell observations on P-site Cys/Glu in this file** (for Chumeng to weigh against the between_timepoint_wilcoxon family): BWM,P:E↑ (`log2_OR`=+0.40, FDR<1e-3) and BWM,E:C↑ (`log2_OR`=+0.84, FDR<0.05); control,P:C↑ (`log2_OR`=+0.34, FDR<0.05) and control,P:E↑ (`log2_OR`=+0.22, FDR<0.01). Does the P-site E↑ shift in both conditions reproduce in `per_timepoint_fisher_aa` and in the within-condition binomial baselines? Cys appears at site-P in control but at site-E in BWM — does this P↔E adjacent-site offset reappear in the codon companion or in `per_timepoint_fisher`?
