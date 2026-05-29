---
input_csv: results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_aa.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), BH-FDR per (condition, site)
test_type_source: user-confirmed
n_tests: 120
n_significant_fdr05: 36
n_significant_fdr10: 47
min_p_adj: 2.27e-12
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md). control,E,P at p_adj=2.27e-12 on log2_OR=-0.63 (~35% relative depletion) is the family-wide minimum and the most extreme single-feature large-N anti-conservative case."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "rare-aa-low-count", proposed_by: dylan, status: confirmed, why: "Trp/Cys/Met sit below count=200 in BWM (W: 108 vs 89; C: 82 vs 34; M: 251 vs 122). BWM,A,W enters Top hits with day_0_count=89 — flagged. control,A,W (93 vs 382) and control,P,M (186 vs 607) also enter Top hits. Direction reliable; magnitude unstable for these rows."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "No zero-cell rows; caveat preserved as ranking-axis discipline (use log2_OR, not OR, when ordering effects)."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Three large-magnitude direction flips in this CSV. **E:K**: BWM log2_OR=-0.42 (FDR<2e-9) vs control log2_OR=+0.26 (FDR<8e-7); same direction-flip seen in d10_vs_d0_aa — Chumeng to weigh whether this reflects a d0-anchored divergence vs other patterns. **A:K** flip (BWM log2_OR=-0.17 ns vs control log2_OR=+0.33 FDR<1e-7). **E:A** flip (BWM log2_OR=+0.02 ns vs control log2_OR=-0.36 sig). Largest-magnitude shared-direction cells: A:N (both enriched, BWM log2_OR=+0.83 / control log2_OR=+0.33, both sig) and A:W (both depleted, BWM log2_OR=-0.70 / control log2_OR=-0.74, both sig — low-count flag)."}
  - {label: "control-E-P-extreme-anticonservative", proposed_by: dylan, status: confirmed, why: "control,E:P at p_adj=2.27e-12 is the smallest p_adj in the entire family for log2_OR=-0.63 — only ~35% relative depletion. The most extreme single-feature large-N anti-conservative case in the family. Per-CSV flag in addition to the family-wide large-N caveat to cap rhetoric on this row: rank by log2_OR and interpret this row's p magnitude as direction-only; effect size carries the biological scale."}
caveats_considered: []
headline: "d5 vs d0 within-condition Fisher (aa-level): 36/120 hits at FDR<0.05 (15/60 BWM, 21/60 control). Largest-magnitude divergence cells: E:K (BWM log2_OR=-0.42 FDR<2e-9 vs control log2_OR=+0.26 FDR<8e-7), A:K (BWM log2_OR=-0.17 ns vs control log2_OR=+0.33 FDR<1e-7), E:A (BWM log2_OR=+0.02 ns vs control log2_OR=-0.36 sig). Largest-magnitude shared-direction cells: A:N both enriched (BWM log2_OR=+0.83 / control log2_OR=+0.33, both sig) and A:W both depleted (BWM log2_OR=-0.70 / control log2_OR=-0.74, both sig — low-count flag). control,E:P at p_adj=2.27e-12 (log2_OR=-0.63) is the file minimum — extreme large-N anti-conservative case flagged via control-E-P-extreme-anticonservative. Cross-contrast hook for Chumeng: E:K direction-flip magnitude is similar to the d10_vs_d0_aa cell (BWM log2_OR=-0.34 / control log2_OR=+0.18)."
user_directives:
  - "(resume probe) order → 'Codon-first pairing (Recommended)' — d5_vs_d0_aa is the fourth file processed this session"
  - "(resume probe) layout → 'Same layout for all 5 (Recommended)'"
  - "(resume probe) caveat flow → 'If Dylan thinks the flags are appropriate then prompt me with them' — Dylan proposes per-CSV, user confirms each"
  - "(triage) test type → user confirmed Fisher's exact, BH-FDR per (condition, site) (locked from family, not re-asked)"
  - "(triage) per-CSV caveats → user confirmed all four Dylan-proposed: rare-aa-low-count, OR-direction-asymmetry, control-vs-BWM-divergent-direction, control-E-P-extreme-anticonservative"
  - "(invocation context) `read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging` — Dylan re-read; test design unchanged"
  - "(history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `control-E-P-extreme-anticonservative` caveat added. Numbers and Top hits unchanged from prior run after spot-check."
  - "(2026-05-25 Stage 4) Family-rule rewrite — Top-hits restructured from 2 per-condition tables (top-5 per direction = 10 rows per condition) to 4 per-(condition, direction) tables with `site` column; all rows with `p_adj` < 0.05 shown (BWM 6+9 / control 9+12 = 36 = `n_significant_fdr05`); effect column `odds_ratio` -> `log2_OR` across Top-hits + frontmatter `headline` + body `## Headline` + 3 caveat `why` fields (`large-N-Fisher-anticonservative`, `control-vs-BWM-divergent-direction`, `control-E-P-extreme-anticonservative`) + `OR-direction-asymmetry` ranking-axis sentence + `## For Chumeng` cross-contrast quotes; `## Methods` sentence dropped `|log2(OR)|`-ranked FDR<0.10 fallback wording and adopted the family-rule sentence; `## Numbers at a glance` per-(condition, site) bullets reordered A/E/P -> A/P/E with direction-broken hit listings matching the new sub-tables. `.qmd` is the source of truth and was not touched by this turn; Stage 4 edits Dylan-upstream only."
  - "(2026-05-28 three-section rollout) layout directive — Top-hits restructured from the 4 per-(condition, direction) sub-tables (BWM/control x Enriched/Depleted) to the 3 paired sections (Significant in both = 6 cells / BWM only = 9 / control only = 15), matching the `.qmd` Olive rebuilt the same day. Columns now `Site | Amino Acid | BWM \`log2_OR\` | control \`log2_OR\` | Effect change | Flag`; raw/adjusted-p columns dropped (significance carried by section membership); `Effect change` = BWM log2_OR - control log2_OR; rows grouped A/P/E then `Effect change` desc. Tables from `within_condition_sig_split.py` block 15 (default 100/200 thresholds). Table flag `rare-aa-low-count` -> `low-count (BWM, C)` on A:W and `low-count (C)` on control,P,M; `rare-aa flag(ged)` prose -> `low-count flag(ged)` in body Headline + frontmatter headline + `control-vs-BWM-divergent-direction` caveat why; the `rare-aa-low-count` confirmed-caveat label is kept (conceptual caveat, distinct from the table flag glyph). All 30 underlying significant cells unchanged (BWM 15 / control 21 / 36 hit-rows); Numbers-at-a-glance / Methods / For-Chumeng numerically intact."
---

# Interpretation — timepoint_fisher_within_condition_d5_vs_d0_aa

> Source: `results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_aa.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (resume probe) "Continuing the in-progress family — which order should I work the 5 remaining CSVs?" → "Codon-first pairing (Recommended)" — this file is fourth in the session.
- (resume probe) "Re-confirm the file-level layout choice for the remaining 5 files?" → "Same layout for all 5 (Recommended)"
- (resume probe) "Streamline caveat confirmation across the family?" → "If Dylan thinks the flags are appropriate then prompt me with them, I don't know myself if they are prevalent on a csv by csv basis"
- (triage) "Per-CSV caveats Dylan wants to flag for d5_vs_d0_aa. Which apply?" → user confirmed all four Dylan-proposed: `rare-aa-low-count`, `OR-direction-asymmetry`, `control-vs-BWM-divergent-direction`, `control-E-P-extreme-anticonservative`
- (invocation context) "read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging" → Dylan re-read; test design unchanged.
- (history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `control-E-P-extreme-anticonservative` caveat added. Numbers and Top hits unchanged from prior run after spot-check.
- (2026-05-25 Stage 4) Family-rule rewrite applied — see frontmatter `user_directives` audit-trail entry for the change set.
- (2026-05-28 three-section rollout) Top-hits swapped from the 4 per-(condition, direction) sub-tables to the 3 paired sections (Significant in both / BWM only / control only) matching the `.qmd`; columns `Site | Amino Acid | BWM log2_OR | control log2_OR | Effect change | Flag`, raw/adjusted-p dropped, table flag -> `low-count (BWM, C)` (A:W) / `low-count (C)` (control,P,M); `rare-aa flagged` prose -> `low-count flagged`. Underlying significant cells unchanged. See frontmatter `user_directives` for the full change set.

## Headline
d5 vs d0 within-condition Fisher (aa-level): **36/120 hits at FDR<0.05** (15/60 BWM, 21/60 control). **Largest-magnitude divergence cell**: E:K — BWM E:K log2_OR=-0.42 FDR=1.95e-9 vs control E:K log2_OR=+0.26 FDR=7.94e-7. Cross-contrast hook for Chumeng: the d10_vs_d0_aa cell (BWM log2_OR=-0.34 / control log2_OR=+0.18) carries similar direction and magnitude; d10_vs_d5 has E:K null in both conditions (BWM log2_OR=+0.08 ns / control log2_OR=-0.08 ns) — Chumeng to weigh whether the numerical sequence supports a single d0-anchored pattern vs other readings. **Largest-magnitude shared-direction cells**: A:N enriched in both (BWM log2_OR=+0.83 vs control log2_OR=+0.33, both sig); A:W depleted in both (BWM log2_OR=-0.70 vs control log2_OR=-0.74, both sig — low-count flagged); A:Y depleted in both (BWM log2_OR=-0.46 / control log2_OR=-0.40, both sig). **Other large-magnitude divergence cells**: **A:K** (BWM log2_OR=-0.17 ns vs control log2_OR=+0.33 FDR<1e-7) and **E:A** (BWM log2_OR=+0.03 ns vs control log2_OR=-0.36 FDR<6e-5).

## Top hits

All cells significant at `p_adj` < 0.05 in at least one condition are shown (family-level rule for `timepoint_fisher_within_condition`; no row cap, no raw-p fallback). The three sections split cells by which condition(s) reach significance: significant in both, BWM only, control only. Each row pairs the BWM and control `log2_OR` for one (site, amino acid) cell. `Effect change` is the BWM `log2_OR` minus the control `log2_OR`: near zero = both conditions moved the same way from day 0 to day 5 (shared timecourse drift), large magnitude = divergent trajectories (positive = BWM rose relative to control, negative = control rose relative to BWM). Within each section, rows are grouped by site in canonical A / P / E order, then sorted by `Effect change` descending. Cells significant in neither condition are omitted.

### Significant in both conditions (n = 6 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | N (Asn) | +0.830 | +0.330 | +0.500 |  |
| A | W (Trp) | -0.699 | -0.735 | +0.036 | low-count (BWM, C) |
| A | Y (Tyr) | -0.463 | -0.399 | -0.063 |  |
| P | K (Lys) | -0.333 | -0.184 | -0.149 |  |
| E | R (Arg) | -0.398 | +0.179 | -0.577 |  |
| E | K (Lys) | -0.419 | +0.256 | -0.675 |  |

### Significant in BWM only (n = 9 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | F (Phe) | +0.424 | -0.014 | +0.438 |  |
| A | D (Asp) | +0.379 | +0.049 | +0.330 |  |
| A | G (Gly) | -0.286 | -0.023 | -0.264 |  |
| P | Q (Gln) | +0.370 | -0.086 | +0.456 |  |
| P | G (Gly) | -0.285 | -0.068 | -0.217 |  |
| P | R (Arg) | -0.285 | +0.141 | -0.427 |  |
| E | Q (Gln) | +0.589 | +0.177 | +0.412 |  |
| E | S (Ser) | +0.512 | +0.128 | +0.384 |  |
| E | G (Gly) | -0.261 | +0.044 | -0.305 |  |

### Significant in control only (n = 15 site x amino acid cells)

| Site | Amino Acid | BWM `log2_OR` | control `log2_OR` | Effect change | Flag |
| --- | --- | --- | --- | --- | --- |
| A | P (Pro) | -0.058 | -0.367 | +0.309 |  |
| A | L (Leu) | -0.049 | -0.354 | +0.305 |  |
| A | A (Ala) | -0.174 | -0.288 | +0.113 |  |
| A | I (Ile) | +0.206 | +0.319 | -0.112 |  |
| A | E (Glu) | -0.010 | +0.140 | -0.149 |  |
| A | K (Lys) | -0.173 | +0.327 | -0.500 |  |
| P | P (Pro) | +0.183 | -0.302 | +0.486 |  |
| P | M (Met) | -0.138 | -0.403 | +0.265 | low-count (C) |
| P | H (His) | -0.105 | -0.332 | +0.227 |  |
| P | E (Glu) | +0.220 | +0.151 | +0.069 |  |
| P | V (Val) | -0.007 | +0.264 | -0.270 |  |
| E | P (Pro) | -0.120 | -0.628 | +0.508 |  |
| E | T (Thr) | +0.235 | -0.232 | +0.468 |  |
| E | A (Ala) | +0.023 | -0.355 | +0.378 |  |
| E | E (Glu) | +0.106 | +0.232 | -0.127 |  |

## Numbers at a glance
- `n_tests`: 120 (60 per condition)
- `n_significant` (adjusted-p < 0.05): 36 (BWM 15, control 21)
- `n_significant` (adjusted-p < 0.10): 47 (BWM 19, control 28)
- `min adjusted-p`: 2.27e-12 (control,E,P)
- Per (condition, site) at FDR<0.05:
  - BWM,A: 6 (N↑, D↑, F↑, Y↓, G↓, W↓); min p_adj = 3.27e-08
  - BWM,P: 4 (Q↑, K↓, G↓, R↓); min p_adj = 2.13e-04
  - BWM,E: 5 (Q↑, S↑, K↓, R↓, G↓); min p_adj = 1.95e-09
  - control,A: 9 (K↑, I↑, N↑, E↑, L↓, Y↓, W↓, P↓, A↓); min p_adj = 6.26e-08
  - control,P: 6 (V↑, E↑, P↓, M↓, K↓, H↓); min p_adj = 5.84e-04
  - control,E: 6 (K↑, E↑, R↑, P↓, A↓, T↓); min p_adj = 2.27e-12

## Methods
Fisher's exact (two-sided) is applied to a 2x2 of (aa_count, total - aa_count) at each timepoint within one condition. The script (`stall_sites_non_consensus_stats.py`) computes BH-FDR within each (condition, site) family of 20 tests for AA-level (so each of the 6 sub-families is corrected independently, not against the full 120). The test does **not** answer whether the same feature is changing in BWM and control in the same direction — that comparison is made by reading paired panels side-by-side (see "control-vs-BWM-divergent-direction" caveat) and is the largest-magnitude reading from this file's cells. Top hits are all rows with `p_adj` < 0.05 (family-level rule for `timepoint_fisher_within_condition`); no row cap and no raw-p fallback. Counts (`day_X_count`) and totals (`day_X_total`) are summed across replicates before the test, hence the family-level pseudoreplication caveat.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)*. (Inherited.)
- **large-N-Fisher-anticonservative** *(family-wide)*. control,E,P log2_OR=-0.63 → p_adj=2.27e-12 — the smallest p_adj in the entire family. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)*. (Inherited.)
- **within-condition-clean** *(family-wide)*. (Inherited.)
- **rare-aa-low-count** *(per-CSV)* — BWM,A,W (108 vs 89), control,A,W (93 vs 382), control,P,M (186 vs 607) all enter Top hits. Direction is reliable; magnitude unstable for these rows.
- **OR-direction-asymmetry** *(per-CSV)* — no zero-cell rows. Ranking discipline only: use log2_OR (already log-transformed; positive/negative bands symmetric by construction) rather than raw OR when ordering effects.
- **control-vs-BWM-divergent-direction** *(per-CSV)* — flagged for Chumeng's reconciliation.
  - **E:K**: BWM log2_OR=-0.42 (FDR<2e-9) vs control log2_OR=+0.26 (FDR<8e-7) — direction flip, both massively significant. Reproduces d10_vs_d0 finding.
  - **A:K**: BWM log2_OR=-0.17 (p_adj=0.056, near-sig depleted) vs control log2_OR=+0.33 (FDR<1e-7) — divergent direction, control side is decisively significant.
  - **E:A**: BWM log2_OR=+0.02 ns vs control log2_OR=-0.36 FDR<6e-5 — control-only depletion.
  - Several control-only-significant cells (E:P depleted, A:L depleted, A:I enriched, A:P depleted) where BWM is in the same direction but ns. control,P,V enriched is the one same-arm cell *not* matched at BWM — BWM,P,V log2_OR=-0.01 is essentially flat, so P:V is a control-only signal rather than a same-direction trend.
- **control-E-P-extreme-anticonservative** *(per-CSV)* — control,E:P (Pro at site E, depleted) carries p_adj=2.27e-12 — the family-wide minimum p_adj — for log2_OR=-0.63 (~35% relative depletion only). The most extreme single-feature large-N anti-conservative case in the family. Cap rhetoric on this row: rank by log2_OR and interpret this row's p magnitude as direction-only; effect size carries the biological scale. Cross-check against `between_condition_wilcoxon_aa` (E:P p_adj=0.27 ns there — the time-collapsed MW does not see this signal because it is a within-control time response, not a perturbation effect).

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: codon-level d5_vs_d0 (refinement), aa-level d10_vs_d0 / d10_vs_d5 (other contrasts).
- Open questions Chumeng should resolve:
  - **E:K cross-contrast tracking question.** The three contrasts give: d10_vs_d0_aa BWM log2_OR=-0.34 / control log2_OR=+0.18; d5_vs_d0_aa (this file) BWM log2_OR=-0.42 / control log2_OR=+0.26; d10_vs_d5_aa BWM log2_OR=+0.08 ns / control log2_OR=-0.08 ns. The numerical sequence is consistent with several patterns Chumeng should weigh — a d0-anchored divergence that persists at d10, a d0→d5 transition that plateaus, or two independent two-vs-d0 coincidences. Does `between_condition_wilcoxon_aa` show E:K? It reported p_adj=0.27 ns; if the d0-anchored reading is correct, time-pooling in MW would attenuate by averaging d0→d5 step against d5→d10 plateau. Chumeng to decide between readings.
  - **Cross-contrast A:N magnitudes** (provided for Chumeng's reconciliation, not as a synthesis claim): d10_vs_d0_aa BWM log2_OR=+0.83 sig / control log2_OR=-0.22 near-sig (p_adj=0.053); d5_vs_d0_aa (this file) BWM log2_OR=+0.83 sig / control log2_OR=+0.33 sig; d10_vs_d5_aa BWM log2_OR=+0.00 ns / control log2_OR=-0.55 sig. The trajectory diverges by d10: BWM rises at d5 and stays elevated; control rises at d5 then reverses below baseline by d10 (control,A,N: +0.33 → -0.22 net via a sig -0.55 step). Does the d5-only co-enrichment reflect a transient shared early response that only BWM locks in, or is the d5_vs_d0 control elevation a coincidence with the d5→d10 reversal? Cross-check at codon level (sister CSV) — does AAC, AAT, or both Asn codons carry the BWM-sustained signal?
  - **Site E in control has the smallest p_adj in the family** (E:P depleted at p_adj=2.27e-12). Cross-check whether `between_condition_wilcoxon_aa` E:P is anywhere near significant (collapsing time would mix d0 P-rich, d5 mid, d10 P-poor — the time-dependent depletion would attenuate). If between_condition shows E:P near-null, then control E:P depletion is a within-control time signature, not a perturbation effect.
  - **Cross-contrast A:K magnitudes** (for Chumeng's reconciliation): d10_vs_d0_aa BWM A:K log2_OR=-0.12 ns vs control A:K log2_OR=+0.19 sig (same direction, BWM ns); d5_vs_d0_aa (this file) BWM A:K log2_OR=-0.17 (p_adj=0.056) vs control A:K log2_OR=+0.33 (FDR<1e-7). Does the control A:K↑ trend from d0→d5 reproduce as a within-control time response in `per_timepoint_fisher_aa`, or does the cell read as a BWM-vs-control divergence at one or more timepoints? If yes to former → control-aligned time feature; if yes to latter → perturbation overlay. Chumeng to weigh.
