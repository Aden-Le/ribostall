---
input_csv: results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d5_aa.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), BH-FDR per (condition, site)
test_type_source: user-confirmed
n_tests: 120
n_significant_fdr05: 23
n_significant_fdr10: 27
min_p_adj: 5.80e-08
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "rare-aa-low-count", proposed_by: dylan, status: confirmed, why: "Trp/Cys carry day_X_count < 100 in BWM (W: 78 vs 108; C: 69 vs 82). Neither enters Top hits at FDR<0.05 in this contrast, but the row exists and would be flagged if it surfaced. Magnitude unstable for these rows."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "No zero-cell rows in this file; caveat preserved as a ranking-axis discipline note (use log2(OR), not OR, when ordering effects)."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Three site×aa cells flip direction: E:R (BWM `log2_OR`=+0.275 sig vs control `log2_OR`=-0.21 ns — cleanest flip in this contrast), A:A (BWM `log2_OR`=-0.18 ns vs control `log2_OR`=+0.268 sig), P:V (BWM `log2_OR`=+0.10 ns vs control `log2_OR`=-0.238 sig). The d10_vs_d0 E:K flip does NOT recur here (BWM `log2_OR`=+0.08 ns / control `log2_OR`=-0.08 ns); cross-contrast cells provided for Chumeng's reconciliation."}
  - {label: "shared-time-trajectory-A/P-Gly", proposed_by: dylan, status: confirmed, why: "Both BWM and control enrich Gly at sites A and P between d5 and d10 (BWM A:G `log2_OR`=+0.312 / P:G `log2_OR`=+0.297; control A:G `log2_OR`=+0.388 / P:G `log2_OR`=+0.430, control P:G p_adj=5.80e-08). Coherent shared-time signal — not perturbation-specific. Treat as a control-aligned time-axis feature for Chumeng, not a BWM-specific finding."}
caveats_considered: []
headline: "d10 vs d5 within-condition Fisher (aa-level): 23/120 hits at FDR<0.05 (8 BWM, 15 control). Largest-magnitude shared-direction cells: site-A G (BWM `log2_OR`=+0.31 / control `log2_OR`=+0.39, both sig) and site-P G (BWM `log2_OR`=+0.30 / control `log2_OR`=+0.43, both sig — control,P,G is the file minimum at p_adj=5.80e-08); flagged via shared-time-trajectory-A/P-Gly. Largest-magnitude divergence cells: E:R (BWM `log2_OR`=+0.28 sig vs control `log2_OR`=-0.21 ns), A:A (BWM `log2_OR`=-0.18 ns vs control `log2_OR`=+0.27 sig), P:V (BWM `log2_OR`=+0.10 ns vs control `log2_OR`=-0.24 sig). Cross-contrast note for Chumeng: E:K (the d10_vs_d0 divergence) is null in both conditions here (BWM `log2_OR`=+0.08 ns / control `log2_OR`=-0.08 ns)."
user_directives:
  - "(resume probe) order → 'Codon-first pairing (Recommended)' — d10_vs_d5_aa is the second file processed this session"
  - "(resume probe) layout → 'Same layout for all 5 (Recommended)' — both conditions in headline; Top hits split BWM vs control"
  - "(resume probe) caveat flow → 'If Dylan thinks the flags are appropriate then prompt me with them' — Dylan proposes per-CSV, user confirms each"
  - "(triage) test type → user confirmed Fisher's exact, BH-FDR per (condition, site) (locked from family, not re-asked)"
  - "(triage) per-CSV caveats → user confirmed all four Dylan-proposed: rare-aa-low-count, OR-direction-asymmetry, control-vs-BWM-divergent-direction, shared-time-trajectory-A/P-Gly"
  - "(invocation context) `read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging` — Dylan re-read; the test design is unchanged from the rest of the family"
  - "(history) Prior crashed run wrote this file via batch-propagation on 2026-04-29 21:59. This session re-triaged caveats per the user's explicit per-CSV preference; the new shared-time-trajectory-A/P-Gly caveat was added. Numbers and Top hits are unchanged from the prior run after spot-check."
  - "(2026-05-26) Olive `.qmd`-side raw-CSV validation of Bio-interp paragraph 5 (the '8-vs-15 power story' clause) inverted the paragraph's thesis after the per-condition d5+d10 marginals were verified balanced (BWM 18880, control 19965; ~1.06:1). Dylan-side: body caveat `large-N-Fisher-anticonservative` line 'pooled totals 6945 / 8788 / 11177 / 11935' rewritten to 'per-timepoint per-condition totals are in the thousands (BWM: day_10=6945, day_5=11935; control: day_10=8788, day_5=11177)' to remove the same 'pooled (day_10 + day_5)' ambiguity caught in CSV 14's 2026-05-25 audit (the four numbers are per-timepoint per-condition marginals, not summed). Dylan does not echo the 8-vs-15 power-story paragraph itself, so no other Dylan-side edit."
---

# Interpretation — timepoint_fisher_within_condition_d10_vs_d5_aa

> Source: `results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d5_aa.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (resume probe) "Continuing the in-progress family — which order should I work the 5 remaining CSVs?" → "Codon-first pairing (Recommended)" — d10_vs_d5_aa is the second file processed this session.
- (resume probe) "Re-confirm the file-level layout choice for the remaining 5 files?" → "Same layout for all 5 (Recommended)"
- (resume probe) "Streamline caveat confirmation across the family?" → "If Dylan thinks the flags are appropriate then prompt me with them, I don't know myself if they are prevalent on a csv by csv basis"
- (triage) "Per-CSV caveats Dylan wants to flag for d10_vs_d5_aa. Which apply?" → user confirmed all four Dylan-proposed: `rare-aa-low-count`, `OR-direction-asymmetry`, `control-vs-BWM-divergent-direction`, `shared-time-trajectory-A/P-Gly`
- (invocation context) "read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging" → Dylan re-read; the test design is unchanged from the rest of the family.
- (history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per the user's explicit per-CSV preference; the new `shared-time-trajectory-A/P-Gly` caveat was added. Numbers and Top hits are unchanged from the prior run after spot-check.
- (2026-05-26) Olive `.qmd`-side raw-CSV validation of Bio-interp paragraph 5 (the "8-vs-15 power story" clause) inverted the paragraph's thesis after the per-condition d5+d10 marginals were verified balanced (BWM 18880, control 19965; ~1.06:1). Dylan-side: body caveat `large-N-Fisher-anticonservative` line `'pooled totals 6945 / 8788 / 11177 / 11935'` rewritten to `'per-timepoint per-condition totals are in the thousands (BWM: day_10=6945, day_5=11935; control: day_10=8788, day_5=11177)'` to remove the same "pooled (day_10 + day_5)" ambiguity caught in CSV 14's 2026-05-25 audit (the four numbers are per-timepoint per-condition marginals, not summed). Dylan does not echo the 8-vs-15 power-story paragraph itself, so no other Dylan-side edit.

## Headline
d10 vs d5 within-condition Fisher (aa-level): 23/120 hits at FDR<0.05 (8/60 BWM, 15/60 control). The largest-magnitude shared-direction cells in this CSV are **site-A and site-P glycine enrichment** (BWM A:G `log2_OR`=+0.31 / P:G `log2_OR`=+0.30 vs control A:G `log2_OR`=+0.39 / P:G `log2_OR`=+0.43; control,P,G is the file minimum at p_adj=5.80e-08). BWM also shows site-A L↓ / Q↓ / R↑ and site-E G/R↑ / I↓; control adds a strong site-A multi-feature signature (N↓, Q↓, P↑, A↑, E↓, R↑) and site-E A↑ / P↑ plus site-P I/L/V/Q↓ and D↑. **Three direction-divergent cells**: E:R (BWM `log2_OR`=+0.28 FDR<0.05 vs control `log2_OR`=-0.21 ns), A:A (BWM `log2_OR`=-0.18 ns vs control `log2_OR`=+0.27 FDR<0.05), P:V (BWM `log2_OR`=+0.10 ns vs control `log2_OR`=-0.24 FDR<0.05). **E:K is null in both conditions in this contrast** (BWM E:K `log2_OR`=+0.08 ns / control E:K `log2_OR`=-0.08 ns) — provided for Chumeng to weigh against the d10_vs_d0 E:K cell.

## Top hits

All rows with `p_adj` < 0.05 are shown per the `timepoint_fisher_within_condition` family rule; no row cap and no raw-p fallback. Rows are grouped by site in A / P / E order within each direction, then sorted by `p_adj` ascending with `|log2_OR|` descending as the tie-breaker.

### BWM (n_sig FDR<0.05 = 8 / 60)

| direction | feature | effect (`log2_OR`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | A:G (Gly) | +0.312 | 9.09e-04 |  |
| ↑ | A:R (Arg) | +0.222 | 0.0496 |  |
| ↑ | P:G (Gly) | +0.297 | 0.00303 |  |
| ↑ | E:G (Gly) | +0.279 | 0.0246 |  |
| ↑ | E:R (Arg) | +0.275 | 0.0279 |  |
| ↓ | A:L (Leu) | -0.399 | 1.95e-04 |  |
| ↓ | A:Q (Gln) | -0.309 | 0.0496 |  |
| ↓ | E:I (Ile) | -0.267 | 0.0312 |  |

### control (n_sig FDR<0.05 = 15 / 60)

| direction | feature | effect (`log2_OR`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | A:G (Gly) | +0.388 | 3.96e-06 |  |
| ↑ | A:P (Pro) | +0.365 | 0.00263 |  |
| ↑ | A:A (Ala) | +0.268 | 0.0166 |  |
| ↑ | A:R (Arg) | +0.206 | 0.0377 |  |
| ↑ | P:G (Gly) | +0.430 | 5.80e-08 |  |
| ↑ | P:D (Asp) | +0.199 | 0.0232 |  |
| ↑ | E:A (Ala) | +0.397 | 0.00133 |  |
| ↑ | E:P (Pro) | +0.414 | 0.00159 |  |
| ↓ | A:N (Asn) | -0.552 | 3.96e-06 |  |
| ↓ | A:Q (Gln) | -0.534 | 6.70e-05 |  |
| ↓ | A:E (Glu) | -0.199 | 0.0166 |  |
| ↓ | P:I (Ile) | -0.382 | 1.29e-04 |  |
| ↓ | P:L (Leu) | -0.246 | 0.0211 |  |
| ↓ | P:V (Val) | -0.238 | 0.0211 |  |
| ↓ | P:Q (Gln) | -0.352 | 0.0232 |  |

## Numbers at a glance
- `n_tests`: 120 (60 per condition)
- `n_significant` (adjusted-p < 0.05): 23 (BWM 8, control 15)
- `n_significant` (adjusted-p < 0.10): 27 (BWM 8, control 19)
- `min adjusted-p`: 3.96e-06 (tied: control,A,G and control,A,N within their family); 5.80e-08 (control,P,G, the file minimum)
- Per (condition, site) at FDR<0.05:
  - BWM,A: 4 (G, R, L, Q); min p_adj = 1.95e-04
  - BWM,P: 1 (G); min p_adj = 0.00303
  - BWM,E: 3 (G, R, I); min p_adj = 0.0246
  - control,A: 7 (G, N, Q, P, A, E, R); min p_adj = 3.96e-06
  - control,P: 6 (G, I, L, V, D, Q); min p_adj = 5.80e-08
  - control,E: 2 (A, P); min p_adj = 0.00133

## Methods
Same as the rest of the family. Dylan proposed Fisher's exact (two-sided) per (condition, site), BH-FDR within each 20-row sub-family; user confirmed via family-level propagation. Top hits are all rows with `p_adj` < 0.05 (family-level rule for `timepoint_fisher_within_condition`); no row cap and no raw-p fallback.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* — replicates summed; p anti-conservative. (Inherited.)
- **large-N-Fisher-anticonservative** *(family-wide)* — per-timepoint per-condition totals are in the thousands (BWM: day_10=6945, day_5=11935; control: day_10=8788, day_5=11177). Tiny effects yield p_adj < 1e-7 (control,P,G `log2_OR`=+0.430 → 5.80e-08). Rank by `log2_OR`. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* — 6 sub-families of 20 corrected independently. (Inherited.)
- **within-condition-clean** *(family-wide)* — no condition pooling. (Inherited.)
- **rare-aa-low-count** *(per-CSV)* — Trp (BWM W: 78 vs 108) and Cys (BWM C: 69 vs 82, control C: 61 vs 89) sit below count=100. None enters Top hits in this contrast, so the caveat carries no Top-hits flags here, but applies to any downstream use of the file.
- **OR-direction-asymmetry** *(per-CSV)* — no zero-cell rows; caveat preserved as ranking discipline (`log2_OR` is the reported effect column, so positive/negative bands are symmetric by construction; rank directly on it).
- **control-vs-BWM-divergent-direction** *(per-CSV)* — three direction-divergent cells: **E:R** (BWM `log2_OR`=+0.275 FDR=0.028 vs control `log2_OR`=-0.21 p_adj=0.21 ns; cleanest flip in the contrast); **A:A** (BWM `log2_OR`=-0.18 ns vs control `log2_OR`=+0.268 FDR=0.017); **P:V** (BWM `log2_OR`=+0.10 ns vs control `log2_OR`=-0.238 FDR=0.021). Several other cells where one condition is sig and the other ns at the same direction (e.g. P:I — both depleted but only control reaches FDR<0.05). The d10_vs_d0 E:K flip does not recur here (both ns) — anchoring that flip at d0.
- **shared-time-trajectory-A/P-Gly** *(per-CSV)* — Both BWM and control enrich Gly at sites A and P from d5→d10 (BWM A:G `log2_OR`=+0.312 p_adj=9.09e-04 / P:G `log2_OR`=+0.297 p_adj=0.00303; control A:G `log2_OR`=+0.388 p_adj=3.96e-06 / P:G `log2_OR`=+0.430 p_adj=5.80e-08 — control,P,G is the file minimum). Coherent shared-time signal across both conditions — not perturbation-specific. Chumeng should treat A/P-Gly enrichment as a time-axis feature, not a BWM-vs-control contrast finding.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: codon-level d10_vs_d5 plus the d10_vs_d0 / d5_vs_d0 contrasts at aa resolution.
- Open questions Chumeng should resolve:
  - **Cross-contrast E:K magnitudes** (provided for Chumeng's reconciliation, not as a synthesis claim). d10_vs_d0_aa: BWM E:K `log2_OR`=-0.34 sig vs control E:K `log2_OR`=+0.18 sig (direction flip). d10_vs_d5_aa (this file): BWM E:K `log2_OR`=+0.08 ns vs control E:K `log2_OR`=-0.08 ns (no flip, both flat). Does this numerical pattern support a single d0-anchored E:K divergence with a d5→d10 plateau, or could the d10_vs_d0 flip be carried by the d0→d5 step alone (with d5_vs_d0_aa as the deciding cell)? Chumeng to weigh.
  - **Per-cell observations on site-A/P glycine across contrasts in this family** (cross-contrast magnitudes provided for Chumeng's reconciliation, not as a synthesis claim): d10_vs_d0_aa BWM P:G ns; control A:G `log2_OR`=+0.37, P:G `log2_OR`=+0.36. d10_vs_d5_aa (this file): BWM A:G `log2_OR`=+0.312, P:G `log2_OR`=+0.297, E:G `log2_OR`=+0.279; control A:G `log2_OR`=+0.388, P:G `log2_OR`=+0.430. Does Gly enrichment at sites A and P track with time in both conditions, or is the d5→d10 step the carrier? Cross-check against `between_condition_wilcoxon_aa` (collapsed time, 0 aa hits at FDR<0.05) — would a real both-condition time-axis Gly signal appear there as a near-nominal trend, or would time-pooling cancel it?
  - **Does the codon-level d10_vs_d5 file confirm Gly enrichment is shared across all 4 Gly codons (GGA/GGC/GGG/GGT) or driven by one synonym?** This determines whether it is an amino-acid-level shift or a codon-usage shift.
